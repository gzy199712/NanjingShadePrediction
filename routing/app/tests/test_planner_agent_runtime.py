"""Unit checks for bounded dynamic planner tool routing (no GPU models)."""

import json
from collections import Counter
from pathlib import Path

import pytest

from routing.app.backend.planner_agent_orchestrator import PlannerAgentRuntime, build_agent_plan, parse_planner_intent
from routing.app.backend.planner_agent_tools import PlannerAgentTool, PlannerToolRegistry, tool_result


SCHEMA = {"type": "object"}


def tool(name, observations=None, status="PASS", *, retry=False, calls=None):
    def run(_state):
        if calls is not None:
            calls.append(name)
        return tool_result(name, status, observations=observations or {})
    return PlannerAgentTool(name, name, SCHEMA, SCHEMA, run, retry_on_failure=retry)


def registry(*, layout="PASS", structural="PASS", accepted=True, generation="PASS", calls=None):
    return PlannerToolRegistry([
        tool("road_space_analysis", {"walkable_space_confirmed": True}, calls=calls),
        tool("phenology_review", {"phenology_reviewed": True}, calls=calls),
        tool("tree_gap_analysis", {
            "structural_scene": "GROUND_STREET", "tree_state": "TREE_GAP_WITH_PLANTING_SPACE",
        }, calls=calls),
        tool("directional_windows", {"directional_windows_ready": True}, calls=calls),
        tool("tree_layout", {"layout_ready": layout == "PASS"}, layout, calls=calls),
        tool("structural_gate", {"structural_gate_pass": structural == "PASS"}, structural, calls=calls),
        PlannerAgentTool(
            "local_generation", "generation", SCHEMA, SCHEMA,
            lambda _state: tool_result("local_generation", generation, observations={"generation_succeeded": generation == "PASS"}),
            retry_on_failure=True, requires_generation_unlock=True,
        ),
        PlannerAgentTool(
            "generation_acceptance", "acceptance", SCHEMA, SCHEMA,
            lambda _state: tool_result("generation_acceptance", "PASS" if accepted else "FAIL", observations={"automatic_acceptance": accepted}),
        ),
        tool("planning_overlay", calls=calls),
    ])


def observations(**changes):
    base = {
        "public_walk_cycle_likelihood": "较高", "high_speed_road_suspected": False,
        "leaf_off_risk": False, "march_phenology_uncertain": False,
        "shade_optimization_need": "较高", "optimization_eligible": True,
        "planning_confidence": 65, "panorama_detected": True,
        "local_generation_available": True,
    }
    base.update(changes)
    return base


def run(obs=None, *, goal="shade_intervention", tools=None):
    return PlannerAgentRuntime(tools or registry()).run({"goal": goal, "observations": obs or observations()})


@pytest.mark.parametrize("goal,changes,action", [
    ("assess_only", {}, "assess_only"),
    ("shade_intervention", {"high_speed_road_suspected": True}, "motor_only"),
    ("shade_intervention", {"leaf_off_risk": True}, "leaf_off"),
])
def test_safe_routes_finish_without_generation(goal, changes, action):
    result = run(observations(**changes), goal=goal)
    assert 2 <= len(result["tool_trace"]) <= 8
    assert result["selected_action"] == action
    assert "local_generation" not in result["completed_tools"]
    assert result["generation_unlocked"] is False


def test_layout_failure_falls_back_to_overlay():
    result = run(tools=registry(layout="FAIL"))
    assert result["selected_action"] == "tree_generation_with_acceptance"
    assert result["final_action"] == "planning_overlay"
    assert result["stop_reason"] == "layout_failed"
    assert "planning_overlay" in result["completed_tools"]


def test_structural_failure_never_unlocks_generation():
    result = run(tools=registry(structural="FAIL"))
    assert result["stop_reason"] == "structural_gate_failed"
    assert result["generation_unlocked"] is False
    assert "local_generation" not in result["completed_tools"]


def test_missing_local_generator_still_returns_overlay():
    result = run(observations(local_generation_available=False))
    assert result["final_action"] == "planning_overlay"
    assert result["generation_unlocked"] is False
    assert result["planner_review_required"] is True
    assert result["generation_gates"]["local_generation_available"] is False
    assert "planning_overlay" in result["completed_tools"]


@pytest.mark.parametrize("accepted,final_action,available", [
    (True, "generation_accepted", True),
    (False, "planning_overlay", False),
])
def test_generation_acceptance_and_rejection(accepted, final_action, available):
    result = run(tools=registry(accepted=accepted))
    assert result["selected_action"] == "tree_generation_with_acceptance"
    assert result["final_action"] == final_action
    assert result["generation_available"] is available
    assert result["generation_unlocked"] is True
    generation_step = next(item for item in result["tool_trace"] if item["tool"] == "local_generation")
    assert generation_step["generation_unlocked"] is True
    if not accepted:
        assert "planning_overlay" in result["completed_tools"]


def test_generation_tool_is_blocked_without_unlock():
    result = registry().execute("local_generation", {"generation_unlocked": False})
    assert result["status"] == "BLOCKED"
    assert result["error"] == "generation_hard_gates_not_unlocked"


def test_phenology_and_road_tools_are_selected_only_when_needed():
    phenology = run(observations(march_phenology_uncertain=True))
    assert phenology["tool_trace"][1]["tool"] == "phenology_review"
    road = run(observations(public_walk_cycle_likelihood="可能"))
    assert road["tool_trace"][1]["tool"] == "road_space_analysis"


def test_human_imported_candidate_is_locally_accepted_or_hidden():
    rejected = PlannerAgentRuntime(registry(accepted=False), max_steps=3).run({
        "goal": "validate_generation", "observations": {"candidate_source": "human_imported_external_image"},
    })
    assert rejected["selected_action"] == "generation_acceptance"
    assert rejected["final_action"] == "planning_overlay"
    assert rejected["generation_available"] is False
    assert "generated_panorama_png_base64" not in rejected


def test_failure_replans_once_and_never_repeats_more_than_twice():
    calls = []
    attempts = {"count": 0}
    tools = registry(calls=calls)
    def flaky(_state):
        attempts["count"] += 1
        return tool_result(
            "tree_gap_analysis", "FAIL" if attempts["count"] == 1 else "PASS",
            observations={} if attempts["count"] == 1 else {
                "structural_scene": "GROUND_STREET", "tree_state": "TREE_GAP_WITH_PLANTING_SPACE",
            },
        )
    tools._tools["tree_gap_analysis"] = PlannerAgentTool(
        "tree_gap_analysis", "flaky", SCHEMA, SCHEMA, flaky, retry_on_failure=True,
    )
    result = run(tools=tools)
    counts = Counter(item["tool"] for item in result["tool_trace"])
    assert counts["tree_gap_analysis"] == 2
    assert max(counts.values()) <= 2
    assert any(item["phase"] == "replan" for item in result["tool_trace"])


def test_runtime_trace_matches_schema_and_omits_base64():
    jsonschema = pytest.importorskip("jsonschema")
    result = run(observations(nested={"preview_base64": "must-not-leak"}))
    root = Path(__file__).resolve().parents[3]
    schema = json.loads((root / "streetscape/schemas/planner_agent_runtime_trace.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    assert "base64" not in json.dumps(result["tool_trace"]).lower()


def test_build_plan_action_contract_has_reachable_generation():
    intent = parse_planner_intent("沿道路增加连续乔木")
    assert build_agent_plan(intent, observations(panorama_detected=False, optimization_eligible=False))["selected_action"] == "directional_perspective_plan"
    layout = build_agent_plan(intent, observations(directional_spatial_anchor_available=True))
    assert layout["selected_action"] == "tree_layout"
    assert "local_generation" not in layout["selected_tools"]
    assert build_agent_plan(intent, observations(automatic_generation_candidate_available=True))["selected_action"] == "tree_generation_with_acceptance"
