"""Auditable planning and tool routing for the local multimodal planner Agent.

The module exposes concise decision summaries and tool calls.  It deliberately
does not persist or return hidden model reasoning.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from routing.app.backend.planner_agent_tools import (
    PlannerAgentTool, PlannerToolRegistry, tool_result,
)


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "routing/configs/planner_agent_tools.json"


def load_agent_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_planner_intent(text: str) -> dict[str, Any]:
    normalized = " ".join(text.strip().split())[:500]
    wants_facility = any(word in normalized for word in ("遮阳棚", "雨棚", "人工遮阳", "廊架"))
    wants_tree = any(word in normalized for word in ("乔木", "树冠", "林荫", "种树", "树木"))
    facility_first = any(word in normalized for word in ("人工遮阳优先", "遮阳棚优先", "只用遮阳棚", "不种树"))
    assess_only = any(word in normalized for word in ("只评估", "不要生成", "仅分析"))
    constraints: list[str] = []
    lexicon = {
        "道路南侧": ("南侧", "道路南"), "道路北侧": ("北侧", "道路北"),
        "保持机动车道": ("不占机动车道", "保留车道", "保持车道"),
        "连续树冠": ("连续", "连片", "林荫带"),
        "高大乔木": ("高大", "乔木", "大树冠"),
        "优先人行道": ("人行道", "步行"), "兼顾骑行": ("骑行", "自行车"),
    }
    for label, words in lexicon.items():
        if any(word in normalized for word in words):
            constraints.append(label)
    return {
        "raw_intent": normalized,
        "objective": "assess_only" if assess_only else "shade_intervention",
        "preferred_intervention": "facility_supplement" if facility_first or (wants_facility and not wants_tree) else "tree_first",
        "constraints": constraints or ["连续乔木优先", "保持道路与建筑结构"],
        "output_contract": ["evidence_overlay", "auditable_tool_trace", "automatic_quality_verification"],
    }


def discover_tools() -> dict[str, dict[str, Any]]:
    cfg = load_agent_config()
    paths = {
        "road_space_analysis": ROOT / "routing/app/backend/planner_analysis_worker.py",
        "phenology_review": ROOT / "routing/app/backend/planner_vlm_agent_worker.py",
        "tree_gap_analysis": ROOT / "models/Qwen_Qwen3-VL-2B-Instruct",
        "directional_windows": ROOT / "streetscape/scripts/stage_44_panorama_perspective_windows.py",
        "tree_layout": ROOT / "streetscape/scripts/stage_45_tree_first_semantic_25d_layout.py",
        "structural_gate": ROOT / "streetscape/scripts/stage_45b_multimodal_structural_scene_gate.py",
        "local_generation": ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/powerpaint-v1",
        "planning_overlay": ROOT / "routing/app/backend/planner_analysis_worker.py",
        "generation_acceptance": ROOT / "streetscape/scripts/stage_47_automatic_generation_acceptance.py",
    }
    return {
        item["name"]: {
            **item,
            "available": paths[item["name"]].exists(),
            "local_path_checked": str(paths[item["name"]].relative_to(ROOT)),
        }
        for item in cfg["tools"]
    }


def build_agent_plan(intent: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    tools = discover_tools()
    motor_only = bool(semantic.get("high_speed_road_suspected"))
    shade_need = str(semantic.get("shade_optimization_need", "需复核"))
    leaf_off = bool(semantic.get("leaf_off_risk") or semantic.get("march_phenology_uncertain"))
    panorama = bool(semantic.get("panorama_detected"))
    requested = intent["preferred_intervention"]
    if intent["objective"] == "assess_only" or shade_need == "通常不需要":
        action = "assess_only"
        reason = "用户仅要求评估，或SVF表明无需新增遮荫。"
    elif motor_only:
        action = "motor_only"
        reason = "先保留遮荫诊断，再将机动车专用属性作为概率性空间类型输出。"
    elif leaf_off:
        action = "leaf_off"
        reason = "落叶期低绿量不能解释为缺树，优先显示现有树列生长目标。"
    elif requested == "facility_supplement":
        action = "tree_first_then_facility_review"
        reason = "尊重自然语言需求，但仍执行项目既定的乔木优先约束。"
    elif panorama and bool(semantic.get("automatic_generation_candidate_available")):
        action = "tree_generation_with_acceptance"
        reason = "确定性空间、物候、遮荫和几何前置条件支持进入布局与生成验收链。"
    elif panorama and bool(semantic.get("directional_spatial_anchor_available")):
        action = "tree_layout"
        reason = "方向空间锚点支持2.5D布局，但生成前置条件尚未全部满足。"
    elif panorama:
        action = "directional_source_required"
        reason = "全景图仅作为方向结果融合总览，不再直接承担方案推断或生成。"
    else:
        action = "directional_perspective_plan"
        reason = "单方位透视图是正式判断单位；先排除道路尽头，再在左右两侧生成可审计遮荫方案。"
    selected = ["planning_overlay"]
    if action == "leaf_off":
        selected.insert(0, "phenology_review")
    if action in {"directional_source_required", "directional_perspective_plan", "tree_layout", "tree_generation_with_acceptance"}:
        selected.insert(0, "tree_gap_analysis")
    if action in {"tree_layout", "tree_generation_with_acceptance"}:
        selected += ["directional_windows", "tree_layout", "structural_gate"]
    if action == "tree_generation_with_acceptance":
        selected += ["local_generation", "generation_acceptance"]
    return {
        "controller": "bounded_observe_plan_act_verify_replan",
        "city_profile": "nanjing",
        "intent": intent,
        "selected_action": action,
        "decision_summary": reason,
        "selected_tools": selected,
        "tool_availability": {name: tools[name]["available"] for name in selected},
        "max_replans": int(load_agent_config()["max_replans"]),
        "hidden_reasoning_exposed": False,
    }


class PlannerAgentRuntime:
    """Deterministic observe/plan/act/verify loop over local tools."""

    def __init__(
        self, registry: PlannerToolRegistry, max_steps: int = 8,
        timeout_seconds: int = 1800,
    ) -> None:
        self.registry = registry
        self.max_steps = max(2, min(8, int(max_steps)))
        self.timeout_seconds = max(1, int(timeout_seconds))

    @staticmethod
    def _compact(values: dict[str, Any]) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: clean(item) for key, item in value.items() if "base64" not in key.lower()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return None if isinstance(value, (bytes, bytearray)) else value
        return clean(values)

    @staticmethod
    def _hard_gates(observations: dict[str, Any]) -> dict[str, bool]:
        return {
            "space_type_not_motor_only": not bool(observations.get("high_speed_road_suspected")),
            "walkable_space": observations.get("walkable_space_confirmed") is True,
            "non_winter_phenology": not bool(observations.get("leaf_off_risk") or observations.get("march_phenology_uncertain")),
            "svf_need": bool(observations.get("optimization_eligible")),
            "confidence_for_formal_generation": int(observations.get("planning_confidence", 65)) >= 50,
            "panorama_geometry": bool(observations.get("panorama_detected")),
            "ground_street_consensus": observations.get("structural_scene") == "GROUND_STREET",
            "tree_gap_consensus": observations.get("tree_state") == "TREE_GAP_WITH_PLANTING_SPACE",
            "layout_ready": bool(observations.get("layout_ready")),
            "structural_gate": observations.get("structural_gate_pass") is True,
            "local_generation_available": observations.get("local_generation_available") is True,
        }

    def _choose_next(self, state: dict[str, Any]) -> tuple[str | None, str]:
        obs = state["observations"]
        done = set(state["completed_tools"])
        attempted = {item["tool"] for item in state["tool_results"]}
        failed = {item["tool"] for item in state["tool_results"] if item["status"] in {"FAIL", "UNCERTAIN", "BLOCKED"}}
        objective = state["goal"]
        selected_action = state["selected_action"]

        if objective == "validate_generation":
            if "generation_acceptance" not in attempted:
                return "generation_acceptance", "human_imported_candidate_requires_local_acceptance"
            if obs.get("automatic_acceptance"):
                state.update(final_action="generation_accepted", stop_reason="automatic_acceptance_passed")
                return None, "accepted"
            state.update(final_action="planning_overlay", stop_reason="automatic_acceptance_rejected")
            return (None, "rejection_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "acceptance_rejected_fallback")
        if objective == "assess_only" or str(obs.get("shade_optimization_need")) == "通常不需要":
            state.update(final_action="assess_only", stop_reason="assessment_only")
            return (None, "assessment_complete") if "planning_overlay" in done else ("planning_overlay", "assessment_requested")
        if obs.get("high_speed_road_suspected"):
            state.update(final_action="motor_only", stop_reason="motor_vehicle_only_hard_gate")
            return (None, "motor_only_stop") if "planning_overlay" in done else ("planning_overlay", "retain_safe_evidence")
        if obs.get("leaf_off_risk") or obs.get("march_phenology_uncertain"):
            if "phenology_review" not in attempted:
                return "phenology_review", "phenology_requires_review"
            state.update(final_action="planner_review", stop_reason="leaf_off_or_phenology_uncertain")
            return (None, "phenology_safe_stop") if "planning_overlay" in done else ("planning_overlay", "retain_review_overlay")

        if obs.get("walkable_space_confirmed") is not True and "road_space_analysis" not in attempted:
            return "road_space_analysis", "walkable_space_not_confirmed"
        if obs.get("walkable_space_confirmed") is not True:
            state.update(final_action="planner_review", stop_reason="walkable_space_not_confirmed")
            return (None, "space_safe_stop") if "planning_overlay" in done else ("planning_overlay", "retain_review_overlay")
        if not obs.get("optimization_eligible"):
            state.update(final_action="directional_perspective_plan", stop_reason="generation_not_requested_or_not_eligible")
            return (None, "directional_plan_complete") if "planning_overlay" in done else ("planning_overlay", "non_generative_plan")
        if selected_action not in {"tree_layout", "tree_generation_with_acceptance"}:
            state.update(
                final_action="planner_review" if selected_action in {"directional_source_required", "tree_first_then_facility_review"} else "directional_perspective_plan",
                stop_reason="selected_action_is_non_generative",
            )
            return (None, "non_generative_action_complete") if "planning_overlay" in done else ("planning_overlay", "selected_non_generative_action")
        if "tree_gap_analysis" not in done and "tree_gap_analysis" not in failed:
            return "tree_gap_analysis", "tree_gap_evidence_needed"
        if obs.get("structural_scene") != "GROUND_STREET" or obs.get("tree_state") != "TREE_GAP_WITH_PLANTING_SPACE":
            state.update(final_action="planner_review", stop_reason="tree_or_structural_evidence_unreliable")
            return (None, "tree_evidence_safe_stop") if "planning_overlay" in done else ("planning_overlay", "fallback_to_overlay")
        if not obs.get("panorama_detected"):
            state.update(final_action="directional_perspective_plan", stop_reason="directional_source_has_no_panorama_geometry")
            return (None, "geometry_safe_stop") if "planning_overlay" in done else ("planning_overlay", "fallback_to_directional_overlay")
        if "directional_windows" not in done:
            return "directional_windows", "layout_requires_directional_windows"
        if "tree_layout" not in done and "tree_layout" not in failed:
            return "tree_layout", "tree_gap_requires_layout"
        if not obs.get("layout_ready"):
            state.update(final_action="planning_overlay", stop_reason="layout_failed")
            return (None, "layout_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "layout_failed_fallback")
        if "structural_gate" not in done and "structural_gate" not in failed:
            return "structural_gate", "layout_requires_structural_verification"
        if obs.get("structural_gate_pass") is not True:
            state.update(final_action="planning_overlay", stop_reason="structural_gate_failed")
            return (None, "structural_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "structural_gate_fallback")
        if selected_action == "tree_layout":
            state.update(final_action="planning_overlay", stop_reason="tree_layout_complete_generation_not_unlocked")
            return (None, "layout_plan_complete") if "planning_overlay" in done else ("planning_overlay", "publish_layout_overlay")
        gates = self._hard_gates(obs)
        state["generation_gates"] = gates
        state["generation_unlocked"] = all(gates.values())
        if not state["generation_unlocked"]:
            state.update(final_action="planning_overlay", stop_reason="generation_hard_gates_failed")
            return (None, "hard_gate_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "hard_gate_fallback")
        if "local_generation" not in done and "local_generation" not in failed:
            return "local_generation", "all_generation_gates_passed"
        if not obs.get("generation_succeeded"):
            state.update(final_action="planning_overlay", stop_reason="generation_failed")
            return (None, "generation_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "generation_failed_fallback")
        if "generation_acceptance" not in done and "generation_acceptance" not in failed:
            return "generation_acceptance", "generated_result_requires_acceptance"
        if obs.get("automatic_acceptance"):
            state.update(final_action="generation_accepted", stop_reason="automatic_acceptance_passed")
            return None, "accepted"
        state.update(final_action="planning_overlay", stop_reason="automatic_acceptance_rejected")
        return (None, "rejection_fallback_complete") if "planning_overlay" in done else ("planning_overlay", "acceptance_rejected_fallback")

    def run(self, task: dict[str, Any]) -> dict[str, Any]:
        started = time.monotonic()
        observations = dict(task.get("observations") or {})
        observations.setdefault("walkable_space_confirmed", self._initial_walkable(observations))
        state: dict[str, Any] = {
            "task_id": str(task.get("task_id") or uuid.uuid4().hex),
            "goal": str(task.get("goal") or "shade_intervention"),
            "constraints": list(task.get("constraints") or []),
            "observations": observations, "completed_tools": [], "pending_tools": [],
            "visited_tools": {}, "tool_results": [], "current_state": "OBSERVING", "generation_unlocked": False,
            "generation_gates": {}, "final_action": "",
            "selected_action": str(task.get("selected_action") or self._initial_action(str(task.get("goal") or "shade_intervention"), observations)),
            "stop_reason": "", "warnings": [], "artifacts": {},
        }
        trace = [{
            "step": 1, "phase": "observe", "tool": "existing_semantic_analysis", "status": "PASS",
            "decision_summary": "reuse_existing_semantic_observation",
            "input_summary": {}, "observations": self._compact(observations), "artifacts": {},
            "warnings": [], "error": None, "agent_state": "OBSERVING", "generation_unlocked": False,
        }]
        visits: dict[str, int] = state["visited_tools"]
        while len(trace) < self.max_steps and time.monotonic() - started < self.timeout_seconds:
            state["current_state"] = "PLANNING"
            retry_tool = state.pop("retry_tool", None)
            next_tool, summary = (retry_tool, "bounded_tool_retry") if retry_tool else self._choose_next(state)
            state["pending_tools"] = [next_tool] if next_tool else []
            if next_tool is None:
                break
            visits[next_tool] = visits.get(next_tool, 0) + 1
            result = self.registry.execute(next_tool, state)
            retry = result["status"] == "FAIL" and self.registry.allows_retry(next_tool) and visits[next_tool] < 2
            if result["status"] == "PASS":
                state["completed_tools"].append(next_tool)
            state["observations"].update(result["observations"])
            state["artifacts"].update(result["artifacts"])
            state["tool_results"].append(result)
            state["warnings"].extend(result["warnings"])
            state["current_state"] = "REPLANNING" if result["status"] != "PASS" else "VERIFYING"
            trace.append({
                "step": len(trace) + 1, "phase": "replan" if result["status"] != "PASS" else "act_verify",
                "tool": next_tool, "status": result["status"], "decision_summary": summary,
                "input_summary": {"attempt": visits[next_tool]},
                "observations": self._compact(result["observations"]),
                "artifacts": self._compact(result["artifacts"]), "warnings": result["warnings"],
                "error": result["error"], "agent_state": state["current_state"],
                "generation_unlocked": bool(state["generation_unlocked"]),
            })
            if retry:
                state["retry_tool"] = next_tool
                continue
        if not state["final_action"]:
            next_tool, _ = self._choose_next(state)
            state["pending_tools"] = [next_tool] if next_tool else []
            if next_tool is not None:
                state.update(final_action="planning_overlay", stop_reason="max_steps_or_timeout")
        else:
            state["pending_tools"] = []
        state["current_state"] = "COMPLETED" if state["final_action"] == "generation_accepted" else "SAFE_STOP"
        return {
            "task_id": state["task_id"], "final_state": state["current_state"],
            "agent_state": state["current_state"], "selected_action": state["selected_action"],
            "final_action": state["final_action"], "stop_reason": state["stop_reason"],
            "next_tool": state["pending_tools"][0] if state["pending_tools"] else "none",
            "generation_unlocked": bool(state["generation_unlocked"]),
            "generation_available": state["final_action"] == "generation_accepted",
            "automatic_acceptance": bool(state["observations"].get("automatic_acceptance")),
            "planner_review_required": state["final_action"] in {"planner_review", "planning_overlay"},
            "quality_metrics": state["observations"].get("acceptance_metrics") or {},
            "tool_trace": trace, "artifacts": state["artifacts"], "warnings": state["warnings"],
            "completed_tools": state["completed_tools"], "visited_tools": state["visited_tools"],
            "generation_gates": state["generation_gates"],
        }

    @staticmethod
    def _initial_walkable(observations: dict[str, Any]) -> bool | None:
        if observations.get("high_speed_road_suspected"):
            return False
        likelihood = str(observations.get("public_walk_cycle_likelihood", ""))
        if likelihood in {"较高", "high"}:
            return True
        return None

    @staticmethod
    def _initial_action(goal: str, observations: dict[str, Any]) -> str:
        if goal == "validate_generation":
            return "generation_acceptance"
        if goal == "assess_only" or str(observations.get("shade_optimization_need")) == "通常不需要":
            return "assess_only"
        if observations.get("high_speed_road_suspected"):
            return "motor_only"
        if observations.get("leaf_off_risk") or observations.get("march_phenology_uncertain"):
            return "leaf_off"
        if observations.get("panorama_detected") and observations.get("optimization_eligible"):
            return "tree_generation_with_acceptance"
        return "directional_perspective_plan"


def build_assessment_agent_result(observations: dict[str, Any], task_id: str) -> dict[str, Any]:
    """Add the bounded contract to existing non-generative analysis endpoints."""
    overlay = PlannerAgentTool(
        "planning_overlay", "Retain the existing planning overlay.", {}, {},
        lambda _state: tool_result("planning_overlay", "PASS", artifacts={
            "overlay_available": bool(observations.get("overlay_png_base64") or observations.get("fused_panorama_png_base64")),
        }),
    )
    return PlannerAgentRuntime(PlannerToolRegistry([overlay]), max_steps=2).run({
        "task_id": task_id, "goal": "assess_only", "selected_action": "assess_only",
        "observations": observations,
    })
