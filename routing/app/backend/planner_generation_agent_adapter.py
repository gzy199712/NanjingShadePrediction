"""Sequential local tools for the bounded planner-facing Agent."""

from __future__ import annotations

import base64
import csv
import json
import shutil
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from routing.app.backend.planner_agent_orchestrator import (
    PlannerAgentRuntime, build_agent_plan, parse_planner_intent,
)
from routing.app.backend.planner_agent_tools import PlannerAgentTool, PlannerToolRegistry, tool_result


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(r"D:\miniconda3\envs\gptthermalcomfort\python.exe")
POWERPAINT_PYTHON = Path(r"D:\miniconda3\envs\gptthermalcomfort-powerpaint\python.exe")
RESULT_SCHEMA = {
    "type": "object",
    "required": ["tool", "status", "observations", "artifacts", "warnings", "error"],
}
LOCAL_GENERATION_REQUIREMENTS = (
    POWERPAINT_PYTHON,
    ROOT / "streetscape/scripts/stage_46b_panorama_tree_generation_agent.py",
    ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/stable-diffusion-v1-5-inpainting",
    ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/powerpaint-v1",
    ROOT / "artifacts/retired/failed_experiment_dependencies/third_party/powerpaint",
)


class PlannerGenerationAgentAdapter:
    def __init__(self) -> None:
        self.lock = threading.Lock()

    def _run(self, command: list[str], cwd: Path, log: Path, timeout: int = 900) -> None:
        completed = subprocess.run(
            command, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log.write_text(completed.stdout + "\n--- STDERR ---\n" + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"Agent tool failed ({completed.returncode}): {Path(command[1]).name}; see {log.name}")

    def analyze_and_generate(
        self, image_path: Path, planner_intent: str, capture_month: int | None,
        semantic_result: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            return self._analyze_and_generate(image_path, planner_intent, capture_month, semantic_result)

    def _analyze_and_generate(
        self, image_path: Path, planner_intent: str, capture_month: int | None,
        semantic_result: dict[str, Any],
    ) -> dict[str, Any]:
        task_dir = ROOT / "routing/app/runtime_uploads" / f"agent_{uuid.uuid4().hex}"
        task_dir.mkdir(parents=True, exist_ok=True)
        intent = parse_planner_intent(planner_intent)
        semantic_result = {
            **semantic_result,
            "local_generation_available": all(path.exists() for path in LOCAL_GENERATION_REQUIREMENTS),
        }
        plan = build_agent_plan(intent, semantic_result)
        context: dict[str, Any] = {"agent_image": image_path}
        point_id = "upload"
        agent_csv = task_dir / "agent.csv"
        benchmark_csv = task_dir / "benchmark.csv"
        windows = task_dir / "windows"
        layout = task_dir / "layout"
        structure = task_dir / "structure"
        generation = task_dir / "generation"
        acceptance = task_dir / "acceptance"

        def run_vlm(_state: dict[str, Any]) -> dict[str, Any]:
            output = task_dir / f"vlm_{uuid.uuid4().hex}.json"
            self._run([
                str(PYTHON), str(ROOT / "routing/app/backend/planner_vlm_agent_worker.py"),
                "--image", str(context["agent_image"]), "--output", str(output),
                "--intent", planner_intent,
            ], ROOT, task_dir / f"{output.stem}.log")
            vlm = json.loads(output.read_text(encoding="utf-8"))
            context["vlm"] = vlm
            return tool_result("tree_gap_analysis", "PASS", observations={
                "structural_scene": vlm["structural_scene"], "tree_state": vlm["tree_state"],
                "planning_action": vlm.get("planning_action"),
            }, artifacts={"rag_case_count": len(vlm.get("rag_safety_exemplar_ids") or [])})

        def road_space(_state: dict[str, Any]) -> dict[str, Any]:
            confirmed = (
                not bool(semantic_result.get("high_speed_road_suspected"))
                and str(semantic_result.get("public_walk_cycle_likelihood")) in {"较高", "high"}
            )
            return tool_result("road_space_analysis", "PASS" if confirmed else "UNCERTAIN", observations={
                "walkable_space_confirmed": confirmed,
                "space_type_assessment": semantic_result.get("space_type_assessment"),
            })

        def phenology_review(state: dict[str, Any]) -> dict[str, Any]:
            result = run_vlm(state)
            result["tool"] = "phenology_review"
            result["observations"]["phenology_reviewed"] = True
            return result

        def planning_overlay(_state: dict[str, Any]) -> dict[str, Any]:
            return tool_result("planning_overlay", "PASS", artifacts={
                "overlay_available": bool(semantic_result.get("overlay_png_base64")),
                "suggestion_region_count": len(semantic_result.get("suggestion_regions") or []),
            })

        def prepare_panorama() -> None:
            if not context.get("normalized"):
                normalized = task_dir / "normalized_panorama.png"
                self._run([
                    str(PYTHON), str(ROOT / "routing/app/backend/planner_panorama_normalize_worker.py"),
                    "--input", str(image_path), "--output", str(normalized),
                ], ROOT, task_dir / "00_normalize.log")
                context.update(agent_image=normalized, normalized=True)
            with agent_csv.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["point_id", "policy_state", "next_tool"])
                writer.writeheader()
                writer.writerow({"point_id": point_id, "policy_state": "READY_FOR_LAYOUT", "next_tool": "propose_tree_layout"})
            with benchmark_csv.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=["point_id", "benchmark_role", "output_path", "width", "height", "north_at_x0", "panorama_exists"])
                writer.writeheader()
                writer.writerow({
                    "point_id": point_id, "benchmark_role": "GROUND_WALKABLE_TREE_GAP",
                    "output_path": str(Path(context["agent_image"]).resolve()), "width": 2048,
                    "height": 512, "north_at_x0": True, "panorama_exists": True,
                })

        def directional_windows(_state: dict[str, Any]) -> dict[str, Any]:
            prepare_panorama()
            self._run([
                str(PYTHON), str(ROOT / "streetscape/scripts/stage_44_panorama_perspective_windows.py"),
                "--mode", "run", "--overwrite", "--agent-results", str(agent_csv),
                "--benchmark", str(benchmark_csv), "--output-directory", str(windows),
                "--report", str(task_dir / "step44.md"),
            ], ROOT, task_dir / "02_windows.log")
            return tool_result("directional_windows", "PASS", observations={"directional_windows_ready": True})

        def tree_layout(_state: dict[str, Any]) -> dict[str, Any]:
            self._run([
                str(PYTHON), str(ROOT / "streetscape/scripts/stage_45_tree_first_semantic_25d_layout.py"),
                "--mode", "run", "--overwrite", "--agent-results", str(agent_csv),
                "--windows-directory", str(windows), "--output-directory", str(layout),
                "--report", str(task_dir / "step45.md"),
            ], ROOT, task_dir / "03_layout.log")
            result = json.loads((layout / "points" / point_id / "layout.json").read_text(encoding="utf-8"))
            ready = result["status"] == "LAYOUT_READY"
            return tool_result("tree_layout", "PASS" if ready else "FAIL", observations={
                "layout_ready": ready, "tree_anchor_count": len(result.get("tree_anchors") or []),
                "selected_heading": result.get("selected_heading"),
            })

        def structural_gate(_state: dict[str, Any]) -> dict[str, Any]:
            self._run([
                str(PYTHON), str(ROOT / "streetscape/scripts/stage_45b_multimodal_structural_scene_gate.py"),
                "--mode", "run", "--overwrite", "--layout-directory", str(layout),
                "--windows-directory", str(windows), "--benchmark", str(benchmark_csv),
                "--output-directory", str(structure), "--report", str(task_dir / "step45b.md"),
            ], ROOT, task_dir / "04_structure.log")
            qc = pd.read_csv(structure / "structural_scene_qc.csv").iloc[0]
            structural_pass = bool(qc.planting_support_pass)
            repeat_output = task_dir / "vlm_repeat.json"
            self._run([
                str(PYTHON), str(ROOT / "routing/app/backend/planner_vlm_agent_worker.py"),
                "--image", str(context["agent_image"]), "--output", str(repeat_output),
                "--intent", planner_intent,
            ], ROOT, task_dir / "04b_vlm_repeat.log")
            repeat = json.loads(repeat_output.read_text(encoding="utf-8"))
            first = context.get("vlm") or {}
            repeatable = (
                first.get("structural_scene") == repeat.get("structural_scene") == "GROUND_STREET"
                and first.get("tree_state") == repeat.get("tree_state") == "TREE_GAP_WITH_PLANTING_SPACE"
            )
            passed = structural_pass and repeatable
            return tool_result("structural_gate", "PASS" if passed else "FAIL", observations={
                "structural_gate_pass": passed, "verified_structural_scene": str(qc.structural_scene),
                "vlm_repeatability_pass": repeatable,
            })

        def local_generation(_state: dict[str, Any]) -> dict[str, Any]:
            self._run([
                str(POWERPAINT_PYTHON), str(ROOT / "streetscape/scripts/stage_46b_panorama_tree_generation_agent.py"),
                "--mode", "run", "--overwrite", "--layout-directory", str(layout),
                "--structural-qc", str(structure / "structural_scene_qc.csv"),
                "--benchmark", str(benchmark_csv), "--windows-directory", str(windows),
                "--output-directory", str(generation), "--report", str(task_dir / "step46b.md"),
            ], ROOT, task_dir / "05_generation.log")
            return tool_result("local_generation", "PASS", observations={"generation_succeeded": True})

        def generation_acceptance(_state: dict[str, Any]) -> dict[str, Any]:
            self._run([
                str(PYTHON), str(ROOT / "streetscape/scripts/stage_47_automatic_generation_acceptance.py"),
                "--mode", "run", "--overwrite", "--agent-state", str(generation / "agent_state.json"),
                "--generation-results", str(generation / "generation_results.csv"),
                "--output-directory", str(acceptance), "--report", str(task_dir / "step47.md"),
            ], ROOT, task_dir / "06_acceptance.log")
            qc = pd.read_csv(acceptance / "acceptance_results.csv")
            accepted = qc.loc[qc.automatic_acceptance.astype(bool)]
            if len(accepted):
                context["accepted"] = accepted.sort_values(
                    ["svf_reduction", "vegetation_ratio_increase"], ascending=False,
                ).iloc[0]
            return tool_result("generation_acceptance", "PASS" if len(accepted) else "FAIL", observations={
                "automatic_acceptance": bool(len(accepted)), "accepted_variants": len(accepted),
                "generated_variants": len(qc),
            }, artifacts={
                "accepted_panorama": Path(context["accepted"].panorama_path).name if len(accepted) else None,
            })

        tools = [
            PlannerAgentTool("road_space_analysis", "Confirm public walk/cycle space from existing evidence.", {}, RESULT_SCHEMA, road_space),
            PlannerAgentTool("phenology_review", "Review existing trees when leaf state is uncertain.", {}, RESULT_SCHEMA, phenology_review, requires_gpu=True),
            PlannerAgentTool("tree_gap_analysis", "Classify structural scene and credible tree gap.", {}, RESULT_SCHEMA, run_vlm, requires_gpu=True, retry_on_failure=True),
            PlannerAgentTool("directional_windows", "Create direction windows only when layout is needed.", {}, RESULT_SCHEMA, directional_windows, requires_gpu=True),
            PlannerAgentTool("tree_layout", "Run the existing semantic-depth 2.5D layout.", {}, RESULT_SCHEMA, tree_layout, requires_gpu=True),
            PlannerAgentTool("structural_gate", "Verify layout with the existing structural gate.", {}, RESULT_SCHEMA, structural_gate, requires_gpu=True),
            PlannerAgentTool("local_generation", "Run the existing local full-GPU tree editor.", {}, RESULT_SCHEMA, local_generation, requires_gpu=True, retry_on_failure=True, requires_generation_unlock=True),
            PlannerAgentTool("generation_acceptance", "Run existing semantic and geometry acceptance.", {}, RESULT_SCHEMA, generation_acceptance, requires_gpu=True),
            PlannerAgentTool("planning_overlay", "Retain the deterministic translucent planning overlay.", {}, RESULT_SCHEMA, planning_overlay),
        ]
        try:
            result = PlannerAgentRuntime(
                PlannerToolRegistry(tools), max_steps=8, timeout_seconds=1800,
            ).run({
                "goal": intent["objective"], "constraints": intent["constraints"],
                "observations": semantic_result, "selected_action": plan["selected_action"],
            })
            response = {
                "agent_name": "planner_multimodal_tool_agent", "agent_operational": True,
                "agent_plan": plan, "intent_contract": intent, "planner_intent_applied": planner_intent,
                "capture_month": capture_month, "vlm": context.get("vlm"),
                "generation_blockers": [name for name, passed in result["generation_gates"].items() if not passed],
                "agent_explanation": result["stop_reason"], **result,
            }
            best = context.get("accepted")
            if best is not None:
                response.update(
                    generated_panorama_png_base64=base64.b64encode(Path(best.panorama_path).read_bytes()).decode("ascii"),
                    selected_seed=int(best.seed),
                    scenario_metrics={
                        "baseline_svf": float(best.baseline_svf), "scenario_svf": float(best.scenario_svf),
                        "svf_reduction": float(best.svf_reduction),
                        "vegetation_ratio_increase": float(best.vegetation_ratio_increase),
                        "tmrt_utci_state": str(best.thermal_evaluation_state),
                    },
                )
            return response
        finally:
            shutil.rmtree(task_dir, ignore_errors=True)
