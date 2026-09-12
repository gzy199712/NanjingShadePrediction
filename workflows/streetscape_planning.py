"""Run the current evidence-based, panorama-scale streetscape planning workflow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from workflows.runtime import PROJECT_ROOT, Stage, make_logger, record_failure, require_cuda, require_paths, require_writable, run_stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "panorama", "localize", "decide", "visualize", "agent-preflight", "agent-single-point", "agent-qa", "agent-rescue", "agent-fuse", "agent-pixel", "agent-heading", "agent-cases", "agent-embed", "agent-index", "agent-plan", "agent-windows", "agent-layout", "agent-structure", "agent-candidate", "agent-tree-consensus", "agent-editor", "agent-generate", "agent-accept", "agent-adaptation", "agent-finalize"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = make_logger("streetscape_planning")
    try:
        scripts = PROJECT_ROOT / "streetscape/scripts"
        panoramas = PROJECT_ROOT / "streetscape/data/panorama/stage_04_formal/images"
        if args.action == "panorama":
            require_paths([
                PROJECT_ROOT / "artifacts/intermediate/pipeline_records/steps/step0_check/image_metadata.csv",
                PROJECT_ROOT / "data/raw/NanjingStreetViewImages_12Directions_pitch0",
            ])
        elif args.action != "check":
            require_paths([panoramas], kind="dir")
        require_writable(PROJECT_ROOT / "streetscape/data/planning")
        mapping = {
            "panorama": scripts / "stage_04_formal_ghost_free_panorama.py",
            "localize": scripts / "stage_31_integrated_streaming_gpu_localization.py",
            "decide": scripts / "stage_32_citywide_final_planning_decisions.py",
            "visualize": scripts / "stage_33_citywide_terminal_visualization.py",
            "agent-preflight": scripts / "stage_37_planner_agent_foundation_preflight.py",
            "agent-single-point": scripts / "stage_38_qwen3vl_quantized_single_point.py",
            "agent-qa": scripts / "stage_39_structured_streetscape_qa.py",
            "agent-rescue": scripts / "stage_40_active_travel_recognition_rescue.py",
            "agent-fuse": scripts / "stage_41_walkable_space_evidence_fusion.py",
            "agent-pixel": scripts / "stage_41b_targeted_gpu_pixel_relocalization.py",
            "agent-heading": scripts / "stage_41c_active_travel_heading_screen.py",
            "agent-cases": scripts / "stage_42_multimodal_case_library.py",
            "agent-embed": scripts / "stage_42b_qwen3vl_embedding_preflight.py",
            "agent-index": scripts / "stage_42c_multimodal_rag_index.py",
            "agent-plan": scripts / "stage_43_tool_constrained_planner_agent.py",
            "agent-windows": scripts / "stage_44_panorama_perspective_windows.py",
            "agent-layout": scripts / "stage_45_tree_first_semantic_25d_layout.py",
            "agent-structure": scripts / "stage_45b_multimodal_structural_scene_gate.py",
            "agent-candidate": scripts / "stage_45c_ground_street_candidate_rediscovery.py",
            "agent-tree-consensus": scripts / "stage_45f_tree_state_consensus_gate.py",
            "agent-editor": scripts / "stage_46_powerpaint_gpu_editor_preflight.py",
            "agent-generate": scripts / "stage_46b_panorama_tree_generation_agent.py",
            "agent-accept": scripts / "stage_47_automatic_generation_acceptance.py",
            "agent-adaptation": scripts / "stage_48_adaptation_and_model_strategy.py",
            "agent-finalize": scripts / "stage_49_finalize_planner_generation_agent.py",
        }
        if args.action == "check":
            require_paths(mapping.values(), kind="file")
            logger.info("streetscape planning preflight passed")
            return 0
        script = mapping[args.action]
        require_paths([script], kind="file")
        if args.action in {"panorama", "localize", "agent-preflight", "agent-single-point", "agent-qa", "agent-rescue", "agent-pixel", "agent-heading", "agent-embed", "agent-index", "agent-plan", "agent-windows", "agent-layout", "agent-structure", "agent-candidate", "agent-tree-consensus", "agent-editor", "agent-generate", "agent-accept"}:
            require_cuda()
        python = Path(r"D:\miniconda3\envs\gptthermalcomfort-powerpaint\python.exe") if args.action in {"agent-editor", "agent-generate"} else Path(sys.executable)
        if args.action == "agent-tree-consensus":
            command = [str(python), str(script),
                "--classification", str(PROJECT_ROOT / "streetscape/data/agent/stage_45c_candidate_rediscovery/candidate_scene_classification.csv"),
                "--structural-qc", str(PROJECT_ROOT / "streetscape/data/agent/stage_45e_rediscovered_structure/structural_scene_qc.csv"),
                "--output-directory", str(PROJECT_ROOT / "streetscape/data/agent/stage_45f_tree_state_consensus")]
        else:
            command = [str(python), str(script), "--mode", "run"]
        if args.overwrite:
            command.append("--overwrite")
        return run_stages("streetscape_planning", [Stage(args.action, command)], logger)
    except Exception as error:
        logger.exception("workflow failed")
        record_failure("streetscape_planning", "preflight", args.action, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
