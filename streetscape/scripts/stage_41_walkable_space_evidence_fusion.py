"""stage_41: fuse VLM heading clues with semantic, depth-ground and road evidence."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/walkable_evidence_fusion.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_41"


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_41")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_41.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def circular_delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def vlm_headings(response: dict[str, Any], keywords: list[str]) -> list[int]:
    results = set()
    keyword_pattern = "|".join(re.escape(value) for value in keywords)
    for claim in response.get("evidence", []):
        text = str(claim).lower()
        if not re.search(keyword_pattern, text):
            continue
        for value in re.findall(r"heading\s+(0|30|60|90|120|150|180|210|240|270|300|330)", text):
            results.add(int(value))
    return sorted(results)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def fuse_point(case: dict[str, Any], decision: dict[str, Any], views: pd.DataFrame, response: dict[str, Any], rules: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    point_id = str(case["point_id"])
    candidates = vlm_headings(response, list(rules["vlm_keywords"]))
    motor_excluded = bool_value(decision.get("motor_vehicle_only_excluded"))
    thermal_relevant = bool_value(decision.get("thermal_comfort_relevant"))
    traces = []
    supported_cardinals = []
    for candidate in candidates:
        neighborhood = views.loc[views.heading.map(lambda value: circular_delta(float(value), float(candidate)) <= float(rules["heading_neighborhood_degrees"]))].copy()
        if len(neighborhood):
            neighborhood["semantic_relaxed"] = neighborhood.lower_half_walkable_probability.ge(float(rules["relaxed_lower_half_probability_minimum"])) | neighborhood.retained_walkable_ratio.ge(float(rules["relaxed_retained_ratio_minimum"]))
            neighborhood["ground_pass"] = neighborhood.ground_score_mean.ge(float(rules["ground_score_minimum"]))
            neighborhood["obstacle_pass"] = neighborhood.obstacle_probability_mean.le(float(rules["obstacle_probability_maximum"]))
            neighborhood["tool_supported"] = neighborhood.semantic_relaxed & neighborhood.ground_pass & neighborhood.obstacle_pass
        support_count = int(neighborhood.tool_supported.sum()) if len(neighborhood) else 0
        if support_count >= int(rules["minimum_adjacent_supported_views"]):
            supported_cardinals.append(candidate)
        traces.append({
            "point_id": point_id,
            "candidate_heading": candidate,
            "available_neighbor_views": int(len(neighborhood)),
            "tool_supported_neighbor_views": support_count,
            "max_lower_half_probability": float(neighborhood.lower_half_walkable_probability.max()) if len(neighborhood) else None,
            "max_retained_walkable_ratio": float(neighborhood.retained_walkable_ratio.max()) if len(neighborhood) else None,
            "mean_ground_score": float(neighborhood.ground_score_mean.mean()) if len(neighborhood) else None,
        })
    if motor_excluded and rules["motor_vehicle_exclusion_is_hard_gate"]:
        fusion_decision = "EXCLUDE_MOTOR_ONLY"
        stop_reason = "road_context_motor_vehicle_only_hard_gate"
    elif views.empty:
        fusion_decision = "REQUEST_GPU_LOCALIZATION"
        stop_reason = "no_semantic_depth_view_metrics"
    elif rules["require_thermal_relevant_road"] and not thermal_relevant:
        fusion_decision = "KEEP_ABSTAIN"
        stop_reason = "road_context_not_thermal_comfort_relevant"
    elif supported_cardinals:
        fusion_decision = "PROMOTE_TO_PIXEL_LOCALIZATION"
        stop_reason = "vlm_heading_plus_adjacent_semantic_depth_ground_support"
    else:
        fusion_decision = "KEEP_ABSTAIN"
        stop_reason = "insufficient_two_family_localization_evidence"
    result = {
        "point_id": point_id,
        "benchmark_role": case["benchmark_role"],
        "fclass": decision.get("fclass"),
        "motor_vehicle_only_excluded": motor_excluded,
        "thermal_comfort_relevant": thermal_relevant,
        "vlm_candidate_headings": "|".join(map(str, candidates)),
        "vlm_candidate_count": len(candidates),
        "tool_supported_headings": "|".join(map(str, supported_cardinals)),
        "tool_supported_heading_count": len(supported_cardinals),
        "fusion_decision": fusion_decision,
        "stop_reason": stop_reason,
        "generation_unlocked": False,
        "next_tool": "rerun_pixel_localization" if fusion_decision == "PROMOTE_TO_PIXEL_LOCALIZATION" else "none",
    }
    return result, traces


def expected_pass(frame: pd.DataFrame) -> dict[str, Any]:
    motor = frame.loc[frame.benchmark_role.eq("motor_only_control")]
    positive = frame.loc[frame.benchmark_role.eq("positive_walkable_control")]
    rescue = frame.loc[frame.benchmark_role.eq("rescue_candidate")]
    return {
        "motor_negative_control_safe": bool(len(motor) and motor.fusion_decision.eq("EXCLUDE_MOTOR_ONLY").all()),
        "positive_control_recovered": bool(len(positive) and positive.fusion_decision.eq("PROMOTE_TO_PIXEL_LOCALIZATION").all()),
        "rescue_candidate_promoted_for_localization": int(rescue.fusion_decision.eq("PROMOTE_TO_PIXEL_LOCALIZATION").sum()),
        "generation_unlock_count": int(frame.generation_unlocked.sum()),
    }


def write_report(summary: dict[str, Any], results: pd.DataFrame, path: Path) -> None:
    rows = "\n".join(f"| {row.point_id} | {row.benchmark_role} | {row.vlm_candidate_headings or '-'} | {row.tool_supported_headings or '-'} | {row.fusion_decision} |" for row in results.itertuples(index=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_41 Walkable-space evidence fusion

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Motor-only negative control safe: {summary['checks']['motor_negative_control_safe']}
- Positive control recovered for pixel localization: {summary['checks']['positive_control_recovered']}
- Rescue candidates promoted for pixel localization: {summary['checks']['rescue_candidate_promoted_for_localization']}
- Generation unlock count: {summary['checks']['generation_unlock_count']}

| Point | Role | VLM headings | Tool-supported headings | Fusion decision |
|---|---|---|---|---|
{rows}

Promotion means only that a GPU pixel-localization tool should be rerun with targeted relaxed
thresholds. It does not prove a usable planting site and never unlocks image generation directly.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        for value in cfg["inputs"].values():
            require(resolve(value))
        manifest = pd.read_csv(resolve(cfg["inputs"]["rescue_manifest"]), dtype={"point_id": str})
        pilot_dir = resolve(cfg["inputs"]["rescue_pilot_directory"])
        pilot_ids = sorted(path.name for path in pilot_dir.iterdir() if path.is_dir() and (path / "structured_response.json").is_file())
        decisions = pd.read_csv(resolve(cfg["inputs"]["decisions"]), dtype={"point_id": str}, low_memory=False)
        views = pd.read_csv(resolve(cfg["inputs"]["view_metrics"]), dtype={"point_id": str}, low_memory=False)
        check = {"status": "PASS" if len(pilot_ids) == 3 else "FAIL", "pilot_points": len(pilot_ids), "view_metric_points": int(views.point_id.nunique()), "generation_unlock_allowed": bool(cfg["rules"]["generation_unlock_allowed"])}
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        out = resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True, exist_ok=True)
        rows, trace_rows, errors = [], [], []
        for point_id in tqdm(pilot_ids, desc="Fusing walkable evidence", unit="point", dynamic_ncols=True):
            try:
                case = manifest.loc[manifest.point_id.eq(point_id)].iloc[0].to_dict()
                decision = decisions.loc[decisions.point_id.eq(point_id)].iloc[0].to_dict()
                point_views = views.loc[views.point_id.eq(point_id)].copy()
                response = json.loads((pilot_dir / point_id / "structured_response.json").read_text(encoding="utf-8"))
                result, traces = fuse_point(case, decision, point_views, response, cfg["rules"])
                rows.append(result)
                trace_rows.extend(traces)
            except Exception as error:
                log.exception("Point %s failed", point_id)
                errors.append({"point_id": point_id, "filename": "evidence_fusion", "error_message": f"{type(error).__name__}: {error}"})
        pd.DataFrame(errors, columns=["point_id", "filename", "error_message"]).to_csv(failed, index=False, encoding="utf-8-sig")
        results = pd.DataFrame(rows)
        traces = pd.DataFrame(trace_rows)
        results.to_csv(out / "pilot_fusion_decisions.csv", index=False, encoding="utf-8-sig")
        traces.to_csv(out / "pilot_heading_evidence.csv", index=False, encoding="utf-8-sig")
        checks = expected_pass(results)
        status = "PILOT_PASS" if len(results) == 3 and checks["motor_negative_control_safe"] and checks["positive_control_recovered"] and checks["generation_unlock_count"] == 0 else "PILOT_FAIL"
        summary = {**check, "status": status, "generated": datetime.now().astimezone().isoformat(), "successful_points": len(results), "failed_points": len(errors), "checks": checks}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        write_report(summary, results, resolve(cfg["outputs"]["report"]))
        log.info("stage_41 %s: %s", status, checks)
        return 0 if status == "PILOT_PASS" else 2
    except Exception as error:
        log.exception("stage_41 failed")
        with failed.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_41", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

