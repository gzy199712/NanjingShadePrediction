"""Step25: point-to-edge candidate matching check and formal execution gate."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.point_edge_matching import (  # noqa: E402
    EdgeCandidate,
    build_edge_grid,
    grid_key,
    nearby_cells,
    nearest_segment_bearing,
    parse_move_directions,
    score_candidate,
    undirected_angle_difference,
)
from routing.src.reporting import atomic_csv, atomic_json, atomic_text  # noqa: E402

SCRIPT_VERSION = "2.0.0"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase G4 point-edge matching.")
    parser.add_argument(
        "--network-config",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/phase_g_network.yaml",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/point_edge_matching_draft.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step25")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    for handler in (
        logging.FileHandler(
            path, mode="w" if overwrite else "a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def read_points(path: Path) -> list[dict[str, Any]]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayer(0)
    rows: list[dict[str, Any]] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading formal thermal points",
        unit="point",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        rows.append(
            {
                "point_id": str(feature.GetField("ID")),
                "x": float(geometry.GetX()),
                "y": float(geometry.GetY()),
            }
        )
    return rows


def read_edges(path: Path) -> list[EdgeCandidate]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayerByName("ThermalComfortNetworkRepaired")
    if layer is None:
        raise KeyError("ThermalComfortNetworkRepaired")
    rows: list[EdgeCandidate] = []
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading repaired thermal edges",
        unit="edge",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        envelope = geometry.GetEnvelope()
        rows.append(
            EdgeCandidate(
                edge_id=str(feature.GetField("edge_id")),
                fclass=str(feature.GetField("fclass") or "unknown"),
                bridge=str(feature.GetField("bridge") or "F"),
                tunnel=str(feature.GetField("tunnel") or "F"),
                layer=feature.GetField("layer"),
                grade_separated=bool(feature.GetField("grade_sep")),
                repair_type=str(feature.GetField("repair_typ") or "original"),
                geometry=geometry.Clone(),
                envelope=(
                    float(envelope[0]),
                    float(envelope[1]),
                    float(envelope[2]),
                    float(envelope[3]),
                ),
            )
        )
    return rows


def preview_confidence(
    best: dict[str, Any],
    second: dict[str, Any] | None,
    rules: dict[str, Any],
) -> str:
    config = rules["confidence_preview"]
    gap = (
        float(second["match_score"]) - float(best["match_score"])
        if second is not None
        else 1.0
    )
    direction = best["direction_difference_deg"]
    if (
        best["distance_m"] <= float(config["high_max_distance_m"])
        and direction is not None
        and direction <= float(config["high_max_direction_difference_deg"])
        and gap >= float(config["high_min_score_gap"])
        and not best["grade_separated"]
    ):
        return "high"
    if (
        best["distance_m"] <= float(config["medium_max_distance_m"])
        and gap >= float(config["medium_min_score_gap"])
    ):
        return "medium"
    return "low"


def run_check(
    config_path: Path,
    rules_path: Path,
    overwrite: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(config_path)
    rules = load_yaml(rules_path)
    root = Path(str(config["project_root"])).resolve()
    output_dir = root / "routing/data/point_edge_mapping"
    report_path = root / "routing/reports/STEP25_POINT_EDGE_CHECK_REPORT.md"
    outputs = {
        "preview": output_dir / "point_edge_mapping_preview.csv",
        "candidates": output_dir / "point_edge_top_candidates.csv",
        "coverage": output_dir / "point_edge_coverage_by_distance.csv",
        "low_confidence": output_dir / "point_edge_low_confidence_review.csv",
        "failed": output_dir / "step25_failed_records.csv",
        "summary": output_dir / "step25_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step25 check outputs exist; pass --overwrite")
    step24 = json.loads(
        (
            root / "routing/data/topology/step24_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not step24.get("ready_for_step25_check"):
        blockers.append("Step24 formal output is not ready for Step25 check")
    points = read_points(
        root / "data/training_data/spatial/main_points_8975.shp"
    )
    expected_point_count = int(config["runtime"]["expected_thermal_points"])
    if len(points) != expected_point_count:
        blockers.append(f"Expected {expected_point_count} points, found {len(points)}")
    point_ids = {row["point_id"] for row in points}
    move_dirs, move_failures = parse_move_directions(
        root / "data/raw/NanjingStreetViewImages_12Directions_pitch0",
        point_ids,
    )
    if move_failures:
        warnings.append(
            f"{len(move_failures)} points/files have move_dir issues; "
            "distance-only preview policy applied where possible."
        )
    edges = read_edges(
        root / "routing/data/topology/step24_repaired_network.gdb"
    )
    expected_edge_count = int(step24["repaired_network_edge_count"])
    if len(edges) != expected_edge_count:
        blockers.append(
            f"Expected {expected_edge_count} repaired edges, found {len(edges)}"
        )
    maximum_distance = float(rules["maximum_candidate_distance_m"])
    grid_size = maximum_distance
    edge_grid = build_edge_grid(edges, grid_size)
    top_n = int(rules["top_candidates_per_point"])
    all_top_candidates: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    runtime_failures: list[dict[str, Any]] = list(move_failures)
    candidate_counts_by_point: dict[str, int] = {}
    nearest_distances: dict[str, float | None] = {}
    for point in tqdm(
        points,
        desc="Scoring point-edge candidates",
        unit="point",
        dynamic_ncols=True,
    ):
        point_id = point["point_id"]
        try:
            candidate_indices: set[int] = set()
            for cell in nearby_cells(
                point["x"],
                point["y"],
                maximum_distance,
                grid_size,
            ):
                candidate_indices.update(edge_grid.get(cell, []))
            candidates: list[dict[str, Any]] = []
            move_dir = move_dirs.get(point_id)
            for edge_index in candidate_indices:
                edge = edges[edge_index]
                distance, bearing = nearest_segment_bearing(
                    edge.geometry, point["x"], point["y"]
                )
                if distance > maximum_distance:
                    continue
                direction_difference = (
                    undirected_angle_difference(move_dir, bearing)
                    if move_dir is not None
                    else None
                )
                score, components = score_candidate(
                    distance_m=distance,
                    direction_difference_deg=direction_difference,
                    fclass=edge.fclass,
                    grade_separated=edge.grade_separated,
                    rules=rules,
                )
                candidates.append(
                    {
                        "point_id": point_id,
                        "candidate_rank": 0,
                        "edge_id": edge.edge_id,
                        "distance_m": distance,
                        "point_move_dir": move_dir,
                        "edge_bearing": bearing,
                        "direction_difference_deg": direction_difference,
                        "fclass": edge.fclass,
                        "bridge": edge.bridge,
                        "tunnel": edge.tunnel,
                        "layer": edge.layer,
                        "grade_separated": edge.grade_separated,
                        "repair_type": edge.repair_type,
                        "distance_cost": components["distance_cost"],
                        "direction_cost": components["direction_cost"],
                        "class_cost": components["class_cost"],
                        "level_cost": components["level_cost"],
                        "match_score": score,
                        "formal_match": False,
                    }
                )
            candidates.sort(key=lambda row: (row["match_score"], row["distance_m"]))
            candidate_counts_by_point[point_id] = len(candidates)
            nearest_distances[point_id] = (
                min(row["distance_m"] for row in candidates)
                if candidates
                else None
            )
            selected = candidates[:top_n]
            for rank, row in enumerate(selected, start=1):
                row["candidate_rank"] = rank
                all_top_candidates.append(row)
            if not selected:
                previews.append(
                    {
                        "point_id": point_id,
                        "candidate_count_75m": 0,
                        "best_edge_id": None,
                        "best_distance_m": None,
                        "best_direction_difference_deg": None,
                        "best_match_score": None,
                        "second_edge_id": None,
                        "second_match_score": None,
                        "score_gap": None,
                        "match_confidence_preview": "unmatched",
                        "manual_review": True,
                        "formal_match": False,
                    }
                )
                continue
            best = selected[0]
            second = selected[1] if len(selected) > 1 else None
            confidence = preview_confidence(best, second, rules)
            previews.append(
                {
                    "point_id": point_id,
                    "candidate_count_75m": len(candidates),
                    "best_edge_id": best["edge_id"],
                    "best_distance_m": best["distance_m"],
                    "best_direction_difference_deg": best[
                        "direction_difference_deg"
                    ],
                    "best_match_score": best["match_score"],
                    "second_edge_id": (
                        second["edge_id"] if second is not None else None
                    ),
                    "second_match_score": (
                        second["match_score"] if second is not None else None
                    ),
                    "score_gap": (
                        second["match_score"] - best["match_score"]
                        if second is not None
                        else None
                    ),
                    "match_confidence_preview": confidence,
                    "manual_review": confidence in {"low", "unmatched"},
                    "formal_match": False,
                }
            )
        except Exception as error:
            runtime_failures.append(
                {
                    "category": "point_candidate_scoring",
                    "object_id": point_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    preview_df = pd.DataFrame(previews)
    candidates_df = pd.DataFrame(all_top_candidates)
    low_confidence = preview_df[
        preview_df["manual_review"].astype(bool)
    ].copy()
    coverage_rows: list[dict[str, Any]] = []
    for distance in rules["candidate_distances_m"]:
        threshold = float(distance)
        matched = sum(
            value is not None and value <= threshold
            for value in nearest_distances.values()
        )
        coverage_rows.append(
            {
                "candidate_distance_m": threshold,
                "matched_point_count": matched,
                "unmatched_point_count": len(points) - matched,
                "matched_point_ratio": matched / len(points),
                "formal_match_performed": False,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["preview"], preview_df.to_dict("records"))
    atomic_csv(outputs["candidates"], candidates_df.to_dict("records"))
    atomic_csv(outputs["coverage"], coverage_rows)
    atomic_csv(outputs["low_confidence"], low_confidence.to_dict("records"))
    if runtime_failures:
        atomic_csv(outputs["failed"], runtime_failures)
    else:
        atomic_text(
            outputs["failed"],
            "category,object_id,error_type,error_message\n",
        )
    confidence_counts = {
        str(key): int(value)
        for key, value in preview_df[
            "match_confidence_preview"
        ].value_counts().items()
    }
    ended_at = now_text()
    summary = {
        "phase": "G4",
        "step": 25,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_point_edge_rule_approval": len(blockers) == 0,
        "ready_for_formal_point_edge_run": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "point_count": len(points),
        "repaired_edge_count": len(edges),
        "move_dir_recovered_count": len(move_dirs),
        "move_dir_issue_count": len(move_failures),
        "top_candidate_record_count": len(candidates_df),
        "coverage_by_distance": coverage_rows,
        "confidence_preview_counts": confidence_counts,
        "manual_review_count": len(low_confidence),
        "failed_record_count": len(runtime_failures),
        "rule_status": rules.get("status"),
        "rule_approved": bool(rules.get("approval", {}).get("approved")),
        "formal_mapping_written": False,
        "source_points_modified": False,
        "step24_network_modified": False,
        "thermal_costs_assigned": False,
        "route_search_performed": False,
        "output_paths": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step25 Point-Edge Matching Check Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_POINT_EDGE_RULE_APPROVAL = {str(summary['ready_for_point_edge_rule_approval']).upper()}`",
        "- `READY_FOR_FORMAL_POINT_EDGE_RUN = FALSE`",
        f"- Points: {len(points):,}",
        f"- Repaired thermal edges: {len(edges):,}",
        f"- move_dir recovered: {len(move_dirs):,}",
        f"- Top candidate records: {len(candidates_df):,}",
        f"- Low/unmatched manual review points: {len(low_confidence):,}",
        "- Formal mappings written: 0",
        "",
        "## Candidate distance coverage",
        "",
        "| distance m | matched points | ratio |",
        "|---:|---:|---:|",
    ]
    for row in coverage_rows:
        lines.append(
            f"| {row['candidate_distance_m']:.0f} | "
            f"{row['matched_point_count']:,} | "
            f"{row['matched_point_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "This is a non-binding candidate preview. Review distance thresholds, score weights, confidence rules and low-confidence cases before approving formal mapping.",
        ]
    )
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step25 check complete: ready=%s points=%s review=%s failed=%s",
        summary["ready_for_point_edge_rule_approval"],
        len(points),
        len(low_confidence),
        len(runtime_failures),
    )
    return summary


def run_formal(
    config_path: Path,
    rules_path: Path,
    overwrite: bool,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Freeze approved high/medium matches and retain low/unmatched as unmapped."""
    started_at = now_text()
    started = time.perf_counter()
    config = load_yaml(config_path)
    rules = load_yaml(rules_path)
    root = Path(str(config["project_root"])).resolve()
    output_dir = root / "routing/data/point_edge_mapping"
    mapping_path = output_dir / "point_edge_mapping.csv"
    review_path = output_dir / "point_edge_unmatched_review.csv"
    failed_path = output_dir / "step25_formal_failed_records.csv"
    summary_path = output_dir / "step25_formal_summary.json"
    qc_path = output_dir / "step25_formal_qc.json"
    report_path = root / "routing/reports/STEP25_FORMAL_RUN_REPORT.md"
    outputs = [
        mapping_path,
        review_path,
        failed_path,
        summary_path,
        qc_path,
        report_path,
    ]
    if resume and all(path.exists() for path in outputs):
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        logger.info("Resume: using validated Step25 formal outputs")
        return prior
    if any(path.exists() for path in outputs) and not overwrite:
        raise FileExistsError(
            "Step25 formal outputs exist; use --resume or --overwrite"
        )
    for path in outputs:
        if path.exists():
            path.unlink()

    check_summary = json.loads(
        (output_dir / "step25_check_summary.json").read_text(encoding="utf-8")
    )
    if not check_summary.get("ready_for_point_edge_rule_approval"):
        raise RuntimeError("Step25 check is not ready for formal execution")
    expected_point_count = int(check_summary["point_count"])
    preview = pd.read_csv(
        output_dir / "point_edge_mapping_preview.csv",
        dtype={"point_id": str, "best_edge_id": str, "second_edge_id": str},
        low_memory=False,
    )
    candidates = pd.read_csv(
        output_dir / "point_edge_top_candidates.csv",
        dtype={"point_id": str, "edge_id": str},
        low_memory=False,
    )
    best = candidates[candidates["candidate_rank"] == 1].copy()
    best = best.rename(
        columns={
            "edge_id": "candidate_edge_id",
            "distance_m": "candidate_distance_m",
            "match_score": "candidate_match_score",
        }
    )
    best_fields = [
        "point_id",
        "candidate_edge_id",
        "candidate_distance_m",
        "point_move_dir",
        "edge_bearing",
        "direction_difference_deg",
        "fclass",
        "bridge",
        "tunnel",
        "layer",
        "grade_separated",
        "repair_type",
        "candidate_match_score",
    ]
    merged = preview.merge(
        best[best_fields],
        on="point_id",
        how="left",
        validate="one_to_one",
    )
    accepted_confidence = set(rules["formal_accept_confidence"])
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in tqdm(
        merged.to_dict("records"),
        desc="Freezing approved point-edge mappings",
        unit="point",
        dynamic_ncols=True,
    ):
        try:
            confidence = str(row["match_confidence_preview"])
            accepted = confidence in accepted_confidence
            if accepted:
                match_status = "matched"
                formal_edge_id = row["candidate_edge_id"]
            elif confidence == "low":
                match_status = "unmatched_low_confidence"
                formal_edge_id = None
            else:
                match_status = "unmatched_no_candidate"
                formal_edge_id = None
            records.append(
                {
                    "point_id": row["point_id"],
                    "edge_id": formal_edge_id,
                    "mode_network": "thermal_comfort_relevant",
                    "distance_m": (
                        row["candidate_distance_m"] if accepted else None
                    ),
                    "point_move_dir": row.get("point_move_dir"),
                    "edge_bearing": (
                        row.get("edge_bearing") if accepted else None
                    ),
                    "direction_difference": (
                        row.get("direction_difference_deg")
                        if accepted
                        else None
                    ),
                    "fclass": row.get("fclass") if accepted else None,
                    "bridge": row.get("bridge") if accepted else None,
                    "tunnel": row.get("tunnel") if accepted else None,
                    "layer": row.get("layer") if accepted else None,
                    "grade_separated": (
                        row.get("grade_separated") if accepted else None
                    ),
                    "match_score": (
                        row["candidate_match_score"] if accepted else None
                    ),
                    "match_confidence": confidence,
                    "match_status": match_status,
                    "manual_review": not accepted,
                    "candidate_edge_id": row.get("candidate_edge_id"),
                    "candidate_distance_m": row.get("candidate_distance_m"),
                    "candidate_match_score": row.get("candidate_match_score"),
                    "formal_match": accepted,
                    "rule_version": rules["rule_version"],
                }
            )
        except Exception as error:
            failures.append(
                {
                    "category": "formal_mapping",
                    "object_id": row.get("point_id"),
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    mapping = pd.DataFrame(records)
    review = mapping[mapping["manual_review"].astype(bool)].copy()
    accepted = mapping[mapping["formal_match"].astype(bool)].copy()
    blockers: list[str] = []
    if failures:
        blockers.append(f"{len(failures)} point mappings failed")
    if len(mapping) != expected_point_count or not mapping["point_id"].is_unique:
        blockers.append("Formal mapping is incomplete or has duplicate point_id")
    confidence_counts = check_summary["confidence_preview_counts"]
    expected_accepted = sum(
        int(confidence_counts.get(value, 0)) for value in accepted_confidence
    )
    expected_review = expected_point_count - expected_accepted
    if len(accepted) != expected_accepted:
        blockers.append(
            f"Expected {expected_accepted} accepted mappings, found {len(accepted)}"
        )
    if len(review) != expected_review:
        blockers.append(f"Expected {expected_review} review points, found {len(review)}")
    candidate_edge_ids = set(candidates["edge_id"].dropna().astype(str))
    invalid_edges = sorted(
        set(accepted["edge_id"].dropna().astype(str)) - candidate_edge_ids
    )
    if invalid_edges:
        blockers.append(
            f"{len(invalid_edges)} accepted edge IDs are absent from candidates"
        )
    if (accepted["distance_m"] > 75.0).any():
        blockers.append("Accepted mapping exceeds 75 m")
    if set(accepted["match_confidence"]) - accepted_confidence:
        blockers.append("A low/unmatched point was formally accepted")
    atomic_csv(mapping_path, mapping.to_dict("records"))
    atomic_csv(review_path, review.to_dict("records"))
    if failures:
        atomic_csv(failed_path, failures)
    else:
        atomic_text(
            failed_path,
            "category,object_id,error_type,error_message\n",
        )
    ended_at = now_text()
    status_counts = {
        str(key): int(value)
        for key, value in mapping["match_status"].value_counts().items()
    }
    summary = {
        "phase": "G4",
        "step": 25,
        "mode": "run",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "rule_version": rules["rule_version"],
        "rule_status": rules["status"],
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": not blockers,
        "ready_for_step26_check": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": 0,
        "warnings": [],
        "point_count": len(mapping),
        "accepted_mapping_count": len(accepted),
        "manual_review_count": len(review),
        "match_status_counts": status_counts,
        "maximum_accepted_distance_m": (
            float(accepted["distance_m"].max()) if len(accepted) else None
        ),
        "failed_record_count": len(failures),
        "source_points_modified": False,
        "step24_network_modified": False,
        "thermal_costs_assigned": False,
        "route_search_performed": False,
        "outputs": {
            "mapping_csv": str(mapping_path),
            "review_csv": str(review_path),
            "failed_records_csv": str(failed_path),
            "summary_json": str(summary_path),
            "qc_json": str(qc_path),
            "report": str(report_path),
        },
    }
    qc = {
        "success": not blockers,
        "point_row_count": len(mapping),
        "unique_point_id_count": int(mapping["point_id"].nunique()),
        "accepted_mapping_count": len(accepted),
        "review_count": len(review),
        "accepted_high_count": int(
            (accepted["match_confidence"] == "high").sum()
        ),
        "accepted_medium_count": int(
            (accepted["match_confidence"] == "medium").sum()
        ),
        "rejected_low_count": int(
            (mapping["match_confidence"] == "low").sum()
        ),
        "unmatched_no_candidate_count": int(
            (mapping["match_confidence"] == "unmatched").sum()
        ),
        "invalid_accepted_edge_id_count": len(invalid_edges),
        "maximum_accepted_distance_m": summary[
            "maximum_accepted_distance_m"
        ],
        "accepted_over_75m_count": int(
            (accepted["distance_m"] > 75.0).sum()
        ),
        "failed_record_count": len(failures),
        "formal_low_confidence_match_count": int(
            (
                mapping["formal_match"].astype(bool)
                & mapping["match_confidence"].isin(["low", "unmatched"])
            ).sum()
        ),
    }
    atomic_json(summary_path, summary)
    atomic_json(qc_path, qc)
    report_lines = [
        "# Step25 Formal Run Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- Success: `{str(summary['success']).upper()}`",
        f"- Ready for Step26 check: `{str(summary['ready_for_step26_check']).upper()}`",
        f"- Total points: {len(mapping):,}",
        f"- Formal high/medium mappings: {len(accepted):,}",
        f"- Low/no-candidate review points: {len(review):,}",
        f"- Maximum accepted distance: {summary['maximum_accepted_distance_m']:.3f} m",
        f"- Failed records: {len(failures)}",
        "",
        "Low-confidence and no-candidate points were not forced onto a road. Candidate fields are retained for manual review, while formal edge_id remains empty.",
        "",
        "No thermal edge costs or route searches were produced.",
    ]
    atomic_text(report_path, "\n".join(report_lines) + "\n")
    logger.info(
        "Step25 formal complete: success=%s accepted=%s review=%s",
        summary["success"],
        len(accepted),
        len(review),
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_yaml(args.network_config.resolve())
    root = Path(str(config["project_root"])).resolve()
    logger = setup_logger(root / "routing/logs/step25.log", args.overwrite)
    failure_path = (
        root / "routing/data/point_edge_mapping/step25_failure.json"
    )
    try:
        if args.mode == "run":
            rules = load_yaml(args.rules.resolve())
            if (
                not args.approved_by_user
                or not rules.get("approval", {}).get("approved")
            ):
                raise PermissionError(
                    "Formal Step25 run requires explicit user approval and an "
                    "approved matching rule file."
                )
            summary = run_formal(
                args.network_config.resolve(),
                args.rules.resolve(),
                args.overwrite,
                args.resume,
                logger,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if summary["success"] else 2
        summary = run_check(
            args.network_config.resolve(),
            args.rules.resolve(),
            args.overwrite,
            logger,
        )
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_POINT_EDGE_RULE_APPROVAL": summary[
                        "ready_for_point_edge_rule_approval"
                    ],
                    "READY_FOR_FORMAL_POINT_EDGE_RUN": False,
                    "point_count": summary["point_count"],
                    "move_dir_recovered_count": summary[
                        "move_dir_recovered_count"
                    ],
                    "coverage_by_distance": summary["coverage_by_distance"],
                    "confidence_preview_counts": summary[
                        "confidence_preview_counts"
                    ],
                    "manual_review_count": summary["manual_review_count"],
                    "failed_record_count": summary["failed_record_count"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_point_edge_rule_approval"] else 2
    except Exception as error:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            failure_path,
            {
                "status": "FAIL",
                "time": now_text(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        logger.exception("Step25 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
