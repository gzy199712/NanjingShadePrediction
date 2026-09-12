"""Phase G7 / Step28 route evaluation and mapping preflight."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.reporting import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_text,
)
from routing.src.route_engine import RouteEngine  # noqa: E402
from routing.src.route_evaluation import run_formal_step28  # noqa: E402

SCRIPT_VERSION = "2.0.0"


def now_text() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve(path: str) -> Path:
    return (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase G7 route evaluation.")
    parser.add_argument(
        "--rules",
        type=Path,
        default=(
            PROJECT_ROOT / "routing/configs/route_evaluation_draft.yaml"
        ),
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step28")
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


def truthy(values: pd.Series) -> np.ndarray:
    return (
        ~values.fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin({"", "0", "F", "FALSE", "N", "NO", "NONE"})
    ).to_numpy(bool)


def route_audit(
    *,
    engine: RouteEngine,
    routes: pd.DataFrame,
    summaries: pd.DataFrame,
    consistency: pd.DataFrame,
    coverage: pd.DataFrame,
    classification: pd.DataFrame,
    rules: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if not np.array_equal(
        coverage["segment_id"].to_numpy(str), engine.segment_ids
    ):
        raise ValueError("Coverage and formal graph segment order differ")
    segment_lookup = {
        value: index for index, value in enumerate(engine.segment_ids)
    }
    bridge = truthy(coverage["bridge"])
    tunnel = truthy(coverage["tunnel"])
    layer = np.round(coverage["layer"].to_numpy(float), 3)
    _, structure_key = np.unique(
        np.rec.fromarrays([bridge, tunnel, layer]), return_inverse=True
    )
    road_metadata = (
        classification[
            ["edge_id", "name", "ref", "fclass", "motor_vehicle_only"]
        ]
        .drop_duplicates("edge_id")
        .set_index("edge_id")
    )
    metadata = coverage[["original_edge_id"]].join(
        road_metadata, on="original_edge_id"
    )
    names = metadata["name"].fillna("").astype(str).to_numpy()
    refs = metadata["ref"].fillna("").astype(str).to_numpy()
    fclasses = metadata["fclass"].fillna("").astype(str).to_numpy()
    motor_only = (
        metadata["motor_vehicle_only"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    summary_lookup = {
        (str(row.case_id), str(row.algorithm)): row
        for row in summaries.itertuples(index=False)
    }
    case_lookup = {
        str(row.case_id): row for row in consistency.itertuples(index=False)
    }
    tolerance = float(rules["qc"]["cost_tolerance"])
    angle_limit = float(
        rules["grade_transition_review"][
            "maximum_unnamed_alignment_angle_deg"
        ]
    )
    audit_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    grouped = routes.sort_values(
        ["case_id", "algorithm", "sequence"]
    ).groupby(["case_id", "algorithm"], sort=True)
    for (case_id, algorithm), members in tqdm(
        grouped,
        total=grouped.ngroups,
        desc="Auditing Step28 validation routes",
        unit="route",
        dynamic_ncols=True,
    ):
        case = case_lookup[str(case_id)]
        summary = summary_lookup[(str(case_id), str(algorithm))]
        edge_indices = np.array(
            [segment_lookup[value] for value in members["segment_id"]],
            dtype=np.int32,
        )
        duplicate_count = int(
            len(edge_indices) - len(np.unique(edge_indices))
        )
        disconnected_count = 0
        suspicious_count = 0
        forbidden_turn_violation_count = 0
        structure_transition_count = 0
        for position, (first, second) in enumerate(
            zip(edge_indices[:-1], edge_indices[1:], strict=True), 1
        ):
            shared = {
                int(engine.edge_u[first]),
                int(engine.edge_v[first]),
            } & {
                int(engine.edge_u[second]),
                int(engine.edge_v[second]),
            }
            if not shared:
                disconnected_count += 1
                continue
            shared_node = next(iter(shared))
            if engine.is_turn_forbidden(shared_node, first, second):
                forbidden_turn_violation_count += 1
            if structure_key[first] == structure_key[second]:
                continue
            structure_transition_count += 1
            first_other = (
                int(engine.edge_v[first])
                if int(engine.edge_u[first]) == shared_node
                else int(engine.edge_u[first])
            )
            second_other = (
                int(engine.edge_v[second])
                if int(engine.edge_u[second]) == shared_node
                else int(engine.edge_u[second])
            )
            incoming = (
                engine.node_xy[shared_node] - engine.node_xy[first_other]
            )
            outgoing = (
                engine.node_xy[second_other] - engine.node_xy[shared_node]
            )
            denominator = float(
                np.linalg.norm(incoming) * np.linalg.norm(outgoing)
            )
            cosine = (
                float(np.dot(incoming, outgoing) / denominator)
                if denominator > 0
                else 1.0
            )
            angle = float(
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
            )
            same_identity = bool(
                (names[first] and names[first] == names[second])
                or (refs[first] and refs[first] == refs[second])
            )
            if same_identity:
                decision = "allow_named_transition"
            elif angle <= angle_limit:
                decision = "allow_aligned_transition"
            else:
                decision = "suspicious_grade_jump"
                suspicious_count += 1
            transition_rows.append(
                {
                    "case_id": case_id,
                    "algorithm": algorithm,
                    "transition_sequence": position,
                    "from_segment_id": engine.segment_ids[first],
                    "to_segment_id": engine.segment_ids[second],
                    "from_original_edge_id": engine.original_edge_ids[first],
                    "to_original_edge_id": engine.original_edge_ids[second],
                    "from_fclass": fclasses[first],
                    "to_fclass": fclasses[second],
                    "from_name": names[first],
                    "to_name": names[second],
                    "from_ref": refs[first],
                    "to_ref": refs[second],
                    "from_bridge": bool(bridge[first]),
                    "to_bridge": bool(bridge[second]),
                    "from_tunnel": bool(tunnel[first]),
                    "to_tunnel": bool(tunnel[second]),
                    "from_layer": float(layer[first]),
                    "to_layer": float(layer[second]),
                    "turn_angle_deg": angle,
                    "same_nonempty_name_or_ref": same_identity,
                    "review_decision": decision,
                    "formal_graph_repaired": False,
                }
            )
        include_fallback = bool(summary.fallback_retry) or (
            str(case.objective) == "shortest"
        )
        weights = engine._weights(
            int(case.hour),
            str(case.objective),
            1.0 if str(case.objective) == "risk_aware" else 0.0,
            include_fallback,
        )
        recomputed_cost = float(weights[edge_indices].sum())
        recomputed_distance = float(engine.lengths[edge_indices].sum())
        coverage_names = np.array(
            [
                engine.coverage_by_code[int(engine.coverage_code[index])]
                for index in edge_indices
            ]
        )
        fallback_length = float(
            engine.lengths[
                edge_indices[coverage_names == "spatial_fallback"]
            ].sum()
        )
        prior_length = float(
            engine.lengths[
                edge_indices[coverage_names == "prior_imputed"]
            ].sum()
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "algorithm": algorithm,
                "hour": int(case.hour),
                "mode": case.mode,
                "objective": case.objective,
                "edge_count": len(edge_indices),
                "duplicate_segment_count": duplicate_count,
                "disconnected_transition_count": disconnected_count,
                "structure_transition_count": structure_transition_count,
                "suspicious_grade_jump_count": suspicious_count,
                "forbidden_turn_violation_count": (
                    forbidden_turn_violation_count
                ),
                "motor_vehicle_only_edge_count": int(
                    motor_only[edge_indices].sum()
                ),
                "reported_cost": float(summary.total_cost),
                "recomputed_cost": recomputed_cost,
                "absolute_cost_difference": abs(
                    recomputed_cost - float(summary.total_cost)
                ),
                "cost_recompute_pass": abs(
                    recomputed_cost - float(summary.total_cost)
                )
                <= tolerance,
                "reported_distance_m": float(summary.distance_m),
                "recomputed_distance_m": recomputed_distance,
                "absolute_distance_difference_m": abs(
                    recomputed_distance - float(summary.distance_m)
                ),
                "reported_spatial_fallback_length_m": float(
                    summary.spatial_fallback_length_m
                ),
                "recomputed_spatial_fallback_length_m": fallback_length,
                "reported_prior_imputed_length_m": float(
                    summary.prior_imputed_length_m
                ),
                "recomputed_prior_imputed_length_m": prior_length,
                "origin_snap_distance_m": float(
                    summary.origin_snap_distance_m
                ),
                "destination_snap_distance_m": float(
                    summary.destination_snap_distance_m
                ),
                "within_snap_limit": (
                    float(summary.origin_snap_distance_m)
                    <= float(rules["qc"]["maximum_snap_distance_m"])
                    and float(summary.destination_snap_distance_m)
                    <= float(rules["qc"]["maximum_snap_distance_m"])
                ),
            }
        )
    unique_suspicious = {
        tuple(
            sorted(
                (
                    str(row["from_segment_id"]),
                    str(row["to_segment_id"]),
                )
            )
        )
        for row in transition_rows
        if row["review_decision"] == "suspicious_grade_jump"
    }
    metrics = {
        "route_count": len(audit_rows),
        "cost_recompute_fail_count": sum(
            not row["cost_recompute_pass"] for row in audit_rows
        ),
        "maximum_cost_recompute_difference": max(
            row["absolute_cost_difference"] for row in audit_rows
        ),
        "distance_recompute_fail_count": sum(
            row["absolute_distance_difference_m"] > tolerance
            for row in audit_rows
        ),
        "route_with_duplicate_segment_count": sum(
            row["duplicate_segment_count"] > 0 for row in audit_rows
        ),
        "route_with_disconnected_transition_count": sum(
            row["disconnected_transition_count"] > 0
            for row in audit_rows
        ),
        "route_with_motor_vehicle_only_edge_count": sum(
            row["motor_vehicle_only_edge_count"] > 0 for row in audit_rows
        ),
        "route_with_suspicious_grade_jump_count": sum(
            row["suspicious_grade_jump_count"] > 0 for row in audit_rows
        ),
        "route_with_forbidden_turn_violation_count": sum(
            row["forbidden_turn_violation_count"] > 0
            for row in audit_rows
        ),
        "forbidden_turn_violation_record_count": sum(
            row["forbidden_turn_violation_count"] for row in audit_rows
        ),
        "suspicious_grade_jump_record_count": sum(
            row["review_decision"] == "suspicious_grade_jump"
            for row in transition_rows
        ),
        "unique_suspicious_grade_transition_count": len(unique_suspicious),
        "route_outside_snap_limit_count": sum(
            not row["within_snap_limit"] for row in audit_rows
        ),
    }
    return audit_rows, transition_rows, metrics


def graph_grade_audit(
    *,
    engine: RouteEngine,
    coverage: pd.DataFrame,
    classification: pd.DataFrame,
    angle_limit: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    bridge = truthy(coverage["bridge"])
    tunnel = truthy(coverage["tunnel"])
    layer = np.round(coverage["layer"].to_numpy(float), 3)
    _, structure_key = np.unique(
        np.rec.fromarrays([bridge, tunnel, layer]), return_inverse=True
    )
    road_metadata = (
        classification[["edge_id", "name", "ref", "fclass"]]
        .drop_duplicates("edge_id")
        .set_index("edge_id")
    )
    metadata = coverage[["original_edge_id"]].join(
        road_metadata, on="original_edge_id"
    )
    names = metadata["name"].fillna("").astype(str).to_numpy()
    refs = metadata["ref"].fillna("").astype(str).to_numpy()
    fclasses = metadata["fclass"].fillna("").astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    mixed_node_count = 0
    incompatible_pair_count = 0
    protected_pair_count = 0
    unprotected_pair_count = 0
    for node in tqdm(
        range(len(engine.node_xy)),
        desc="Auditing all graph grade transitions",
        unit="node",
        dynamic_ncols=True,
    ):
        edges = np.unique(
            engine.adjacency_edges[
                int(engine.offsets[node]) : int(engine.offsets[node + 1])
            ]
        )
        if len(edges) < 2 or len(np.unique(structure_key[edges])) < 2:
            continue
        mixed_node_count += 1
        for first_position in range(len(edges) - 1):
            for second_position in range(first_position + 1, len(edges)):
                first = int(edges[first_position])
                second = int(edges[second_position])
                if structure_key[first] == structure_key[second]:
                    continue
                incompatible_pair_count += 1
                first_other = (
                    int(engine.edge_v[first])
                    if int(engine.edge_u[first]) == node
                    else int(engine.edge_u[first])
                )
                second_other = (
                    int(engine.edge_v[second])
                    if int(engine.edge_u[second]) == node
                    else int(engine.edge_u[second])
                )
                incoming = engine.node_xy[node] - engine.node_xy[first_other]
                outgoing = engine.node_xy[second_other] - engine.node_xy[node]
                denominator = float(
                    np.linalg.norm(incoming) * np.linalg.norm(outgoing)
                )
                cosine = (
                    float(np.dot(incoming, outgoing) / denominator)
                    if denominator > 0
                    else 1.0
                )
                angle = float(
                    np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                )
                same_identity = bool(
                    (names[first] and names[first] == names[second])
                    or (refs[first] and refs[first] == refs[second])
                )
                if same_identity or angle <= angle_limit:
                    continue
                protected = engine.is_turn_forbidden(node, first, second)
                if protected:
                    protected_pair_count += 1
                else:
                    unprotected_pair_count += 1
                rows.append(
                    {
                        "node_id": node,
                        "node_x": float(engine.node_xy[node, 0]),
                        "node_y": float(engine.node_xy[node, 1]),
                        "from_segment_id": engine.segment_ids[first],
                        "to_segment_id": engine.segment_ids[second],
                        "from_original_edge_id": engine.original_edge_ids[first],
                        "to_original_edge_id": engine.original_edge_ids[second],
                        "from_fclass": fclasses[first],
                        "to_fclass": fclasses[second],
                        "from_name": names[first],
                        "to_name": names[second],
                        "from_ref": refs[first],
                        "to_ref": refs[second],
                        "from_bridge": bool(bridge[first]),
                        "to_bridge": bool(bridge[second]),
                        "from_tunnel": bool(tunnel[first]),
                        "to_tunnel": bool(tunnel[second]),
                        "from_layer": float(layer[first]),
                        "to_layer": float(layer[second]),
                        "turn_angle_deg": angle,
                        "same_nonempty_name_or_ref": same_identity,
                        "review_decision": (
                            "forbidden_turn_applied"
                            if protected
                            else "suspicious_grade_jump"
                        ),
                        "formal_graph_repaired": protected,
                    }
                )
    return rows, {
        "mixed_grade_node_count": mixed_node_count,
        "incompatible_structure_pair_count": incompatible_pair_count,
        "graph_grade_transition_candidate_count": len(rows),
        "graph_protected_grade_transition_count": protected_pair_count,
        "graph_suspicious_grade_transition_count": unprotected_pair_count,
    }


def run_check(
    rules_path: Path, overwrite: bool, logger: logging.Logger
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    rules = load_yaml(rules_path)
    evaluation_dir = PROJECT_ROOT / "routing/data/evaluation"
    report_path = (
        PROJECT_ROOT / "routing/reports/STEP28_EVALUATION_CHECK_REPORT.md"
    )
    outputs = {
        "route_qc": evaluation_dir / "step28_route_qc.csv",
        "grade_transitions": (
            evaluation_dir / "step28_grade_transition_review.csv"
        ),
        "graph_grade_candidates": (
            evaluation_dir
            / "step28_graph_grade_transition_candidates.csv"
        ),
        "failed": evaluation_dir / "step28_failed_records.csv",
        "summary": evaluation_dir / "step28_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step28 check outputs exist; pass --overwrite")
    step27 = json.loads(
        (
            PROJECT_ROOT / "routing/data/graph/step27_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not step27.get("ready_for_step28_check"):
        blockers.append("Step27 formal engine is not ready for Step28")
    inputs = {key: resolve(value) for key, value in rules["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        blockers.extend(f"Missing input: {value}" for value in missing)
    engine = RouteEngine(
        inputs["formal_graph"],
        inputs["formal_costs"],
        inputs["graph_metadata"],
    )
    routes = pd.read_csv(
        inputs["validation_routes"], dtype={"segment_id": str}
    )
    summaries = pd.read_csv(inputs["validation_summaries"])
    consistency = pd.read_csv(inputs["algorithm_consistency"])
    coverage = pd.read_csv(
        inputs["segment_coverage"],
        dtype={"segment_id": str, "original_edge_id": str},
        low_memory=False,
    )
    classification = pd.read_csv(
        inputs["road_classification"],
        dtype={"edge_id": str},
        low_memory=False,
    )
    audit_rows, transition_rows, metrics = route_audit(
        engine=engine,
        routes=routes,
        summaries=summaries,
        consistency=consistency,
        coverage=coverage,
        classification=classification,
        rules=rules,
    )
    graph_grade_rows, graph_grade_metrics = graph_grade_audit(
        engine=engine,
        coverage=coverage,
        classification=classification,
        angle_limit=float(
            rules["grade_transition_review"][
                "maximum_unnamed_alignment_angle_deg"
            ]
        ),
    )
    metrics.update(graph_grade_metrics)
    if metrics["cost_recompute_fail_count"]:
        blockers.append("Route cost recomputation failed")
    if metrics["distance_recompute_fail_count"]:
        blockers.append("Route distance recomputation failed")
    if metrics["route_with_duplicate_segment_count"]:
        blockers.append("Validation routes contain repeated segments")
    if metrics["route_with_disconnected_transition_count"]:
        blockers.append("Validation routes contain disconnected transitions")
    if metrics["route_with_motor_vehicle_only_edge_count"]:
        blockers.append("Validation routes contain motor-vehicle-only edges")
    if metrics["route_outside_snap_limit_count"]:
        blockers.append("Validation routes exceed OD snap limit")
    if metrics["route_with_forbidden_turn_violation_count"]:
        blockers.append("Validation routes violate approved forbidden turns")
    if metrics["graph_suspicious_grade_transition_count"]:
        blockers.append(
            f"{metrics['graph_suspicious_grade_transition_count']} graph-wide "
            "suspicious bridge/tunnel/layer turns require graph repair review"
        )
    warnings.append(
        "Bridge/tunnel transition legality uses name/ref and 45-degree "
        "alignment as a conservative check because explicit ramp/access "
        "fields are unavailable."
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["route_qc"], audit_rows)
    atomic_csv(outputs["grade_transitions"], transition_rows)
    atomic_csv(outputs["graph_grade_candidates"], graph_grade_rows)
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    ended_at = now_text()
    summary = {
        "phase": "G7",
        "step": 28,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_grade_transition_review": (
            metrics["graph_suspicious_grade_transition_count"] > 0
        ),
        "ready_for_formal_step28_run": len(blockers) == 0,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "route_metrics": metrics,
        "algorithm_consistency_pass_count": int(
            consistency["consistency_pass"].sum()
        ),
        "algorithm_consistency_case_count": len(consistency),
        "formal_gis_generated": False,
        "formal_figures_generated": False,
        "step27_graph_modified": False,
        "failed_record_count": 0,
        "rule_status": rules["status"],
        "outputs": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step28 Route Evaluation Check Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_GRADE_TRANSITION_REVIEW = {str(summary['ready_for_grade_transition_review']).upper()}`",
        f"- `READY_FOR_FORMAL_STEP28_RUN = {str(summary['ready_for_formal_step28_run']).upper()}`",
        f"- Audited algorithm routes: {metrics['route_count']:,}",
        f"- Cost recomputation failures: {metrics['cost_recompute_fail_count']:,}",
        f"- Distance recomputation failures: {metrics['distance_recompute_fail_count']:,}",
        f"- Routes with repeated segments: {metrics['route_with_duplicate_segment_count']:,}",
        f"- Routes with disconnected transitions: {metrics['route_with_disconnected_transition_count']:,}",
        f"- Routes with motor-vehicle-only edges: {metrics['route_with_motor_vehicle_only_edge_count']:,}",
        f"- Routes with suspicious grade jumps: {metrics['route_with_suspicious_grade_jump_count']:,}",
        f"- Routes violating forbidden turns: {metrics['route_with_forbidden_turn_violation_count']:,}",
        f"- Suspicious transition records: {metrics['suspicious_grade_jump_record_count']:,}",
        f"- Unique suspicious transitions: {metrics['unique_suspicious_grade_transition_count']:,}",
        f"- Mixed-grade graph nodes: {metrics['mixed_grade_node_count']:,}",
        f"- Protected forbidden grade turns: {metrics['graph_protected_grade_transition_count']:,}",
        f"- Graph-wide suspicious turns: {metrics['graph_suspicious_grade_transition_count']:,}",
        "",
        (
            "The approved node-specific forbidden-turn table is active. "
            "No unprotected suspicious bridge/tunnel/layer transition "
            "remains; Step28 is ready for a separate formal-run approval."
            if summary["ready_for_formal_step28_run"]
            else
            "Formal Step28 evaluation and maps are blocked until suspicious "
            "bridge/tunnel/layer transitions are reviewed and the formal "
            "graph is repaired or explicitly approved."
        ),
        "No GIS or figure output was generated in this check.",
    ]
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step28 check complete: ready=%s routes=%s suspicious=%s",
        summary["ready_for_formal_step28_run"],
        metrics["route_count"],
        metrics["graph_suspicious_grade_transition_count"],
    )
    return summary


def main() -> int:
    args = parse_args()
    logger = setup_logger(
        PROJECT_ROOT / "routing/logs/step28.log", args.overwrite
    )
    failure_path = (
        PROJECT_ROOT / "routing/data/evaluation/step28_failure.json"
    )
    try:
        if args.mode == "run":
            rules = load_yaml(args.rules.resolve())
            if (
                not args.approved_by_user
                or not rules.get("approval", {}).get("approved")
                or rules.get("status") != "APPROVED"
            ):
                raise PermissionError(
                    "Formal Step28 requires explicit approval and an "
                    "APPROVED evaluation rule file."
                )
            summary = run_formal_step28(
                root=PROJECT_ROOT,
                rules=rules,
                overwrite=args.overwrite,
                logger=logger,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(
                json.dumps(
                    {
                        "SUCCESS": summary["success"],
                        "READY_FOR_PHASE_G8": summary[
                            "ready_for_phase_g8"
                        ],
                        "route_count": summary["route_count"],
                        "od_count": summary["od_count"],
                        "representative_od": summary[
                            "representative_od"
                        ],
                        "formal_gis_generated": summary[
                            "formal_gis_generated"
                        ],
                        "formal_figures_generated": summary[
                            "formal_figures_generated"
                        ],
                        "report_path": summary["outputs"]["report"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0 if summary["success"] else 2
        summary = run_check(args.rules.resolve(), args.overwrite, logger)
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_GRADE_TRANSITION_REVIEW": summary[
                        "ready_for_grade_transition_review"
                    ],
                    "READY_FOR_FORMAL_STEP28_RUN": summary[
                        "ready_for_formal_step28_run"
                    ],
                    "blocker_count": summary["blocker_count"],
                    "route_metrics": summary["route_metrics"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_formal_step28_run"] else 2
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
        logger.exception("Step28 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
