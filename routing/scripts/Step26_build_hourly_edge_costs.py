"""Step26: hourly thermal edge-cost check and formal execution gate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.reporting import atomic_csv, atomic_json, atomic_text  # noqa: E402
from routing.src.thermal_edge_costs import (  # noqa: E402
    ThermalEdge,
    aggregate_direct_edge_hours,
    assign_along_edge_weights,
    node_key,
)
from routing.src.thermal_segment_costs import (  # noqa: E402
    run_formal_segment_costs,
)

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
    parser = argparse.ArgumentParser(description="Phase G5 thermal edge costs.")
    parser.add_argument(
        "--network-config",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/phase_g_network.yaml",
    )
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/thermal_edge_cost_draft.yaml",
    )
    parser.add_argument(
        "--expansion-rules",
        type=Path,
        default=(
            PROJECT_ROOT
            / "routing/configs/thermal_edge_coverage_expansion_draft.yaml"
        ),
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step26")
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


def _endpoints(geometry: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    from osgeo import ogr

    if ogr.GT_Flatten(geometry.GetGeometryType()) == ogr.wkbMultiLineString:
        first = geometry.GetGeometryRef(0)
        last = geometry.GetGeometryRef(geometry.GetGeometryCount() - 1)
    else:
        first = last = geometry
    start = first.GetPoint(0)
    end = last.GetPoint(last.GetPointCount() - 1)
    return (start[0], start[1]), (end[0], end[1])


def read_edges(path: Path) -> dict[str, ThermalEdge]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(path), 0)
    if dataset is None:
        raise FileNotFoundError(path)
    layer = dataset.GetLayerByName("ThermalComfortNetworkRepaired")
    if layer is None:
        raise KeyError("ThermalComfortNetworkRepaired")
    rows: dict[str, ThermalEdge] = {}
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Reading Step24 edges for thermal costs",
        unit="edge",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef().Clone()
        start, end = _endpoints(geometry)
        edge_id = str(feature.GetField("edge_id"))
        rows[edge_id] = ThermalEdge(
            edge_id=edge_id,
            fclass=str(feature.GetField("fclass") or "unknown"),
            bridge=str(feature.GetField("bridge") or "F"),
            tunnel=str(feature.GetField("tunnel") or "F"),
            layer=float(feature.GetField("layer") or 0.0),
            repair_type=str(feature.GetField("repair_typ") or "original"),
            length_m=float(geometry.Length()),
            start=start,
            end=end,
            geometry=geometry,
        )
    return rows


def compatible(first: ThermalEdge, second: ThermalEdge) -> bool:
    if first.repair_type == "gap_connector":
        return True
    if second.repair_type == "gap_connector":
        return True
    return (
        first.fclass == second.fclass
        and first.bridge == second.bridge
        and first.tunnel == second.tunnel
        and math.isclose(first.layer, second.layer)
    )


def interpolation_candidates(
    edges: dict[str, ThermalEdge],
    supported: set[str],
    maximum_distance: float,
) -> list[dict[str, Any]]:
    node_edges: dict[tuple[float, float], list[str]] = defaultdict(list)
    for edge in edges.values():
        node_edges[node_key(edge.start)].append(edge.edge_id)
        node_edges[node_key(edge.end)].append(edge.edge_id)
    records: list[dict[str, Any]] = []
    for edge in tqdm(
        edges.values(),
        desc="Auditing one-hop interpolation candidates",
        unit="edge",
        dynamic_ncols=True,
    ):
        if edge.edge_id in supported:
            continue
        adjacent = set(node_edges[node_key(edge.start)]) | set(
            node_edges[node_key(edge.end)]
        )
        candidates: list[tuple[float, str]] = []
        for source_id in adjacent - {edge.edge_id}:
            if source_id not in supported:
                continue
            source = edges[source_id]
            if not compatible(edge, source):
                continue
            distance = (edge.length_m + source.length_m) / 2
            if distance <= maximum_distance:
                candidates.append((distance, source_id))
        if candidates:
            distance, source_id = min(candidates)
            records.append(
                {
                    "edge_id": edge.edge_id,
                    "source_edge_id": source_id,
                    "interpolation_distance_m": distance,
                    "fclass": edge.fclass,
                    "bridge": edge.bridge,
                    "tunnel": edge.tunnel,
                    "layer": edge.layer,
                    "repair_type": edge.repair_type,
                    "shared_endpoint": True,
                    "compatible_class_and_level": True,
                    "formal_interpolation": False,
                }
            )
    return records


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
    output_dir = root / "routing/data/thermal_edges"
    report_path = root / "routing/reports/STEP26_THERMAL_EDGE_CHECK_REPORT.md"
    outputs = {
        "direct_preview": output_dir / "road_hour_direct_cost_preview.csv",
        "point_weights": output_dir / "point_edge_along_weights_preview.csv",
        "interpolation": output_dir / "edge_interpolation_candidates.csv",
        "coverage": output_dir / "edge_thermal_coverage_preview.csv",
        "failed": output_dir / "step26_failed_records.csv",
        "summary": output_dir / "step26_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step26 check outputs exist; pass --overwrite")
    step25 = json.loads(
        (
            root
            / "routing/data/point_edge_mapping/step25_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not step25.get("ready_for_step26_check"):
        blockers.append("Step25 formal mapping is not ready for Step26 check")
    mapping = pd.read_csv(
        root / "routing/data/point_edge_mapping/point_edge_mapping.csv",
        dtype={"point_id": str, "edge_id": str},
        low_memory=False,
    )
    matched = mapping[mapping["match_status"] == "matched"].copy()
    predictions = pd.read_csv(
        root
        / "training/predictions/full_sample/routing_point_costs_20240729.csv",
        dtype={"point_id": str},
        low_memory=False,
    )
    expected_hours = set(map(int, rules["hours"]))
    expected_prediction_rows = (
        int(config["runtime"]["expected_thermal_points"]) * len(expected_hours)
    )
    if len(predictions) != expected_prediction_rows:
        blockers.append(
            f"Expected {expected_prediction_rows} point-hour rows, found {len(predictions)}"
        )
    if set(predictions["hour"].unique()) != expected_hours:
        blockers.append("Prediction hours differ from approved 6-18 range")
    edges = read_edges(
        root / "routing/data/topology/step24_repaired_network.gdb"
    )
    step24 = json.loads(
        (root / "routing/data/topology/step24_formal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    expected_edge_count = int(step24["repaired_network_edge_count"])
    if len(edges) != expected_edge_count:
        blockers.append(
            f"Expected {expected_edge_count} edges, found {len(edges)}"
        )
    prediction_points = set(predictions["point_id"])
    missing_prediction_points = sorted(set(matched["point_id"]) - prediction_points)
    if missing_prediction_points:
        blockers.append(
            f"{len(missing_prediction_points)} mapped points lack predictions"
        )
    static = matched[
        ["point_id", "edge_id", "distance_m", "match_confidence"]
    ].merge(
        predictions[
            ["point_id", "utm_x", "utm_y"]
        ].drop_duplicates("point_id"),
        on="point_id",
        how="left",
        validate="one_to_one",
    )
    weighted_points = assign_along_edge_weights(
        static,
        edges,
        float(rules["direct_aggregation"]["minimum_weight_m"]),
    )
    matched_predictions = predictions.merge(
        weighted_points[
            [
                "point_id",
                "edge_id",
                "edge_position_m",
                "projection_distance_m",
                "along_edge_weight_m",
                "along_edge_weight",
                "match_confidence",
            ]
        ],
        on="point_id",
        how="inner",
        validate="many_to_one",
    )
    expected_matched_rows = len(matched) * len(expected_hours)
    if len(matched_predictions) != expected_matched_rows:
        blockers.append(
            f"Matched point-hour rows {len(matched_predictions)} != "
            f"{expected_matched_rows}"
        )
    direct_preview = aggregate_direct_edge_hours(matched_predictions)
    supported_edges = set(direct_preview["edge_id"].astype(str))
    interpolation = interpolation_candidates(
        edges,
        supported_edges,
        float(rules["interpolation_check"]["maximum_one_hop_distance_m"]),
    )
    interpolation_lookup = {
        row["edge_id"]: row for row in interpolation
    }
    coverage_records: list[dict[str, Any]] = []
    for edge in tqdm(
        edges.values(),
        desc="Classifying thermal edge coverage",
        unit="edge",
        dynamic_ncols=True,
    ):
        if edge.edge_id in supported_edges:
            coverage_type = "direct"
            source_edge_id = edge.edge_id
            interpolation_distance = 0.0
        elif edge.edge_id in interpolation_lookup:
            coverage_type = "interpolation_candidate"
            source_edge_id = interpolation_lookup[edge.edge_id]["source_edge_id"]
            interpolation_distance = interpolation_lookup[edge.edge_id][
                "interpolation_distance_m"
            ]
        else:
            coverage_type = "unknown"
            source_edge_id = None
            interpolation_distance = None
        coverage_records.append(
            {
                "edge_id": edge.edge_id,
                "length_m": edge.length_m,
                "fclass": edge.fclass,
                "bridge": edge.bridge,
                "tunnel": edge.tunnel,
                "layer": edge.layer,
                "repair_type": edge.repair_type,
                "coverage_type_preview": coverage_type,
                "candidate_source_edge_id": source_edge_id,
                "interpolation_distance_m": interpolation_distance,
                "formal_edge_cost_written": False,
            }
        )
    coverage = pd.DataFrame(coverage_records)
    coverage_counts = {
        str(key): int(value)
        for key, value in coverage[
            "coverage_type_preview"
        ].value_counts().items()
    }
    total_network_length_m = float(coverage["length_m"].sum())
    coverage_length = {
        str(key): {
            "length_km": float(members["length_m"].sum() / 1000),
            "length_ratio": float(
                members["length_m"].sum() / total_network_length_m
            ),
        }
        for key, members in coverage.groupby("coverage_type_preview")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["direct_preview"], direct_preview.to_dict("records"))
    atomic_csv(outputs["point_weights"], weighted_points.to_dict("records"))
    if interpolation:
        atomic_csv(outputs["interpolation"], interpolation)
    else:
        atomic_text(
            outputs["interpolation"],
            "edge_id,source_edge_id,interpolation_distance_m\n",
        )
    atomic_csv(outputs["coverage"], coverage_records)
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    ended_at = now_text()
    summary = {
        "phase": "G5",
        "step": 26,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_thermal_edge_rule_approval": len(blockers) == 0,
        "ready_for_formal_thermal_edge_run": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "network_edge_count": len(edges),
        "point_hour_input_count": len(predictions),
        "matched_point_count": len(matched),
        "matched_point_hour_count": len(matched_predictions),
        "direct_supported_edge_count": len(supported_edges),
        "direct_edge_hour_preview_count": len(direct_preview),
        "interpolation_candidate_edge_count": len(interpolation),
        "coverage_type_counts": coverage_counts,
        "coverage_length": coverage_length,
        "unknown_edge_count": int(
            coverage_counts.get("unknown", 0)
        ),
        "failed_record_count": 0,
        "rule_status": rules.get("status"),
        "rule_approved": bool(rules.get("approval", {}).get("approved")),
        "formal_edge_cost_written": False,
        "source_predictions_modified": False,
        "step25_mapping_modified": False,
        "route_search_performed": False,
        "output_paths": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step26 Thermal Edge Cost Check Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_THERMAL_EDGE_RULE_APPROVAL = {str(summary['ready_for_thermal_edge_rule_approval']).upper()}`",
        "- `READY_FOR_FORMAL_THERMAL_EDGE_RUN = FALSE`",
        f"- Network edges: {len(edges):,}",
        f"- Matched point-hour records: {len(matched_predictions):,}",
        f"- Direct supported edges: {len(supported_edges):,}",
        f"- Direct edge-hour previews: {len(direct_preview):,}",
        f"- Compatible one-hop interpolation candidates: {len(interpolation):,}",
        f"- Unknown edges: {summary['unknown_edge_count']:,}",
        f"- Direct supported length ratio: {coverage_length['direct']['length_ratio']:.6f}",
        f"- Unknown length ratio: {coverage_length['unknown']['length_ratio']:.6f}",
        "- Formal edge-hour costs written: 0",
        "",
        "Direct previews use along-edge Voronoi representative-length weights and mixture variance. Interpolation candidates share a real endpoint and preserve road class/bridge/tunnel/layer, except approved topology connectors which inherit adjacent thermal support.",
        "",
        "No unknown edge is zero-filled. No formal interpolation or route search was performed.",
    ]
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step26 check complete: ready=%s direct=%s interpolation=%s unknown=%s",
        summary["ready_for_thermal_edge_rule_approval"],
        len(supported_edges),
        len(interpolation),
        summary["unknown_edge_count"],
    )
    return summary


def main() -> int:
    args = parse_args()
    config = load_yaml(args.network_config.resolve())
    root = Path(str(config["project_root"])).resolve()
    logger = setup_logger(root / "routing/logs/step26.log", args.overwrite)
    failure_path = root / "routing/data/thermal_edges/step26_failure.json"
    try:
        if args.mode == "run":
            rules = load_yaml(args.rules.resolve())
            expansion_rules = load_yaml(args.expansion_rules.resolve())
            if (
                not args.approved_by_user
                or not expansion_rules.get("approval", {}).get("approved")
                or expansion_rules.get("status") != "APPROVED"
            ):
                raise PermissionError(
                    "Formal Step26 run requires explicit user approval and an "
                    "approved thermal coverage expansion rule file."
                )
            summary = run_formal_segment_costs(
                root=root,
                network_config=config,
                base_rules=rules,
                expansion_rules=expansion_rules,
                overwrite=args.overwrite,
                logger=logger,
            )
            if failure_path.exists():
                failure_path.unlink()
            print(
                json.dumps(
                    {
                        "SUCCESS": summary["success"],
                        "READY_FOR_STEP27_CHECK": summary[
                            "ready_for_step27_check"
                        ],
                        "center_thermal_segment_count": summary[
                            "center_thermal_segment_count"
                        ],
                        "formal_edge_hour_count": summary[
                            "formal_edge_hour_count"
                        ],
                        "prior_imputed_length_ratio": summary[
                            "prior_imputed_length_ratio"
                        ],
                        "failed_record_count": summary[
                            "failed_record_count"
                        ],
                        "report_path": summary["outputs"]["report"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
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
                    "READY_FOR_THERMAL_EDGE_RULE_APPROVAL": summary[
                        "ready_for_thermal_edge_rule_approval"
                    ],
                    "READY_FOR_FORMAL_THERMAL_EDGE_RUN": False,
                    "matched_point_hour_count": summary[
                        "matched_point_hour_count"
                    ],
                    "direct_supported_edge_count": summary[
                        "direct_supported_edge_count"
                    ],
                    "direct_edge_hour_preview_count": summary[
                        "direct_edge_hour_preview_count"
                    ],
                    "interpolation_candidate_edge_count": summary[
                        "interpolation_candidate_edge_count"
                    ],
                    "coverage_type_counts": summary["coverage_type_counts"],
                    "failed_record_count": summary["failed_record_count"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_thermal_edge_rule_approval"] else 2
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
        logger.exception("Step26 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
