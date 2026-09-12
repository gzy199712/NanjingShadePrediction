"""Phase G6 / Step27 route-search preflight and formal execution gate."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
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
from routing.src.route_validation import run_formal_step27  # noqa: E402

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
    parser = argparse.ArgumentParser(description="Phase G6 route search.")
    parser.add_argument(
        "--rules",
        type=Path,
        default=PROJECT_ROOT / "routing/configs/route_search_draft.yaml",
    )
    parser.add_argument("--mode", required=True, choices=("check", "run"))
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path, overwrite: bool) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("phase_g_step27")
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


def endpoint_key(point: tuple[float, ...]) -> tuple[float, float]:
    return round(float(point[0]), 3), round(float(point[1]), 3)


def segment_endpoints(geometry: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    line = geometry.GetGeometryRef(0) if geometry.GetGeometryCount() else geometry
    if line.GetPointCount() < 2:
        raise ValueError("Segment geometry has fewer than two points")
    return endpoint_key(line.GetPoint(0)), endpoint_key(
        line.GetPoint(line.GetPointCount() - 1)
    )


def graph_metrics(graph: nx.Graph, scope: str) -> dict[str, Any]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    components = list(nx.connected_components(graph))
    node_sizes = sorted((len(value) for value in components), reverse=True)
    edge_sizes = sorted(
        (graph.subgraph(value).number_of_edges() for value in components),
        reverse=True,
    )
    reachable_pairs = sum(size * (size - 1) / 2 for size in node_sizes)
    possible_pairs = node_count * (node_count - 1) / 2
    degrees = dict(graph.degree())
    return {
        "network_scope": scope,
        "node_count": node_count,
        "edge_count": edge_count,
        "connected_component_count": len(components),
        "largest_component_node_count": node_sizes[0] if node_sizes else 0,
        "largest_component_node_ratio": (
            node_sizes[0] / node_count if node_sizes and node_count else 0
        ),
        "largest_component_edge_count": edge_sizes[0] if edge_sizes else 0,
        "largest_component_edge_ratio": (
            edge_sizes[0] / edge_count if edge_sizes and edge_count else 0
        ),
        "random_node_pair_reachable_ratio": (
            reachable_pairs / possible_pairs if possible_pairs else 0
        ),
        "degree_1_nodes": sum(value == 1 for value in degrees.values()),
        "degree_2_nodes": sum(value == 2 for value in degrees.values()),
        "degree_ge_3_nodes": sum(value >= 3 for value in degrees.values()),
        "largest_component_node_counts_top10": ";".join(
            map(str, node_sizes[:10])
        ),
        "largest_component_edge_counts_top10": ";".join(
            map(str, edge_sizes[:10])
        ),
    }


def audit_graph(
    gdb_path: Path,
    failures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb_path), 0)
    if dataset is None:
        raise FileNotFoundError(gdb_path)
    layer = dataset.GetLayerByName("ThermalCostSegments")
    if layer is None:
        raise KeyError("ThermalCostSegments")
    graphs = {
        "all_segments": nx.MultiGraph(),
        "thermal_primary": nx.MultiGraph(),
        "normal_confidence": nx.MultiGraph(),
    }
    coverage_counts: Counter[str] = Counter()
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Auditing Step27 graph connectivity",
        unit="segment",
        dynamic_ncols=True,
    ):
        segment_id = str(feature.GetField("segment_id"))
        try:
            geometry = feature.GetGeometryRef()
            if geometry is None or geometry.IsEmpty():
                raise ValueError("Empty geometry")
            first, second = segment_endpoints(geometry)
            coverage = str(feature.GetField("coverage"))
            routing_policy = str(feature.GetField("routing"))
            coverage_counts[coverage] += 1
            graphs["all_segments"].add_edge(
                first, second, segment_id=segment_id
            )
            if routing_policy != "fallback_only":
                graphs["thermal_primary"].add_edge(
                    first, second, segment_id=segment_id
                )
            if routing_policy == "normal":
                graphs["normal_confidence"].add_edge(
                    first, second, segment_id=segment_id
                )
        except Exception as error:
            failures.append(
                {
                    "category": "graph_geometry",
                    "object_id": segment_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    rows = [
        graph_metrics(graph, name) for name, graph in graphs.items()
    ]
    return rows, coverage_counts, int(layer.GetFeatureCount())


def audit_costs(
    path: Path,
    expected_segments: int,
    required_hours: set[int],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    required_columns = {
        "segment_id",
        "original_edge_id",
        "hour",
        "length_m",
        "shade_mean",
        "shade_std",
        "tmrt_mean",
        "tmrt_std",
        "utci_mean",
        "utci_std",
        "utci_cost_norm",
        "uncertainty_cost",
        "uncertainty_cost_norm",
        "coverage_type",
        "coverage_confidence",
        "routing_policy",
    }
    header = set(pd.read_csv(path, nrows=0).columns)
    missing_columns = sorted(required_columns - header)
    rows_by_hour: Counter[int] = Counter()
    rows_by_coverage: Counter[str] = Counter()
    ranges: dict[tuple[int, str], list[float]] = {}
    nonfinite_count = 0
    total_rows = 0
    columns = [
        "hour",
        "shade_mean",
        "shade_std",
        "tmrt_mean",
        "tmrt_std",
        "utci_mean",
        "utci_std",
        "utci_cost_norm",
        "uncertainty_cost",
        "uncertainty_cost_norm",
        "coverage_type",
    ]
    if missing_columns:
        return [], {}, missing_columns
    for chunk in tqdm(
        pd.read_csv(path, usecols=columns, chunksize=200_000),
        desc="Auditing Step27 hourly costs",
        unit="chunk",
        dynamic_ncols=True,
    ):
        total_rows += len(chunk)
        for key, value in chunk["hour"].value_counts().items():
            rows_by_hour[int(key)] += int(value)
        for key, value in chunk["coverage_type"].value_counts().items():
            rows_by_coverage[str(key)] += int(value)
        numeric_columns = [
            column for column in columns if column not in {"hour", "coverage_type"}
        ]
        numeric = chunk[numeric_columns].to_numpy(float)
        nonfinite_count += int((~np.isfinite(numeric)).sum())
        for hour, members in chunk.groupby("hour"):
            for column in numeric_columns:
                key = (int(hour), column)
                current = ranges.setdefault(key, [math.inf, -math.inf])
                current[0] = min(current[0], float(members[column].min()))
                current[1] = max(current[1], float(members[column].max()))
    range_rows = [
        {
            "hour": hour,
            "field": field,
            "minimum": values[0],
            "maximum": values[1],
            "within_unit_interval": (
                values[0] >= -1e-9 and values[1] <= 1 + 1e-9
                if field.endswith("_norm")
                else None
            ),
        }
        for (hour, field), values in sorted(ranges.items())
    ]
    summary = {
        "total_cost_rows": total_rows,
        "expected_cost_rows": expected_segments * len(required_hours),
        "rows_by_hour": dict(sorted(rows_by_hour.items())),
        "rows_by_coverage": dict(sorted(rows_by_coverage.items())),
        "observed_hours": sorted(rows_by_hour),
        "expected_hours": sorted(required_hours),
        "all_hours_have_one_row_per_segment": all(
            rows_by_hour[hour] == expected_segments for hour in required_hours
        ),
        "nonfinite_numeric_value_count": nonfinite_count,
        "utci_norm_outside_unit_interval": any(
            row["field"] == "utci_cost_norm"
            and not row["within_unit_interval"]
            for row in range_rows
        ),
        "uncertainty_norm_outside_unit_interval": any(
            row["field"] == "uncertainty_cost_norm"
            and not row["within_unit_interval"]
            for row in range_rows
        ),
    }
    return range_rows, summary, missing_columns


def run_check(
    rules_path: Path, overwrite: bool, logger: logging.Logger
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    rules = load_yaml(rules_path)
    output_dir = PROJECT_ROOT / "routing/data/graph"
    report_path = PROJECT_ROOT / "routing/reports/STEP27_ROUTE_CHECK_REPORT.md"
    outputs = {
        "connectivity": output_dir / "step27_graph_connectivity_qc.csv",
        "cost_fields": output_dir / "step27_cost_field_qc.csv",
        "failed": output_dir / "step27_failed_records.csv",
        "summary": output_dir / "step27_check_summary.json",
    }
    if any(path.exists() for path in outputs.values()) and not overwrite:
        raise FileExistsError("Step27 check outputs exist; pass --overwrite")
    step26 = json.loads(
        (
            PROJECT_ROOT
            / "routing/data/thermal_edges/step26_formal_summary.json"
        ).read_text(encoding="utf-8")
    )
    blockers: list[str] = []
    warnings: list[str] = []
    if not step26.get("ready_for_step27_check"):
        blockers.append("Step26 formal output is not ready for Step27")
    failures: list[dict[str, Any]] = []
    graph_rows, coverage_counts, segment_count = audit_graph(
        PROJECT_ROOT
        / "routing/data/thermal_edges/step26_thermal_segments.gdb",
        failures,
    )
    cost_rows, cost_summary, missing_columns = audit_costs(
        PROJECT_ROOT
        / "routing/data/thermal_edges/road_hour_thermal_cost.csv",
        segment_count,
        set(map(int, rules["interface"]["hours"])),
    )
    if missing_columns:
        blockers.append(
            "Missing required Step27 cost columns: " + ", ".join(missing_columns)
        )
    if failures:
        blockers.append(f"{len(failures)} graph geometries failed")
    all_graph = next(
        row for row in graph_rows if row["network_scope"] == "all_segments"
    )
    primary_graph = next(
        row for row in graph_rows if row["network_scope"] == "thermal_primary"
    )
    if all_graph["largest_component_edge_ratio"] < 0.60:
        blockers.append("Largest all-segment component contains under 60% of edges")
    if all_graph["connected_component_count"] > 1:
        warnings.append(
            f"Center network has {all_graph['connected_component_count']} "
            "components; cross-component OD pairs must return no_path"
        )
    if primary_graph["largest_component_edge_ratio"] < 0.50:
        warnings.append(
            "Thermal-primary graph largest component contains under 50% of "
            "eligible edges; fallback retry may be needed often"
        )
    if cost_summary.get("uncertainty_norm_outside_unit_interval"):
        warnings.append(
            "Step26 uncertainty_cost_norm exceeds [0,1]; approved Step27 "
            "must recompute per-hour min-max normalization"
        )
    if cost_summary and (
        cost_summary["total_cost_rows"] != cost_summary["expected_cost_rows"]
        or not cost_summary["all_hours_have_one_row_per_segment"]
        or cost_summary["nonfinite_numeric_value_count"] != 0
    ):
        blockers.append("Formal Step26 hourly costs failed Step27 completeness QC")
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(outputs["connectivity"], graph_rows)
    atomic_csv(outputs["cost_fields"], cost_rows)
    if failures:
        atomic_csv(outputs["failed"], failures)
    else:
        atomic_text(
            outputs["failed"],
            "category,object_id,error_type,error_message\n",
        )
    ended_at = now_text()
    summary = {
        "phase": "G6",
        "step": 27,
        "mode": "check",
        "script": Path(__file__).name,
        "script_version": SCRIPT_VERSION,
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "ready_for_route_rule_approval": len(blockers) == 0,
        "ready_for_formal_route_search": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "warning_count": len(warnings),
        "warnings": warnings,
        "segment_count": segment_count,
        "coverage_segment_counts": dict(sorted(coverage_counts.items())),
        "graph_metrics": graph_rows,
        "cost_summary": cost_summary,
        "mode_network_policy": rules["mode_networks"],
        "rule_status": rules["status"],
        "formal_graph_written": False,
        "astar_executed": False,
        "dijkstra_executed": False,
        "formal_route_search_performed": False,
        "failed_record_count": len(failures),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "report_path": str(report_path),
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step27 Route Search Check Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `READY_FOR_ROUTE_RULE_APPROVAL = {str(summary['ready_for_route_rule_approval']).upper()}`",
        "- `READY_FOR_FORMAL_ROUTE_SEARCH = FALSE`",
        f"- Formal thermal segments: {segment_count:,}",
        f"- Blockers: {len(blockers)}",
        f"- Warnings: {len(warnings)}",
        "- A* runs: 0",
        "- Dijkstra runs: 0",
        "",
        "## Connectivity",
        "",
        "| Scope | Nodes | Edges | Components | Largest edge ratio | Reachable random node-pair ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in graph_rows:
        lines.append(
            f"| {row['network_scope']} | {row['node_count']:,} | "
            f"{row['edge_count']:,} | {row['connected_component_count']:,} | "
            f"{row['largest_component_edge_ratio']:.4%} | "
            f"{row['random_node_pair_reachable_ratio']:.4%} |"
        )
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- {value}" for value in warnings)
    if not warnings:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Draft routing policy",
            "",
            "- walk, bike and shared currently use the same user-approved "
            "thermal-comfort-relevant network because the two-class override "
            "does not provide reliable finer access fields.",
            "- shortest allows all segments and reports fallback lengths.",
            "- thermal objectives first exclude fallback-only segments; when "
            "disconnected, a clearly reported high-penalty retry is proposed.",
            "- origin and destination must snap into the same component; "
            "cross-component requests return an explicit no_path reason.",
            "- UTCI and uncertainty are normalized independently by hour. "
            "Raw values remain in route outputs.",
            "- A* cannot become the default until fixed-OD Dijkstra "
            "consistency tests pass in the formal run.",
            "",
            "No path search or formal graph serialization was performed.",
        ]
    )
    atomic_text(report_path, "\n".join(lines) + "\n")
    logger.info(
        "Step27 check complete: ready=%s segments=%s components=%s",
        summary["ready_for_route_rule_approval"],
        segment_count,
        all_graph["connected_component_count"],
    )
    return summary


def main() -> int:
    args = parse_args()
    logger = setup_logger(
        PROJECT_ROOT / "routing/logs/step27.log", args.overwrite
    )
    failure_path = PROJECT_ROOT / "routing/data/graph/step27_failure.json"
    try:
        if args.mode == "run":
            rules = load_yaml(args.rules.resolve())
            if (
                not args.approved_by_user
                or not rules.get("approval", {}).get("approved")
                or rules.get("status") != "APPROVED"
            ):
                raise PermissionError(
                    "Formal Step27 requires explicit user approval and an "
                    "APPROVED route-search rule file."
                )
            summary = run_formal_step27(
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
                        "READY_FOR_STEP28_CHECK": summary[
                            "ready_for_step28_check"
                        ],
                        "formal_graph_node_count": summary[
                            "formal_graph_node_count"
                        ],
                        "formal_graph_edge_count": summary[
                            "formal_graph_edge_count"
                        ],
                        "validation_case_count": summary[
                            "validation_case_count"
                        ],
                        "consistency_pass_count": summary[
                            "consistency_pass_count"
                        ],
                        "maximum_absolute_cost_difference": summary[
                            "maximum_absolute_cost_difference"
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
        summary = run_check(args.rules.resolve(), args.overwrite, logger)
        if failure_path.exists():
            failure_path.unlink()
        print(
            json.dumps(
                {
                    "READY_FOR_ROUTE_RULE_APPROVAL": summary[
                        "ready_for_route_rule_approval"
                    ],
                    "READY_FOR_FORMAL_ROUTE_SEARCH": False,
                    "segment_count": summary["segment_count"],
                    "blocker_count": summary["blocker_count"],
                    "warning_count": summary["warning_count"],
                    "report_path": summary["report_path"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0 if summary["ready_for_route_rule_approval"] else 2
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
        logger.exception("Step27 failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
