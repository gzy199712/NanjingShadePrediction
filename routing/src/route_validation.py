"""Formal Step27 artifact build and A*/Dijkstra consistency validation."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from routing.src.reporting import atomic_csv, atomic_json, atomic_text
from routing.src.route_engine import RouteEngine, build_graph_artifacts
from routing.src.thermal_segment_costs import now_text, sha256_path


def _sample_od_pairs(
    engine: RouteEngine,
    count: int,
    seed: int,
    minimum_distance_m: float,
) -> list[tuple[int, int]]:
    labels, counts = np.unique(engine.component_all, return_counts=True)
    largest_label = int(labels[np.argmax(counts)])
    candidates = np.flatnonzero(engine.component_all == largest_label)
    rng = np.random.default_rng(seed)
    pairs: list[tuple[int, int]] = []
    observed: set[tuple[int, int]] = set()
    attempts = 0
    with tqdm(
        total=count,
        desc="Sampling turn-reachable validation ODs",
        unit="OD",
        dynamic_ncols=True,
    ) as progress:
        while len(pairs) < count and attempts < count * 20_000:
            attempts += 1
            first, second = map(
                int, rng.choice(candidates, size=2, replace=False)
            )
            straight = float(
                np.linalg.norm(
                    engine.node_xy[first] - engine.node_xy[second]
                )
            )
            if straight < minimum_distance_m or straight > 8_000:
                continue
            key = tuple(sorted((first, second)))
            if key in observed:
                continue
            observed.add(key)
            reachable, _ = engine.route_nodes(
                first,
                second,
                12,
                mode="walk",
                objective="shortest",
                algorithm="dijkstra",
            )
            if not reachable.found:
                continue
            pairs.append((first, second))
            progress.update(1)
    if len(pairs) != count:
        raise RuntimeError(f"Could only sample {len(pairs)} of {count} OD pairs")
    return pairs


def run_formal_step27(
    *,
    root: Path,
    rules: dict[str, Any],
    overwrite: bool,
    logger: Any,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    graph_dir = root / "routing/data/graph"
    tables_dir = root / "routing/outputs/tables"
    routes_dir = root / "routing/outputs/routes/validation"
    reports_dir = root / "routing/reports"
    outputs = {
        "graph": graph_dir / "step27_route_graph.npz",
        "costs": graph_dir / "step27_hourly_costs.npz",
        "metadata": graph_dir / "step27_graph_metadata.json",
        "od_samples": tables_dir / "step27_validation_od_samples.csv",
        "consistency": tables_dir / "step27_algorithm_consistency.csv",
        "route_summary": tables_dir / "step27_validation_route_summary.csv",
        "route_edges": routes_dir / "step27_validation_route_edges.csv",
        "failed": graph_dir / "step27_formal_failed_records.csv",
        "summary": graph_dir / "step27_formal_summary.json",
        "qc": graph_dir / "step27_formal_qc.json",
        "report": reports_dir / "STEP27_FORMAL_RUN_REPORT.md",
    }
    for path in outputs.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Step27 formal output exists: {path}")
    source_paths = {
        "segments_gdb": (
            root
            / "routing/data/thermal_edges/step26_thermal_segments.gdb"
        ),
        "coverage": (
            root
            / "routing/data/thermal_edges/thermal_segment_coverage.csv"
        ),
        "hourly_costs": (
            root
            / "routing/data/thermal_edges/road_hour_thermal_cost.csv"
        ),
        "turn_restrictions": (
            root / rules["turn_restrictions"]["source_csv"]
        ),
    }
    source_hashes_before = {
        key: sha256_path(path) for key, path in source_paths.items()
    }
    metadata = build_graph_artifacts(
        coverage_path=source_paths["coverage"],
        gdb_path=source_paths["segments_gdb"],
        costs_path=source_paths["hourly_costs"],
        graph_output=outputs["graph"],
        cost_output=outputs["costs"],
        metadata_output=outputs["metadata"],
        rules=rules,
        turn_restrictions_path=source_paths["turn_restrictions"],
    )
    engine = RouteEngine(
        outputs["graph"], outputs["costs"], outputs["metadata"]
    )
    validation = rules["algorithm_validation"]
    case_count = int(validation["fixed_od_sample_count"])
    objective_names = ["shortest", "shade", "utci", "risk_aware"]
    if case_count % len(objective_names) != 0:
        raise ValueError("fixed_od_sample_count must be divisible by 4")
    od_count = case_count // len(objective_names)
    od_pairs = _sample_od_pairs(
        engine,
        od_count,
        int(validation["random_seed"]),
        float(validation["minimum_od_separation_m"]),
    )
    od_rows = []
    for od_index, (origin_node, destination_node) in enumerate(od_pairs, 1):
        origin = engine.node_xy[origin_node]
        destination = engine.node_xy[destination_node]
        od_rows.append(
            {
                "od_id": f"od_{od_index:03d}",
                "origin_node": origin_node,
                "origin_x": origin[0],
                "origin_y": origin[1],
                "destination_node": destination_node,
                "destination_x": destination[0],
                "destination_y": destination[1],
                "euclidean_distance_m": float(
                    np.linalg.norm(origin - destination)
                ),
                "component_id": int(engine.component_all[origin_node]),
                "fixed_validation_sample": True,
            }
        )
    consistency_rows: list[dict[str, Any]] = []
    route_summary_rows: list[dict[str, Any]] = []
    route_edge_rows: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    modes = ["walk", "bike", "shared"]
    tolerance = float(validation["astar_cost_tolerance"])
    cases = []
    for od_index, (origin_node, destination_node) in enumerate(od_pairs):
        hour = 6 + (od_index % 13)
        mode = modes[od_index % len(modes)]
        for objective in objective_names:
            cases.append(
                (
                    f"case_{len(cases) + 1:03d}",
                    f"od_{od_index + 1:03d}",
                    origin_node,
                    destination_node,
                    hour,
                    mode,
                    objective,
                )
            )
    for (
        case_id,
        od_id,
        origin_node,
        destination_node,
        hour,
        mode,
        objective,
    ) in tqdm(
        cases,
        desc="Validating A* against Dijkstra",
        unit="case",
        dynamic_ncols=True,
    ):
        try:
            uncertainty_weight = 1.0 if objective == "risk_aware" else 0.0
            dijkstra, dijkstra_fallback = engine.route_nodes(
                origin_node,
                destination_node,
                hour,
                mode=mode,
                objective=objective,
                algorithm="dijkstra",
                uncertainty_weight=uncertainty_weight,
            )
            astar, astar_fallback = engine.route_nodes(
                origin_node,
                destination_node,
                hour,
                mode=mode,
                objective=objective,
                algorithm="astar",
                uncertainty_weight=uncertainty_weight,
            )
            cost_difference = (
                abs(astar.total_cost - dijkstra.total_cost)
                if astar.found and dijkstra.found
                else math.inf
            )
            dijkstra_length = (
                float(
                    engine.lengths[
                        np.asarray(dijkstra.edge_indices, dtype=np.int32)
                    ].sum()
                )
                if dijkstra.found
                else math.inf
            )
            astar_length = (
                float(
                    engine.lengths[
                        np.asarray(astar.edge_indices, dtype=np.int32)
                    ].sum()
                )
                if astar.found
                else math.inf
            )
            passed = (
                dijkstra.found
                and astar.found
                and cost_difference <= tolerance
                and dijkstra_fallback == astar_fallback
            )
            consistency_rows.append(
                {
                    "case_id": case_id,
                    "od_id": od_id,
                    "hour": hour,
                    "mode": mode,
                    "objective": objective,
                    "dijkstra_found": dijkstra.found,
                    "astar_found": astar.found,
                    "dijkstra_cost": dijkstra.total_cost,
                    "astar_cost": astar.total_cost,
                    "absolute_cost_difference": cost_difference,
                    "dijkstra_distance_m": dijkstra_length,
                    "astar_distance_m": astar_length,
                    "distance_difference_m": abs(
                        dijkstra_length - astar_length
                    ),
                    "dijkstra_expanded_nodes": dijkstra.expanded_nodes,
                    "astar_expanded_nodes": astar.expanded_nodes,
                    "dijkstra_runtime_seconds": dijkstra.elapsed_seconds,
                    "astar_runtime_seconds": astar.elapsed_seconds,
                    "dijkstra_fallback_retry": dijkstra_fallback,
                    "astar_fallback_retry": astar_fallback,
                    "same_edge_sequence": (
                        dijkstra.edge_indices == astar.edge_indices
                    ),
                    "cost_tolerance": tolerance,
                    "consistency_pass": passed,
                }
            )
            for algorithm, result, fallback in (
                ("dijkstra", dijkstra, dijkstra_fallback),
                ("astar", astar, astar_fallback),
            ):
                route_summary = engine.summarize(
                    result,
                    hour,
                    mode=mode,
                    objective=objective,
                    algorithm=algorithm,
                    fallback_retry=fallback,
                )
                segment_ids = route_summary.pop("segment_ids", [])
                route_summary_rows.append(
                    {"case_id": case_id, "od_id": od_id, **route_summary}
                )
                for sequence, segment_id in enumerate(segment_ids, 1):
                    route_edge_rows.append(
                        {
                            "case_id": case_id,
                            "od_id": od_id,
                            "algorithm": algorithm,
                            "sequence": sequence,
                            "segment_id": segment_id,
                        }
                    )
        except Exception as error:
            failed_rows.append(
                {
                    "category": "algorithm_validation",
                    "object_id": case_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
    detour_smoke = engine.route(
        tuple(map(float, engine.node_xy[od_pairs[0][0]])),
        tuple(map(float, engine.node_xy[od_pairs[0][1]])),
        14,
        mode="walk",
        objective="risk_aware",
        algorithm="astar",
        max_detour_ratio=0.20,
        uncertainty_weight=1.0,
    )
    detour_segment_ids = detour_smoke.pop("segment_ids", [])
    source_hashes_after = {
        key: sha256_path(path) for key, path in source_paths.items()
    }
    atomic_csv(outputs["od_samples"], od_rows)
    atomic_csv(outputs["consistency"], consistency_rows)
    atomic_csv(outputs["route_summary"], route_summary_rows)
    atomic_csv(outputs["route_edges"], route_edge_rows)
    if failed_rows:
        atomic_csv(outputs["failed"], failed_rows)
    else:
        atomic_text(
            outputs["failed"],
            "category,object_id,error_type,error_message\n",
        )
    passed_count = sum(
        bool(row["consistency_pass"]) for row in consistency_rows
    )
    objective_counts = Counter(row["objective"] for row in consistency_rows)
    astar_expansion_reduction = [
        1.0
        - row["astar_expanded_nodes"] / row["dijkstra_expanded_nodes"]
        for row in consistency_rows
        if row["dijkstra_expanded_nodes"] > 0
    ]
    maximum_cost_difference = max(
        (
            float(row["absolute_cost_difference"])
            for row in consistency_rows
        ),
        default=math.inf,
    )
    qc = {
        "success": (
            len(failed_rows) == 0
            and len(consistency_rows) == case_count
            and passed_count == case_count
            and source_hashes_before == source_hashes_after
            and metadata["node_count"] == len(engine.node_xy)
            and metadata["edge_count"] == len(engine.segment_ids)
            and metadata["turn_restrictions"]["count"]
            == int(rules["turn_restrictions"]["expected_count"])
            and len(engine.forbidden_turns)
            == int(rules["turn_restrictions"]["expected_count"])
            and detour_smoke.get("found", False)
            and detour_smoke.get("detour_limit_satisfied", False)
        ),
        "validation_case_count": len(consistency_rows),
        "consistency_pass_count": passed_count,
        "consistency_fail_count": len(consistency_rows) - passed_count,
        "maximum_absolute_cost_difference": maximum_cost_difference,
        "cost_tolerance": tolerance,
        "objective_case_counts": dict(sorted(objective_counts.items())),
        "dijkstra_found_count": sum(
            bool(row["dijkstra_found"]) for row in consistency_rows
        ),
        "astar_found_count": sum(
            bool(row["astar_found"]) for row in consistency_rows
        ),
        "fallback_retry_case_count": sum(
            bool(row["dijkstra_fallback_retry"])
            for row in consistency_rows
        ),
        "same_edge_sequence_count": sum(
            bool(row["same_edge_sequence"]) for row in consistency_rows
        ),
        "mean_astar_expansion_reduction": float(
            np.mean(astar_expansion_reduction)
        ),
        "detour_smoke": {
            key: value
            for key, value in detour_smoke.items()
            if key not in {"segment_ids"}
        },
        "detour_smoke_edge_count": len(detour_segment_ids),
        "source_hashes_unchanged": source_hashes_before
        == source_hashes_after,
        "failed_record_count": len(failed_rows),
        "route_search_qc_performed": True,
        "turn_restriction_method": metadata["turn_restrictions"][
            "method"
        ],
        "turn_restriction_count": metadata["turn_restrictions"]["count"],
        "turn_restriction_source_sha256": metadata[
            "turn_restrictions"
        ]["source_sha256"],
        "step28_performed": False,
    }
    atomic_json(outputs["qc"], qc)
    ended_at = now_text()
    summary = {
        "phase": "G6",
        "step": 27,
        "mode": "run",
        "script": "Step27_route_search.py",
        "script_version": "2.1.0",
        "rule_version": rules["rule_version"],
        "rule_status": rules["status"],
        "approved_by": rules["approval"]["approved_by"],
        "approved_at": rules["approval"]["approved_at"],
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": qc["success"],
        "ready_for_step28_check": qc["success"],
        "formal_graph_node_count": metadata["node_count"],
        "formal_graph_edge_count": metadata["edge_count"],
        "formal_hour_count": len(metadata["hours"]),
        "formal_turn_restriction_count": metadata[
            "turn_restrictions"
        ]["count"],
        "validation_od_count": len(od_rows),
        "validation_case_count": len(consistency_rows),
        "consistency_pass_count": passed_count,
        "maximum_absolute_cost_difference": maximum_cost_difference,
        "fallback_retry_case_count": qc["fallback_retry_case_count"],
        "astar_default_approved_by_qc": qc["success"],
        "failed_record_count": len(failed_rows),
        "source_step26_modified": False,
        "step28_performed": False,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    atomic_json(outputs["summary"], summary)
    lines = [
        "# Step27 Formal Route Engine Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `SUCCESS = {str(qc['success']).upper()}`",
        f"- `READY_FOR_STEP28_CHECK = {str(qc['success']).upper()}`",
        f"- Formal graph nodes: {metadata['node_count']:,}",
        f"- Formal graph edges: {metadata['edge_count']:,}",
        f"- Protected forbidden turns: {metadata['turn_restrictions']['count']:,}",
        f"- Validation OD pairs: {len(od_rows):,}",
        f"- A*/Dijkstra objective cases: {len(consistency_rows):,}",
        f"- Consistency passes: {passed_count:,}",
        f"- Maximum absolute cost difference: {maximum_cost_difference:.12g}",
        f"- Fallback retry cases: {qc['fallback_retry_case_count']:,}",
        f"- Mean A* expansion reduction: {qc['mean_astar_expansion_reduction']:.2%}",
        f"- Failed records: {len(failed_rows):,}",
        f"- Source Step26 hashes unchanged: {qc['source_hashes_unchanged']}",
        "",
        "The reusable interface supports walk, bike and shared modes; "
        "shortest, shade, UTCI and risk-aware objectives; and A* or "
        "Dijkstra. Formal validation used 25 fixed OD pairs and all four "
        "objectives, producing 100 paired algorithm cases. Fallback-only "
        "segments are excluded on the first thermal pass and used only in "
        "a reported high-penalty retry. No Step28 evaluation or mapping was "
        "performed. The approved node-specific grade-separation turn table "
        "is enforced by a turn-aware state search; no road segment or "
        "Step26 thermal cost was deleted or modified.",
    ]
    atomic_text(outputs["report"], "\n".join(lines) + "\n")
    logger.info(
        "Formal Step27 complete: success=%s cases=%s pass=%s max_diff=%s",
        qc["success"],
        len(consistency_rows),
        passed_count,
        maximum_cost_difference,
    )
    return summary
