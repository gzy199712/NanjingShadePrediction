"""Reusable Step27 graph build and deterministic thermal route engine."""

from __future__ import annotations

import heapq
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


HOURS = tuple(range(6, 19))
COST_FIELDS = (
    "shade_mean",
    "shade_std",
    "tmrt_mean",
    "tmrt_std",
    "utci_mean",
    "utci_std",
    "uncertainty_cost",
)


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.uint8)

    def find(self, value: int) -> int:
        current = value
        while self.parent[current] != current:
            self.parent[current] = self.parent[self.parent[current]]
            current = int(self.parent[current])
        return current

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1

    def labels(self) -> np.ndarray:
        roots = np.array(
            [self.find(index) for index in range(len(self.parent))],
            dtype=np.int32,
        )
        _, labels = np.unique(roots, return_inverse=True)
        return labels.astype(np.int32)


def _endpoint_key(point: tuple[float, ...]) -> tuple[float, float]:
    return round(float(point[0]), 3), round(float(point[1]), 3)


def _segment_endpoints(geometry: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    line = geometry.GetGeometryRef(0) if geometry.GetGeometryCount() else geometry
    if line.GetPointCount() < 2:
        raise ValueError("Segment geometry has fewer than two points")
    return _endpoint_key(line.GetPoint(0)), _endpoint_key(
        line.GetPoint(line.GetPointCount() - 1)
    )


def build_graph_artifacts(
    *,
    coverage_path: Path,
    gdb_path: Path,
    costs_path: Path,
    graph_output: Path,
    cost_output: Path,
    metadata_output: Path,
    rules: dict[str, Any],
    turn_restrictions_path: Path | None = None,
) -> dict[str, Any]:
    gdal_proxy_dir = graph_output.parent / ".gdal_pam_proxy"
    gdal_proxy_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(gdal_proxy_dir))
    from osgeo import ogr

    ogr.UseExceptions()
    coverage = pd.read_csv(
        coverage_path,
        dtype={"segment_id": str, "original_edge_id": str},
        low_memory=False,
    )
    if coverage["segment_id"].duplicated().any():
        raise ValueError("Duplicate segment_id in formal coverage")
    segment_ids = coverage["segment_id"].to_numpy(str)
    original_edge_ids = coverage["original_edge_id"].to_numpy(str)
    segment_lookup = {
        value: index for index, value in enumerate(segment_ids)
    }
    coverage_names = sorted(coverage["coverage_type"].unique())
    coverage_codes = {value: index for index, value in enumerate(coverage_names)}
    coverage_code = (
        coverage["coverage_type"].map(coverage_codes).to_numpy(np.uint8)
    )
    routing_policy = coverage["routing_policy"].astype(str).to_numpy()
    fallback_only = routing_policy == "fallback_only"
    lengths = coverage["length_m"].to_numpy(np.float64)
    node_lookup: dict[tuple[float, float], int] = {}
    node_coordinates: list[tuple[float, float]] = []
    edge_u = np.full(len(coverage), -1, dtype=np.int32)
    edge_v = np.full(len(coverage), -1, dtype=np.int32)
    dataset = ogr.Open(str(gdb_path), 0)
    layer = dataset.GetLayerByName("ThermalCostSegments")
    if layer is None:
        raise KeyError("ThermalCostSegments")
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Building formal Step27 graph",
        unit="segment",
        dynamic_ncols=True,
    ):
        segment_id = str(feature.GetField("segment_id"))
        edge_index = segment_lookup[segment_id]
        first, second = _segment_endpoints(feature.GetGeometryRef())
        for point in (first, second):
            if point not in node_lookup:
                node_lookup[point] = len(node_coordinates)
                node_coordinates.append(point)
        edge_u[edge_index] = node_lookup[first]
        edge_v[edge_index] = node_lookup[second]
    dataset = None
    if (edge_u < 0).any() or (edge_v < 0).any():
        raise ValueError("Some formal segments lack graph endpoints")
    node_xy = np.asarray(node_coordinates, dtype=np.float64)
    degree = np.bincount(
        np.concatenate((edge_u, edge_v)), minlength=len(node_xy)
    )
    offsets = np.zeros(len(node_xy) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(degree)
    adjacency_nodes = np.empty(len(segment_ids) * 2, dtype=np.int32)
    adjacency_edges = np.empty(len(segment_ids) * 2, dtype=np.int32)
    cursor = offsets[:-1].copy()
    for edge_index in tqdm(
        range(len(segment_ids)),
        desc="Indexing formal Step27 adjacency",
        unit="segment",
        dynamic_ncols=True,
    ):
        first = edge_u[edge_index]
        second = edge_v[edge_index]
        position = cursor[first]
        adjacency_nodes[position] = second
        adjacency_edges[position] = edge_index
        cursor[first] += 1
        position = cursor[second]
        adjacency_nodes[position] = first
        adjacency_edges[position] = edge_index
        cursor[second] += 1
    all_union = UnionFind(len(node_xy))
    primary_union = UnionFind(len(node_xy))
    primary_used = np.zeros(len(node_xy), dtype=bool)
    for edge_index in tqdm(
        range(len(segment_ids)),
        desc="Computing Step27 graph components",
        unit="segment",
        dynamic_ncols=True,
    ):
        first = int(edge_u[edge_index])
        second = int(edge_v[edge_index])
        all_union.union(first, second)
        if not fallback_only[edge_index]:
            primary_union.union(first, second)
            primary_used[first] = True
            primary_used[second] = True
    component_all = all_union.labels()
    component_primary = primary_union.labels()
    component_primary[~primary_used] = -1
    forbidden_turn_nodes = np.empty(0, dtype=np.int32)
    forbidden_turn_edge_a = np.empty(0, dtype=np.int32)
    forbidden_turn_edge_b = np.empty(0, dtype=np.int32)
    restriction_sha256 = None
    if turn_restrictions_path is not None:
        restrictions = pd.read_csv(
            turn_restrictions_path,
            dtype={
                "from_segment_id": str,
                "to_segment_id": str,
            },
            low_memory=False,
        )
        required = {"node_id", "from_segment_id", "to_segment_id"}
        missing = required - set(restrictions.columns)
        if missing:
            raise ValueError(
                "Turn restriction file lacks columns: "
                + ", ".join(sorted(missing))
            )
        normalized: set[tuple[int, int, int]] = set()
        for row in tqdm(
            restrictions.itertuples(index=False),
            total=len(restrictions),
            desc="Validating approved turn restrictions",
            unit="turn",
            dynamic_ncols=True,
        ):
            node = int(row.node_id)
            if node < 0 or node >= len(node_xy):
                raise ValueError(f"Invalid restriction node_id: {node}")
            try:
                first = segment_lookup[str(row.from_segment_id)]
                second = segment_lookup[str(row.to_segment_id)]
            except KeyError as error:
                raise ValueError(
                    f"Restriction references unknown segment: {error}"
                ) from error
            if first == second:
                raise ValueError("Turn restriction cannot repeat one segment")
            for edge_index in (first, second):
                if node not in {
                    int(edge_u[edge_index]),
                    int(edge_v[edge_index]),
                }:
                    raise ValueError(
                        "Restricted segment is not incident to node "
                        f"{node}: {segment_ids[edge_index]}"
                    )
            normalized.add((node, min(first, second), max(first, second)))
        ordered_restrictions = sorted(normalized)
        if len(ordered_restrictions) != len(restrictions):
            raise ValueError(
                "Approved turn restriction file contains duplicate rows"
            )
        forbidden_turn_nodes = np.asarray(
            [value[0] for value in ordered_restrictions], dtype=np.int32
        )
        forbidden_turn_edge_a = np.asarray(
            [value[1] for value in ordered_restrictions], dtype=np.int32
        )
        forbidden_turn_edge_b = np.asarray(
            [value[2] for value in ordered_restrictions], dtype=np.int32
        )
        restriction_sha256 = hashlib.sha256(
            turn_restrictions_path.read_bytes()
        ).hexdigest()
    matrices = {
        field: np.empty((len(segment_ids), len(HOURS)), dtype=np.float32)
        for field in COST_FIELDS
    }
    expected_row_count = len(segment_ids) * len(HOURS)
    row_offset = 0
    use_columns = ["segment_id", "hour", *COST_FIELDS]
    for chunk in tqdm(
        pd.read_csv(
            costs_path,
            usecols=use_columns,
            dtype={"segment_id": str},
            chunksize=200_000,
        ),
        desc="Building formal Step27 hourly arrays",
        unit="chunk",
        dynamic_ncols=True,
    ):
        start = row_offset
        end = start + len(chunk)
        positions = np.arange(start, end, dtype=np.int64)
        expected_segment_ids = segment_ids[positions // len(HOURS)]
        if not np.array_equal(chunk["segment_id"].to_numpy(str), expected_segment_ids):
            raise ValueError("Cost rows are not aligned with formal segment order")
        expected_hours = np.asarray(HOURS, dtype=int)[positions % len(HOURS)]
        if not np.array_equal(chunk["hour"].to_numpy(int), expected_hours):
            raise ValueError("Cost rows are not in expected 6-18 order")
        for field in COST_FIELDS:
            matrices[field].reshape(-1)[start:end] = chunk[field].to_numpy(
                np.float32
            )
        row_offset = end
    if row_offset != expected_row_count:
        raise ValueError(
            f"Formal hourly row count {row_offset} != {expected_row_count}"
        )
    utci_normalized = np.empty_like(matrices["utci_mean"])
    uncertainty_normalized = np.empty_like(matrices["uncertainty_cost"])
    normalization_rows: list[dict[str, Any]] = []
    for hour_index, hour in enumerate(HOURS):
        for source, target, name in (
            (
                matrices["utci_mean"],
                utci_normalized,
                "utci_mean",
            ),
            (
                matrices["uncertainty_cost"],
                uncertainty_normalized,
                "uncertainty_cost",
            ),
        ):
            minimum = float(source[:, hour_index].min())
            maximum = float(source[:, hour_index].max())
            value_range = maximum - minimum
            target[:, hour_index] = (
                (source[:, hour_index] - minimum) / value_range
                if value_range > 0
                else 0.0
            )
            normalization_rows.append(
                {
                    "hour": hour,
                    "field": name,
                    "minimum": minimum,
                    "maximum": maximum,
                    "range": value_range,
                }
            )
    graph_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_graph = graph_output.with_suffix(".npz.tmp")
    with temporary_graph.open("wb") as handle:
        np.savez_compressed(
            handle,
            node_xy=node_xy,
            offsets=offsets,
            adjacency_nodes=adjacency_nodes,
            adjacency_edges=adjacency_edges,
            edge_u=edge_u,
            edge_v=edge_v,
            segment_ids=segment_ids,
            original_edge_ids=original_edge_ids,
            lengths=lengths.astype(np.float32),
            coverage_code=coverage_code,
            fallback_only=fallback_only,
            component_all=component_all,
            component_primary=component_primary,
            forbidden_turn_nodes=forbidden_turn_nodes,
            forbidden_turn_edge_a=forbidden_turn_edge_a,
            forbidden_turn_edge_b=forbidden_turn_edge_b,
        )
    os.replace(temporary_graph, graph_output)
    temporary_cost = cost_output.with_suffix(".npz.tmp")
    with temporary_cost.open("wb") as handle:
        np.savez_compressed(
            handle,
            **matrices,
            utci_normalized=utci_normalized,
            uncertainty_normalized=uncertainty_normalized,
        )
    os.replace(temporary_cost, cost_output)
    metadata = {
        "format_version": 2,
        "crs": "EPSG:32650",
        "node_count": len(node_xy),
        "edge_count": len(segment_ids),
        "hours": list(HOURS),
        "coverage_codes": coverage_codes,
        "normalization": normalization_rows,
        "mode_networks": rules["mode_networks"],
        "objective_weights": rules["objective_weights"],
        "coverage_policy": rules["coverage_policy"],
        "origin_destination_snap": rules["origin_destination_snap"],
        "algorithm_validation": rules["algorithm_validation"],
        "duration": rules["duration"],
        "turn_restrictions": {
            "method": "node_specific_unordered_edge_pair",
            "count": len(forbidden_turn_nodes),
            "source": (
                str(turn_restrictions_path)
                if turn_restrictions_path is not None
                else None
            ),
            "source_sha256": restriction_sha256,
        },
    }
    temporary_metadata = metadata_output.with_suffix(".json.tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_output)
    return metadata


@dataclass
class SearchResult:
    found: bool
    node_indices: list[int]
    edge_indices: list[int]
    total_cost: float
    expanded_nodes: int
    elapsed_seconds: float


class RouteEngine:
    def __init__(
        self,
        graph_path: Path,
        cost_path: Path,
        metadata_path: Path,
    ) -> None:
        graph = np.load(graph_path, allow_pickle=False)
        costs = np.load(cost_path, allow_pickle=False)
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.node_xy = graph["node_xy"]
        self.offsets = graph["offsets"]
        self.adjacency_nodes = graph["adjacency_nodes"]
        self.adjacency_edges = graph["adjacency_edges"]
        self.edge_u = graph["edge_u"]
        self.edge_v = graph["edge_v"]
        self.segment_ids = graph["segment_ids"].astype(str)
        self.original_edge_ids = graph["original_edge_ids"].astype(str)
        self.lengths = graph["lengths"].astype(np.float64)
        self.coverage_code = graph["coverage_code"]
        self.fallback_only = graph["fallback_only"]
        self.component_all = graph["component_all"]
        self.component_primary = graph["component_primary"]
        if "forbidden_turn_nodes" in graph.files:
            self.forbidden_turn_nodes = graph[
                "forbidden_turn_nodes"
            ].astype(np.int32)
            self.forbidden_turn_edge_a = graph[
                "forbidden_turn_edge_a"
            ].astype(np.int32)
            self.forbidden_turn_edge_b = graph[
                "forbidden_turn_edge_b"
            ].astype(np.int32)
        else:
            self.forbidden_turn_nodes = np.empty(0, dtype=np.int32)
            self.forbidden_turn_edge_a = np.empty(0, dtype=np.int32)
            self.forbidden_turn_edge_b = np.empty(0, dtype=np.int32)
        self.forbidden_turns = {
            (int(node), int(first), int(second))
            for node, first, second in zip(
                self.forbidden_turn_nodes,
                self.forbidden_turn_edge_a,
                self.forbidden_turn_edge_b,
                strict=True,
            )
        }
        self.shade_mean = costs["shade_mean"].astype(np.float64)
        self.shade_std = costs["shade_std"].astype(np.float64)
        self.tmrt_mean = costs["tmrt_mean"].astype(np.float64)
        self.tmrt_std = costs["tmrt_std"].astype(np.float64)
        self.utci_mean = costs["utci_mean"].astype(np.float64)
        self.utci_std = costs["utci_std"].astype(np.float64)
        self.uncertainty_cost = costs["uncertainty_cost"].astype(np.float64)
        self.utci_normalized = costs["utci_normalized"].astype(np.float64)
        self.uncertainty_normalized = costs[
            "uncertainty_normalized"
        ].astype(np.float64)
        self.coverage_by_code = {
            int(code): name
            for name, code in self.metadata["coverage_codes"].items()
        }
        self.node_tree = cKDTree(self.node_xy)

    def is_turn_forbidden(
        self, node: int, first_edge: int, second_edge: int
    ) -> bool:
        if first_edge == second_edge:
            return False
        return (
            int(node),
            min(int(first_edge), int(second_edge)),
            max(int(first_edge), int(second_edge)),
        ) in self.forbidden_turns

    def _arrival_node(self, state: int) -> int:
        edge_index = state // 2
        return int(
            self.edge_v[edge_index]
            if state % 2 == 0
            else self.edge_u[edge_index]
        )

    def _hour_index(self, hour: int) -> int:
        if hour not in HOURS:
            raise ValueError(f"hour must be 6-18, got {hour}")
        return hour - HOURS[0]

    def _weights(
        self,
        hour: int,
        objective: str,
        uncertainty_weight: float,
        include_fallback: bool,
    ) -> np.ndarray:
        hour_index = self._hour_index(hour)
        weights_config = self.metadata["objective_weights"]
        if objective == "shortest":
            weights = self.lengths.copy()
        elif objective == "shade":
            config = weights_config["shade"]
            weights = self.lengths * (
                float(config["alpha"])
                + float(config["beta"])
                * (1.0 - self.shade_mean[:, hour_index])
            )
        elif objective == "utci":
            config = weights_config["utci"]
            weights = self.lengths * (
                float(config["alpha"])
                + float(config["beta"])
                * self.utci_normalized[:, hour_index]
            )
        elif objective == "risk_aware":
            config = weights_config["risk_aware"]
            gamma = (
                uncertainty_weight
                if uncertainty_weight > 0
                else float(config["gamma_default"])
            )
            weights = self.lengths * (
                float(config["alpha"])
                + float(config["beta"])
                * self.utci_normalized[:, hour_index]
                + gamma * self.uncertainty_normalized[:, hour_index]
            )
        else:
            raise ValueError(f"Unknown objective: {objective}")
        if include_fallback and objective != "shortest":
            penalties = self.metadata["coverage_policy"][
                "thermal_objectives"
            ]["penalties"]
            default = float(penalties["default_fallback_only"])
            multiplier = np.ones(len(weights), dtype=np.float64)
            for edge_index in np.flatnonzero(self.fallback_only):
                coverage = self.coverage_by_code[
                    int(self.coverage_code[edge_index])
                ]
                multiplier[edge_index] = float(
                    penalties.get(coverage, default)
                )
            weights *= multiplier
        return weights

    def _search(
        self,
        origin_node: int,
        destination_node: int,
        weights: np.ndarray,
        *,
        algorithm: str,
        include_fallback: bool,
    ) -> SearchResult:
        started = time.perf_counter()
        edge_count = len(self.edge_u)
        origin_state = edge_count * 2
        state_count = origin_state + 1
        distances = np.full(state_count, np.inf, dtype=np.float64)
        previous_state = np.full(state_count, -1, dtype=np.int32)
        distances[origin_state] = 0.0
        enabled = (
            np.ones(len(weights), dtype=bool)
            if include_fallback
            else ~self.fallback_only
        )
        minimum_unit_cost = float(
            np.min(weights[enabled] / self.lengths[enabled])
        )

        def heuristic(node: int) -> float:
            if algorithm == "dijkstra":
                return 0.0
            return float(
                np.linalg.norm(
                    self.node_xy[node] - self.node_xy[destination_node]
                )
                * minimum_unit_cost
            )

        if origin_node == destination_node:
            return SearchResult(
                True,
                [origin_node],
                [],
                0.0,
                0,
                time.perf_counter() - started,
            )
        queue: list[tuple[float, float, int]] = [
            (heuristic(origin_node), 0.0, origin_state)
        ]
        expanded = 0
        destination_state = -1
        while queue:
            _, current_distance, state = heapq.heappop(queue)
            if current_distance != distances[state]:
                continue
            node = (
                origin_node
                if state == origin_state
                else self._arrival_node(state)
            )
            incoming_edge = (
                -1 if state == origin_state else state // 2
            )
            expanded += 1
            if node == destination_node:
                destination_state = state
                break
            for position in range(
                int(self.offsets[node]), int(self.offsets[node + 1])
            ):
                edge_index = int(self.adjacency_edges[position])
                if not enabled[edge_index]:
                    continue
                if edge_index == incoming_edge:
                    continue
                if incoming_edge >= 0 and self.is_turn_forbidden(
                    node, incoming_edge, edge_index
                ):
                    continue
                neighbor = int(self.adjacency_nodes[position])
                next_state = (
                    edge_index * 2
                    if neighbor == int(self.edge_v[edge_index])
                    else edge_index * 2 + 1
                )
                candidate = current_distance + float(weights[edge_index])
                if candidate < distances[next_state]:
                    distances[next_state] = candidate
                    previous_state[next_state] = state
                    heapq.heappush(
                        queue,
                        (
                            candidate + heuristic(neighbor),
                            candidate,
                            next_state,
                        ),
                    )
        if destination_state < 0:
            return SearchResult(
                False,
                [],
                [],
                math.inf,
                expanded,
                time.perf_counter() - started,
            )
        nodes = [destination_node]
        edges: list[int] = []
        current_state = destination_state
        while current_state != origin_state:
            edge_index = current_state // 2
            parent_state = int(previous_state[current_state])
            if edge_index < 0 or parent_state < 0:
                raise RuntimeError("Broken predecessor chain")
            edges.append(edge_index)
            parent_node = (
                origin_node
                if parent_state == origin_state
                else self._arrival_node(parent_state)
            )
            nodes.append(parent_node)
            current_state = parent_state
        nodes.reverse()
        edges.reverse()
        return SearchResult(
            True,
            nodes,
            edges,
            float(distances[destination_state]),
            expanded,
            time.perf_counter() - started,
        )

    def snap_pair(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> tuple[int, int, float, float]:
        config = self.metadata["origin_destination_snap"]
        count = int(config["candidate_count"])
        maximum = float(config["maximum_distance_m"])
        origin_distances, origin_nodes = self.node_tree.query(
            [origin], k=count
        )
        destination_distances, destination_nodes = self.node_tree.query(
            [destination], k=count
        )
        candidates = []
        for origin_distance, origin_node in zip(
            np.atleast_1d(origin_distances).reshape(-1),
            np.atleast_1d(origin_nodes).reshape(-1),
            strict=True,
        ):
            for destination_distance, destination_node in zip(
                np.atleast_1d(destination_distances).reshape(-1),
                np.atleast_1d(destination_nodes).reshape(-1),
                strict=True,
            ):
                if origin_distance > maximum or destination_distance > maximum:
                    continue
                if (
                    self.component_all[int(origin_node)]
                    != self.component_all[int(destination_node)]
                ):
                    continue
                candidates.append(
                    (
                        float(origin_distance + destination_distance),
                        int(origin_node),
                        int(destination_node),
                        float(origin_distance),
                        float(destination_distance),
                    )
                )
        if not candidates:
            raise ValueError(
                "No origin/destination snap pair within 100 m and the same "
                "connected component"
            )
        _, origin_node, destination_node, origin_distance, destination_distance = min(
            candidates
        )
        return (
            origin_node,
            destination_node,
            origin_distance,
            destination_distance,
        )

    def route_nodes(
        self,
        origin_node: int,
        destination_node: int,
        hour: int,
        *,
        mode: str = "walk",
        objective: str = "shortest",
        algorithm: str = "astar",
        uncertainty_weight: float = 0.0,
    ) -> tuple[SearchResult, bool]:
        if mode not in self.metadata["mode_networks"] or mode in {
            "source_rule",
            "note",
        }:
            raise ValueError(f"Unknown mode: {mode}")
        if algorithm not in {"astar", "dijkstra"}:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        if (
            self.component_all[origin_node]
            != self.component_all[destination_node]
        ):
            return (
                SearchResult(False, [], [], math.inf, 0, 0.0),
                False,
            )
        include_fallback = objective == "shortest"
        if not include_fallback:
            same_primary = (
                self.component_primary[origin_node] >= 0
                and self.component_primary[origin_node]
                == self.component_primary[destination_node]
            )
            if same_primary:
                primary_weights = self._weights(
                    hour, objective, uncertainty_weight, False
                )
                primary = self._search(
                    origin_node,
                    destination_node,
                    primary_weights,
                    algorithm=algorithm,
                    include_fallback=False,
                )
                if primary.found:
                    return primary, False
            include_fallback = True
        weights = self._weights(
            hour, objective, uncertainty_weight, include_fallback
        )
        return (
            self._search(
                origin_node,
                destination_node,
                weights,
                algorithm=algorithm,
                include_fallback=include_fallback,
            ),
            include_fallback and objective != "shortest",
        )

    def summarize(
        self,
        result: SearchResult,
        hour: int,
        *,
        mode: str,
        objective: str,
        algorithm: str,
        fallback_retry: bool,
        origin_snap_distance_m: float = 0.0,
        destination_snap_distance_m: float = 0.0,
    ) -> dict[str, Any]:
        if not result.found:
            return {
                "found": False,
                "reason": "no_path_between_snapped_components",
                "hour": hour,
                "mode": mode,
                "objective": objective,
                "algorithm": algorithm,
            }
        edges = np.asarray(result.edge_indices, dtype=np.int32)
        lengths = self.lengths[edges]
        total_length = float(lengths.sum())
        hour_index = self._hour_index(hour)
        weighted = lambda values: float(
            np.sum(values[edges, hour_index] * lengths) / total_length
        )
        coverage = [
            self.coverage_by_code[int(self.coverage_code[index])]
            for index in edges
        ]
        fallback_length = float(
            lengths[
                np.array(
                    [value == "spatial_fallback" for value in coverage],
                    dtype=bool,
                )
            ].sum()
        )
        prior_length = float(
            lengths[
                np.array(
                    [value == "prior_imputed" for value in coverage],
                    dtype=bool,
                )
            ].sum()
        )
        speed = float(self.metadata["duration"][f"{mode}_speed_mps"])
        uncertainty_mean = weighted(self.uncertainty_cost)
        hour_uncertainty = self.uncertainty_cost[:, hour_index]
        uncertainty_percentile = float(
            100.0 * np.mean(hour_uncertainty <= uncertainty_mean)
        )
        return {
            "found": True,
            "hour": hour,
            "mode": mode,
            "objective": objective,
            "algorithm": algorithm,
            "distance_m": total_length,
            "estimated_duration_min": total_length / speed / 60,
            "total_cost": result.total_cost,
            "mean_shade": weighted(self.shade_mean),
            "shade_exposure": weighted(1.0 - self.shade_mean),
            "mean_tmrt": weighted(self.tmrt_mean),
            "max_tmrt": float(self.tmrt_mean[edges, hour_index].max()),
            "mean_utci": weighted(self.utci_mean),
            "max_utci": float(self.utci_mean[edges, hour_index].max()),
            "heat_exposure_length_m": float(
                lengths[self.utci_mean[edges, hour_index] >= 32.0].sum()
            ),
            "uncertainty_mean": uncertainty_mean,
            "uncertainty_max": float(
                self.uncertainty_cost[edges, hour_index].max()
            ),
            "uncertainty_percentile": uncertainty_percentile,
            "unknown_length_m": prior_length,
            "spatial_fallback_length_m": fallback_length,
            "prior_imputed_length_m": prior_length,
            "center_inside_length_m": total_length,
            "center_outside_length_m": 0.0,
            "edge_count": len(edges),
            "expanded_nodes": result.expanded_nodes,
            "runtime_seconds": result.elapsed_seconds,
            "fallback_retry": fallback_retry,
            "origin_snap_distance_m": origin_snap_distance_m,
            "destination_snap_distance_m": destination_snap_distance_m,
            "segment_ids": self.segment_ids[edges].tolist(),
            "node_indices": result.node_indices,
        }

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        hour: int,
        mode: str = "walk",
        objective: str = "shortest",
        algorithm: str = "astar",
        max_detour_ratio: float | None = None,
        uncertainty_weight: float = 0.0,
    ) -> dict[str, Any]:
        origin_node, destination_node, origin_snap, destination_snap = (
            self.snap_pair(origin, destination)
        )
        result, fallback_retry = self.route_nodes(
            origin_node,
            destination_node,
            hour,
            mode=mode,
            objective=objective,
            algorithm=algorithm,
            uncertainty_weight=uncertainty_weight,
        )
        summary = self.summarize(
            result,
            hour,
            mode=mode,
            objective=objective,
            algorithm=algorithm,
            fallback_retry=fallback_retry,
            origin_snap_distance_m=origin_snap,
            destination_snap_distance_m=destination_snap,
        )
        if not summary["found"]:
            return summary
        if objective == "shortest":
            summary["detour_ratio"] = 0.0
            return summary
        shortest_result, _ = self.route_nodes(
            origin_node,
            destination_node,
            hour,
            mode=mode,
            objective="shortest",
            algorithm="dijkstra",
        )
        shortest_distance = float(
            self.lengths[np.asarray(shortest_result.edge_indices, dtype=int)].sum()
        )
        summary["detour_ratio"] = (
            summary["distance_m"] / shortest_distance - 1.0
        )
        if (
            max_detour_ratio is not None
            and summary["detour_ratio"] > max_detour_ratio
        ):
            if max_detour_ratio < 0:
                raise ValueError("max_detour_ratio must be nonnegative")
            maximum_length = shortest_distance * (1.0 + max_detour_ratio)
            thermal_weights = self._weights(
                hour, objective, uncertainty_weight, True
            )
            shortest_edges = np.asarray(
                shortest_result.edge_indices, dtype=np.int32
            )
            candidates: list[tuple[float, SearchResult]] = [
                (
                    float(thermal_weights[shortest_edges].sum()),
                    SearchResult(
                        True,
                        shortest_result.node_indices,
                        shortest_result.edge_indices,
                        float(thermal_weights[shortest_edges].sum()),
                        shortest_result.expanded_nodes,
                        shortest_result.elapsed_seconds,
                    ),
                )
            ]
            base_unit_cost = float(
                np.median(thermal_weights / self.lengths)
            )
            for factor in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64):
                composite = (
                    thermal_weights
                    + base_unit_cost * factor * self.lengths
                )
                candidate = self._search(
                    origin_node,
                    destination_node,
                    composite,
                    algorithm=algorithm,
                    include_fallback=True,
                )
                if not candidate.found:
                    continue
                candidate_edges = np.asarray(
                    candidate.edge_indices, dtype=np.int32
                )
                candidate_length = float(
                    self.lengths[candidate_edges].sum()
                )
                if candidate_length <= maximum_length + 1e-6:
                    thermal_cost = float(
                        thermal_weights[candidate_edges].sum()
                    )
                    candidate.total_cost = thermal_cost
                    candidates.append((thermal_cost, candidate))
            _, selected = min(candidates, key=lambda value: value[0])
            selected_fallback = bool(
                self.fallback_only[
                    np.asarray(selected.edge_indices, dtype=np.int32)
                ].any()
            )
            summary = self.summarize(
                selected,
                hour,
                mode=mode,
                objective=objective,
                algorithm=algorithm,
                fallback_retry=selected_fallback,
                origin_snap_distance_m=origin_snap,
                destination_snap_distance_m=destination_snap,
            )
            summary["detour_ratio"] = (
                summary["distance_m"] / shortest_distance - 1.0
            )
            summary["detour_limit_satisfied"] = (
                summary["distance_m"] <= maximum_length + 1e-6
            )
            summary["detour_enforcement_method"] = (
                "deterministic_lagrangian_candidate_search"
            )
        else:
            summary["detour_limit_satisfied"] = True
        return summary


_DEFAULT_ENGINE: RouteEngine | None = None


def route(
    origin: tuple[float, float],
    destination: tuple[float, float],
    hour: int,
    mode: str = "walk",
    objective: str = "shortest",
    algorithm: str = "astar",
    max_detour_ratio: float | None = None,
    uncertainty_weight: float = 0.0,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Formal unified interface requested by Phase G6."""

    global _DEFAULT_ENGINE
    root = project_root or Path(__file__).resolve().parents[2]
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = RouteEngine(
            root / "routing/data/graph/step27_route_graph.npz",
            root / "routing/data/graph/step27_hourly_costs.npz",
            root / "routing/data/graph/step27_graph_metadata.json",
        )
    return _DEFAULT_ENGINE.route(
        origin,
        destination,
        hour,
        mode=mode,
        objective=objective,
        algorithm=algorithm,
        max_detour_ratio=max_detour_ratio,
        uncertainty_weight=uncertainty_weight,
    )
