"""Endpoint-only graph, dead-end, snap and grade-separation audits."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

import networkx as nx
import numpy as np
from tqdm import tqdm

from routing.src.mode_classification import mode_is_candidate
from routing.src.road_geometry_qc import flag_value
from routing.src.road_schema import RoadFeature


def node_key(point: tuple[float, float]) -> tuple[float, float]:
    return round(point[0], 3), round(point[1], 3)


def direction_difference(first: float, second: float) -> float:
    value = abs(first - second) % 180.0
    return float(min(value, 180.0 - value))


def build_graph(roads: Iterable[RoadFeature]) -> nx.MultiGraph:
    graph = nx.MultiGraph()
    for road in roads:
        graph.add_edge(
            node_key(road.start),
            node_key(road.end),
            key=road.edge_id,
            edge_id=road.edge_id,
            length_m=road.length_m,
        )
    return graph


def graph_metrics(name: str, roads: list[RoadFeature], small_edges: int) -> dict[str, Any]:
    graph = build_graph(roads)
    if graph.number_of_nodes() == 0:
        return {
            "network_scope": name,
            "node_count": 0,
            "edge_count": 0,
            "connected_component_count": 0,
            "largest_component_node_ratio": 0,
            "largest_component_edge_ratio": 0,
            "degree_1_nodes": 0,
            "degree_2_nodes": 0,
            "degree_ge_3_nodes": 0,
            "isolated_edges": 0,
            "small_component_count": 0,
        }
    components = list(nx.connected_components(graph))
    component_edges = [graph.subgraph(nodes).number_of_edges() for nodes in components]
    largest_index = int(np.argmax([len(nodes) for nodes in components]))
    degrees = dict(graph.degree())
    return {
        "network_scope": name,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "connected_component_count": len(components),
        "largest_component_node_ratio": len(components[largest_index])
        / graph.number_of_nodes(),
        "largest_component_edge_ratio": component_edges[largest_index]
        / max(graph.number_of_edges(), 1),
        "degree_1_nodes": sum(value == 1 for value in degrees.values()),
        "degree_2_nodes": sum(value == 2 for value in degrees.values()),
        "degree_ge_3_nodes": sum(value >= 3 for value in degrees.values()),
        "isolated_edges": sum(
            len(nodes) == 2 and edges == 1
            for nodes, edges in zip(components, component_edges, strict=True)
        ),
        "small_component_count": sum(edges <= small_edges for edges in component_edges),
    }


def audit_network_scopes(
    roads: list[RoadFeature], small_component_edges: int
) -> list[dict[str, Any]]:
    scopes = [
        ("all_nanjing", roads),
        ("center", [road for road in roads if road.inside_center]),
        ("center_plus_500m", [road for road in roads if road.inside_center_500m]),
        ("center_plus_1000m", [road for road in roads if road.inside_center_1000m]),
        ("candidate_walk", [road for road in roads if mode_is_candidate(road, "walk")]),
        ("candidate_bike", [road for road in roads if mode_is_candidate(road, "bike")]),
        (
            "candidate_shared",
            [road for road in roads if mode_is_candidate(road, "shared")],
        ),
    ]
    return [
        graph_metrics(name, members, small_component_edges)
        for name, members in tqdm(
            scopes,
            desc="Auditing endpoint-only graph scopes",
            unit="network",
            dynamic_ncols=True,
        )
    ]


def _grid_key(x: float, y: float, size: float) -> tuple[int, int]:
    return math.floor(x / size), math.floor(y / size)


def _nearby_cells(x: float, y: float, radius: float, size: float):
    minimum = _grid_key(x - radius, y - radius, size)
    maximum = _grid_key(x + radius, y + radius, size)
    for gx in range(minimum[0], maximum[0] + 1):
        for gy in range(minimum[1], maximum[1] + 1):
            yield gx, gy


def snap_candidates(
    roads: list[RoadFeature], max_tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    graph = build_graph(roads)
    edge_lookup = {road.edge_id: road for road in roads}
    node_edges: dict[tuple[float, float], list[str]] = defaultdict(list)
    for road in roads:
        node_edges[node_key(road.start)].append(road.edge_id)
        node_edges[node_key(road.end)].append(road.edge_id)
    dead_nodes = [node for node, degree in graph.degree() if degree == 1]
    grid: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    for node in dead_nodes:
        grid[_grid_key(node[0], node[1], max_tolerance)].append(node)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for node in tqdm(
        dead_nodes,
        desc="Simulating endpoint snap candidates",
        unit="endpoint",
        dynamic_ncols=True,
    ):
        for cell in _nearby_cells(
            node[0], node[1], max_tolerance, max_tolerance
        ):
            for other in grid.get(cell, []):
                key = tuple(sorted((node, other)))
                if node == other or key in seen:
                    continue
                seen.add(key)
                distance = math.dist(node, other)
                if distance > max_tolerance:
                    continue
                first = edge_lookup[node_edges[node][0]]
                second = edge_lookup[node_edges[other][0]]
                bridge_conflict = flag_value(first.attrs.get("bridge")) != flag_value(
                    second.attrs.get("bridge")
                )
                tunnel_conflict = flag_value(first.attrs.get("tunnel")) != flag_value(
                    second.attrs.get("tunnel")
                )
                class_conflict = str(first.attrs.get("fclass")) != str(
                    second.attrs.get("fclass")
                )
                angle = direction_difference(
                    first.bearing_deg, second.bearing_deg
                )
                pairs.append(
                    {
                        "pair_id": f"SP{len(pairs) + 1:07d}",
                        "edge_id_a": first.edge_id,
                        "edge_id_b": second.edge_id,
                        "x_a": node[0],
                        "y_a": node[1],
                        "x_b": other[0],
                        "y_b": other[1],
                        "distance_m": distance,
                        "same_fclass": not class_conflict,
                        "bridge_conflict": bridge_conflict,
                        "tunnel_conflict": tunnel_conflict,
                        "parallel_candidate": angle < 15,
                        "direction_difference_deg": angle,
                        "manual_review": True,
                    }
                )
    simulation: list[dict[str, Any]] = []
    base_components = nx.number_connected_components(graph)
    base_largest = max(len(nodes) for nodes in nx.connected_components(graph))
    base_degree_one = sum(degree == 1 for _, degree in graph.degree())
    for tolerance in tqdm(
        (1.0, 2.0, 3.0),
        desc="Summarizing snap tolerances",
        unit="tolerance",
        dynamic_ncols=True,
    ):
        candidates = [row for row in pairs if row["distance_m"] <= tolerance]
        simulated = graph.copy()
        for row in candidates:
            simulated.add_edge(
                node_key((row["x_a"], row["y_a"])),
                node_key((row["x_b"], row["y_b"])),
                simulated_snap=True,
            )
        components = list(nx.connected_components(simulated))
        simulation.append(
            {
                "tolerance_m": tolerance,
                "candidate_endpoint_pairs": len(candidates),
                "same_class_candidate_connections": sum(
                    row["same_fclass"] for row in candidates
                ),
                "different_class_connections": sum(
                    not row["same_fclass"] for row in candidates
                ),
                "bridge_tunnel_conflicts": sum(
                    row["bridge_conflict"] or row["tunnel_conflict"]
                    for row in candidates
                ),
                "elevated_ground_conflicts": sum(
                    row["bridge_conflict"] for row in candidates
                ),
                "parallel_road_candidates": sum(
                    row["parallel_candidate"] for row in candidates
                ),
                "base_components": base_components,
                "simulated_components": len(components),
                "component_change": len(components) - base_components,
                "base_largest_component_nodes": base_largest,
                "simulated_largest_component_nodes": max(map(len, components)),
                "base_degree_1": base_degree_one,
                "simulated_degree_1": sum(
                    degree == 1 for _, degree in simulated.degree()
                ),
                "formal_snap_performed": False,
            }
        )
    return pairs, simulation


def _road_grid(
    roads: list[RoadFeature], grid_size: float
) -> dict[tuple[int, int], list[int]]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, road in enumerate(roads):
        min_x, max_x, min_y, max_y = road.envelope
        for cell in _nearby_cells(
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            max(max_x - min_x, max_y - min_y) / 2,
            grid_size,
        ):
            grid[cell].append(index)
    return grid


def dead_end_inventory(
    roads: list[RoadFeature],
    center_geometry: Any,
    search_radius: float,
    grid_size: float,
) -> list[dict[str, Any]]:
    from osgeo import ogr

    graph = build_graph(roads)
    component_lookup: dict[tuple[float, float], int] = {}
    component_sizes: dict[int, int] = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        component_sizes[component_id] = len(nodes)
        for node in nodes:
            component_lookup[node] = component_id
    node_edge: dict[tuple[float, float], RoadFeature] = {}
    endpoint_grid: dict[tuple[int, int], list[tuple[tuple[float, float], RoadFeature]]] = defaultdict(list)
    for road in roads:
        for endpoint in (node_key(road.start), node_key(road.end)):
            node_edge.setdefault(endpoint, road)
            endpoint_grid[_grid_key(endpoint[0], endpoint[1], search_radius)].append(
                (endpoint, road)
            )
    road_grid = _road_grid(roads, grid_size)
    boundary = center_geometry.Boundary()
    rows: list[dict[str, Any]] = []
    dead_nodes = [node for node, degree in graph.degree() if degree == 1]
    for node in tqdm(
        dead_nodes,
        desc="Auditing dead ends",
        unit="endpoint",
        dynamic_ncols=True,
    ):
        source = node_edge[node]
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(node[0], node[1])
        nearest_endpoint_distance = math.inf
        for cell in _nearby_cells(
            node[0], node[1], search_radius, search_radius
        ):
            for other, _ in endpoint_grid.get(cell, []):
                if other == node:
                    continue
                nearest_endpoint_distance = min(
                    nearest_endpoint_distance, math.dist(node, other)
                )
        nearest_road = None
        nearest_edge_distance = math.inf
        for cell in _nearby_cells(node[0], node[1], search_radius, grid_size):
            for index in road_grid.get(cell, []):
                candidate = roads[index]
                if candidate.edge_id == source.edge_id:
                    continue
                distance = point.Distance(candidate.geometry)
                if distance < nearest_edge_distance:
                    nearest_edge_distance = distance
                    nearest_road = candidate
        component_size = component_sizes[component_lookup[node]]
        inside = bool(point.Within(center_geometry))
        boundary_distance = float(point.Distance(boundary))
        bridge_conflict = (
            nearest_road is not None
            and flag_value(source.attrs.get("bridge"))
            != flag_value(nearest_road.attrs.get("bridge"))
        )
        tunnel_conflict = (
            nearest_road is not None
            and flag_value(source.attrs.get("tunnel"))
            != flag_value(nearest_road.attrs.get("tunnel"))
        )
        if boundary_distance <= 5:
            suspected = "boundary_created"
            action = "retain pending scope review"
        elif bridge_conflict or tunnel_conflict:
            suspected = "grade_separated_crossing"
            action = "do not connect without level evidence"
        elif nearest_endpoint_distance <= 3:
            suspected = "possible_unsnapped_endpoint"
            action = "manual snap candidate review"
        elif nearest_edge_distance <= 3:
            suspected = "possible_missing_segment"
            action = "inspect endpoint-to-edge gap"
        elif component_size <= 10:
            suspected = "isolated_component"
            action = "inspect component context"
        else:
            suspected = "valid_dead_end" if inside else "uncertain"
            action = "retain unless manual evidence supports repair"
        rows.append(
            {
                "dead_end_id": f"DE{len(rows) + 1:07d}",
                "source_edge_id": source.edge_id,
                "fclass": source.attrs.get("fclass"),
                "fclass_cn": source.attrs.get("fclass_cn"),
                "bridge": flag_value(source.attrs.get("bridge")),
                "tunnel": flag_value(source.attrs.get("tunnel")),
                "x": node[0],
                "y": node[1],
                "inside_center": inside,
                "distance_to_center_boundary_m": boundary_distance,
                "nearest_compatible_endpoint_m": (
                    nearest_endpoint_distance
                    if math.isfinite(nearest_endpoint_distance)
                    else None
                ),
                "nearest_compatible_edge_m": (
                    nearest_edge_distance
                    if math.isfinite(nearest_edge_distance)
                    else None
                ),
                "nearest_edge_fclass": (
                    nearest_road.attrs.get("fclass") if nearest_road else None
                ),
                "nearest_edge_bridge": (
                    flag_value(nearest_road.attrs.get("bridge"))
                    if nearest_road
                    else None
                ),
                "nearest_edge_tunnel": (
                    flag_value(nearest_road.attrs.get("tunnel"))
                    if nearest_road
                    else None
                ),
                "direction_difference_deg": (
                    direction_difference(
                        source.bearing_deg, nearest_road.bearing_deg
                    )
                    if nearest_road
                    else None
                ),
                "suspected_type": suspected,
                "suggested_action": action,
                "manual_review": suspected != "valid_dead_end",
            }
        )
    return rows


def grade_separation_crossings(
    roads: list[RoadFeature], grid_size: float
) -> list[dict[str, Any]]:
    from osgeo import ogr

    grid = _road_grid(roads, grid_size)
    candidate_pairs: set[tuple[int, int]] = set()
    for members in tqdm(
        grid.values(),
        desc="Indexing 2D crossing pairs",
        unit="cell",
        dynamic_ncols=True,
    ):
        for first, second in combinations(sorted(set(members)), 2):
            candidate_pairs.add((first, second))
    rows: list[dict[str, Any]] = []
    for first_index, second_index in tqdm(
        sorted(candidate_pairs),
        desc="Auditing 2D road crossings",
        unit="pair",
        dynamic_ncols=True,
    ):
        first = roads[first_index]
        second = roads[second_index]
        if not first.geometry.Intersects(second.geometry):
            continue
        intersection = first.geometry.Intersection(second.geometry)
        if intersection is None or intersection.IsEmpty():
            continue
        shared_endpoint = bool(
            {node_key(first.start), node_key(first.end)}
            & {node_key(second.start), node_key(second.end)}
        )
        first_bridge = flag_value(first.attrs.get("bridge"))
        second_bridge = flag_value(second.attrs.get("bridge"))
        first_tunnel = flag_value(first.attrs.get("tunnel"))
        second_tunnel = flag_value(second.attrs.get("tunnel"))
        if shared_endpoint:
            classification = "endpoint_connection"
        elif "T" in {first_bridge, second_bridge} and "T" in {
            first_tunnel,
            second_tunnel,
        }:
            classification = "bridge_tunnel_crossing"
        elif "T" in {first_bridge, second_bridge}:
            classification = "bridge_ground_crossing"
        elif "T" in {first_tunnel, second_tunnel}:
            classification = "tunnel_ground_crossing"
        elif first.attrs.get("layer") is not None and second.attrs.get("layer") is not None:
            classification = (
                "same_level_likely_connected"
                if float(first.attrs["layer"]) == float(second.attrs["layer"])
                else "unknown_level_crossing"
            )
        else:
            classification = "interior_crossing_requires_review"
        point = intersection.Centroid()
        rows.append(
            {
                "crossing_id": f"GX{len(rows) + 1:08d}",
                "edge_id_a": first.edge_id,
                "edge_id_b": second.edge_id,
                "fclass_a": first.attrs.get("fclass"),
                "fclass_b": second.attrs.get("fclass"),
                "bridge_a": first_bridge,
                "bridge_b": second_bridge,
                "tunnel_a": first_tunnel,
                "tunnel_b": second_tunnel,
                "layer_a": first.attrs.get("layer"),
                "layer_b": second.attrs.get("layer"),
                "classification": classification,
                "x": float(point.GetX()),
                "y": float(point.GetY()),
                "new_node_created": False,
                "manual_review": classification
                not in {"endpoint_connection", "same_level_likely_connected"},
            }
        )
    return rows
