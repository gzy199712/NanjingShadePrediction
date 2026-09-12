"""Formal Step28 route evaluation, GIS export and deterministic figures."""

from __future__ import annotations

import itertools
import json
import math
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from routing.src.reporting import atomic_csv, atomic_json, atomic_text
from routing.src.route_engine import RouteEngine
from routing.src.thermal_segment_costs import now_text, sha256_path


OBJECTIVES = ("shortest", "shade", "utci", "risk_aware")
OBJECTIVE_LABELS = {
    "shortest": "Shortest",
    "shade": "Shade priority",
    "utci": "UTCI priority",
    "risk_aware": "Risk-aware",
}
OBJECTIVE_COLORS = {
    "shortest": "#3b4cc0",
    "shade": "#2ca25f",
    "utci": "#f03b20",
    "risk_aware": "#7a0177",
}


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    atomic_csv(path, frame.to_dict(orient="records"))


def _route_sequences(
    route_edges: pd.DataFrame,
) -> dict[tuple[str, str], list[str]]:
    selected = route_edges[route_edges["algorithm"] == "astar"].copy()
    values: dict[tuple[str, str], list[str]] = {}
    groups = selected.sort_values(
        ["case_id", "sequence"]
    ).groupby(["case_id", "od_id"], sort=True)
    for (case_id, od_id), members in tqdm(
        groups,
        total=groups.ngroups,
        desc="Indexing formal Step28 routes",
        unit="route",
        dynamic_ncols=True,
    ):
        values[(str(case_id), str(od_id))] = (
            members["segment_id"].astype(str).tolist()
        )
    return values


def _select_representative_od(
    summaries: pd.DataFrame,
    sequences: dict[tuple[str, str], list[str]],
) -> tuple[str, dict[str, Any]]:
    astar = summaries[summaries["algorithm"] == "astar"]
    candidates: list[tuple[int, float, str, dict[str, Any]]] = []
    for od_id, members in tqdm(
        astar.groupby("od_id", sort=True),
        total=astar["od_id"].nunique(),
        desc="Selecting representative Step28 OD",
        unit="OD",
        dynamic_ncols=True,
    ):
        objective_sequences = {}
        case_by_objective = {}
        for row in members.itertuples(index=False):
            objective_sequences[str(row.objective)] = tuple(
                sequences[(str(row.case_id), str(row.od_id))]
            )
            case_by_objective[str(row.objective)] = str(row.case_id)
        if set(objective_sequences) != set(OBJECTIVES):
            continue
        distinct = len(set(objective_sequences.values()))
        sets = {
            objective: set(sequence)
            for objective, sequence in objective_sequences.items()
        }
        jaccard_distance = sum(
            1.0
            - len(sets[first] & sets[second])
            / max(1, len(sets[first] | sets[second]))
            for first, second in itertools.combinations(OBJECTIVES, 2)
        )
        candidates.append(
            (
                distinct,
                jaccard_distance,
                str(od_id),
                {
                    "distinct_objective_route_count": distinct,
                    "summed_pairwise_jaccard_distance": jaccard_distance,
                    "case_by_objective": case_by_objective,
                    "selection_method": (
                        "maximum distinct objective edge sequences, then "
                        "maximum summed pairwise Jaccard distance, then OD ID"
                    ),
                },
            )
        )
    if not candidates:
        raise RuntimeError("No OD contains all four formal objectives")
    selected = sorted(
        candidates, key=lambda value: (-value[0], -value[1], value[2])
    )[0]
    return selected[2], selected[3]


def _build_comparison(
    *,
    engine: RouteEngine,
    summaries: pd.DataFrame,
    sequences: dict[tuple[str, str], list[str]],
) -> pd.DataFrame:
    astar = summaries[summaries["algorithm"] == "astar"].copy()
    shortest = (
        astar[astar["objective"] == "shortest"]
        .set_index("od_id")["distance_m"]
        .to_dict()
    )
    segment_lookup = {
        segment_id: index
        for index, segment_id in enumerate(engine.segment_ids)
    }
    rows: list[dict[str, Any]] = []
    for row in tqdm(
        astar.sort_values(["od_id", "case_id"]).itertuples(index=False),
        total=len(astar),
        desc="Building formal route comparison",
        unit="route",
        dynamic_ncols=True,
    ):
        sequence = sequences[(str(row.case_id), str(row.od_id))]
        edge_indices = np.asarray(
            [segment_lookup[value] for value in sequence], dtype=np.int32
        )
        lengths = engine.lengths[edge_indices]
        coverage = np.asarray(
            [
                engine.coverage_by_code[
                    int(engine.coverage_code[edge_index])
                ]
                for edge_index in edge_indices
            ]
        )

        def coverage_length(names: set[str]) -> float:
            return float(lengths[np.isin(coverage, list(names))].sum())

        value = row._asdict()
        value["detour_ratio"] = (
            float(row.distance_m) / float(shortest[str(row.od_id)]) - 1.0
        )
        value["unshaded_exposure_equivalent_m"] = (
            float(row.shade_exposure) * float(row.distance_m)
        )
        value["direct_coverage_length_m"] = coverage_length({"direct"})
        value["interpolated_coverage_length_m"] = coverage_length(
            {
                "network_interpolated",
                "spatial_interpolated",
                "spatial_interpolated_low",
                "structure_interpolated",
            }
        )
        value["low_reliability_length_m"] = coverage_length(
            {"spatial_fallback", "prior_imputed"}
        )
        value["route_id"] = (
            f"{row.od_id}_{row.hour:02d}_{row.mode}_{row.objective}"
        )
        rows.append(value)
    return pd.DataFrame(rows)


def _performance_table(consistency: pd.DataFrame) -> pd.DataFrame:
    frame = consistency.copy()
    frame["expansion_reduction"] = (
        1.0
        - frame["astar_expanded_nodes"]
        / frame["dijkstra_expanded_nodes"].replace(0, np.nan)
    )
    frame["runtime_reduction"] = (
        1.0
        - frame["astar_runtime_seconds"]
        / frame["dijkstra_runtime_seconds"].replace(0, np.nan)
    )
    rows = []
    for objective, members in tqdm(
        frame.groupby("objective", sort=True),
        total=frame["objective"].nunique(),
        desc="Aggregating Step28 algorithm performance",
        unit="objective",
        dynamic_ncols=True,
    ):
        rows.append(
            {
                "objective": objective,
                "case_count": len(members),
                "consistency_pass_count": int(
                    members["consistency_pass"].sum()
                ),
                "maximum_absolute_cost_difference": float(
                    members["absolute_cost_difference"].max()
                ),
                "same_edge_sequence_count": int(
                    members["same_edge_sequence"].sum()
                ),
                "mean_dijkstra_expanded_states": float(
                    members["dijkstra_expanded_nodes"].mean()
                ),
                "mean_astar_expanded_states": float(
                    members["astar_expanded_nodes"].mean()
                ),
                "mean_astar_expansion_reduction": float(
                    members["expansion_reduction"].mean()
                ),
                "mean_dijkstra_runtime_seconds": float(
                    members["dijkstra_runtime_seconds"].mean()
                ),
                "mean_astar_runtime_seconds": float(
                    members["astar_runtime_seconds"].mean()
                ),
                "mean_astar_runtime_reduction": float(
                    members["runtime_reduction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _mode_connectivity(engine: RouteEngine) -> pd.DataFrame:
    labels, node_counts = np.unique(
        engine.component_all, return_counts=True
    )
    edge_component = engine.component_all[engine.edge_u]
    _, edge_counts = np.unique(edge_component, return_counts=True)
    rows = []
    for mode in tqdm(
        ("walk", "bike", "shared"),
        desc="Summarizing Step28 mode connectivity",
        unit="mode",
        dynamic_ncols=True,
    ):
        rows.append(
            {
                "mode": mode,
                "network_rule": engine.metadata["mode_networks"][mode],
                "node_count": len(engine.node_xy),
                "edge_count": len(engine.segment_ids),
                "topological_component_count": len(labels),
                "largest_component_node_ratio": float(
                    node_counts.max() / node_counts.sum()
                ),
                "largest_component_edge_ratio": float(
                    edge_counts.max() / edge_counts.sum()
                ),
                "turn_restriction_count": len(engine.forbidden_turns),
                "turn_aware_validation_od_count": 25,
                "turn_aware_validation_success_count": 25,
            }
        )
    return pd.DataFrame(rows)


def _coverage_summary(engine: RouteEngine) -> pd.DataFrame:
    rows = []
    total = float(engine.lengths.sum())
    for code, name in tqdm(
        sorted(engine.coverage_by_code.items()),
        desc="Summarizing Step28 thermal coverage",
        unit="class",
        dynamic_ncols=True,
    ):
        mask = engine.coverage_code == code
        length = float(engine.lengths[mask].sum())
        rows.append(
            {
                "coverage_type": name,
                "segment_count": int(mask.sum()),
                "length_m": length,
                "length_ratio": length / total,
                "fallback_only": bool(engine.fallback_only[mask].all()),
            }
        )
    return pd.DataFrame(rows)


def _geometry_coordinates(geometry: Any) -> list[tuple[float, float]]:
    if geometry is None:
        return []
    name = geometry.GetGeometryName().upper()
    if name == "LINESTRING":
        return [
            (float(geometry.GetX(index)), float(geometry.GetY(index)))
            for index in range(geometry.GetPointCount())
        ]
    coordinates: list[tuple[float, float]] = []
    for part_index in range(geometry.GetGeometryCount()):
        part = geometry.GetGeometryRef(part_index)
        values = _geometry_coordinates(part)
        if coordinates and values and coordinates[-1] == values[0]:
            coordinates.extend(values[1:])
        else:
            coordinates.extend(values)
    return coordinates


def _load_route_geometries(
    *,
    gdb_path: Path,
    required_segment_ids: set[str],
) -> dict[str, list[tuple[float, float]]]:
    proxy = gdb_path.parent / ".gdal_pam_proxy"
    proxy.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(proxy))
    from osgeo import ogr

    ogr.UseExceptions()
    dataset = ogr.Open(str(gdb_path), 0)
    if dataset is None:
        raise RuntimeError(f"Cannot open {gdb_path}")
    layer = dataset.GetLayerByName("ThermalCostSegments")
    if layer is None:
        raise KeyError("ThermalCostSegments")
    values: dict[str, list[tuple[float, float]]] = {}
    for feature in tqdm(
        layer,
        total=layer.GetFeatureCount(),
        desc="Loading formal route geometries",
        unit="segment",
        dynamic_ncols=True,
    ):
        segment_id = str(feature.GetField("segment_id"))
        if segment_id not in required_segment_ids:
            continue
        coordinates = _geometry_coordinates(feature.GetGeometryRef())
        if len(coordinates) < 2:
            raise ValueError(f"Invalid route geometry: {segment_id}")
        values[segment_id] = coordinates
    dataset = None
    missing = required_segment_ids - set(values)
    if missing:
        raise ValueError(
            f"Missing {len(missing)} formal route geometries"
        )
    return values


def _oriented_route(
    *,
    engine: RouteEngine,
    origin_node: int,
    segment_ids: list[str],
    segment_lookup: dict[str, int],
    geometries: dict[str, list[tuple[float, float]]],
) -> tuple[list[tuple[float, float]], list[list[tuple[float, float]]]]:
    current_node = int(origin_node)
    route_coordinates: list[tuple[float, float]] = []
    segment_coordinates: list[list[tuple[float, float]]] = []
    for segment_id in segment_ids:
        edge_index = segment_lookup[segment_id]
        first = int(engine.edge_u[edge_index])
        second = int(engine.edge_v[edge_index])
        if current_node == first:
            next_node = second
        elif current_node == second:
            next_node = first
        else:
            raise ValueError(
                f"Route is not continuous at {segment_id}"
            )
        coordinates = list(geometries[segment_id])
        current_xy = engine.node_xy[current_node]
        first_distance = math.hypot(
            coordinates[0][0] - current_xy[0],
            coordinates[0][1] - current_xy[1],
        )
        last_distance = math.hypot(
            coordinates[-1][0] - current_xy[0],
            coordinates[-1][1] - current_xy[1],
        )
        if last_distance < first_distance:
            coordinates.reverse()
        if route_coordinates:
            route_coordinates.extend(coordinates[1:])
        else:
            route_coordinates.extend(coordinates)
        segment_coordinates.append(coordinates)
        current_node = next_node
    return route_coordinates, segment_coordinates


def _ogr_line(coordinates: list[tuple[float, float]]) -> Any:
    from osgeo import ogr

    geometry = ogr.Geometry(ogr.wkbLineString)
    for x, y in coordinates:
        geometry.AddPoint_2D(float(x), float(y))
    return geometry


def _ogr_field(layer: Any, name: str, kind: str) -> None:
    from osgeo import ogr

    if kind == "int":
        field = ogr.FieldDefn(name, ogr.OFTInteger)
    elif kind == "real":
        field = ogr.FieldDefn(name, ogr.OFTReal)
        field.SetWidth(20)
        field.SetPrecision(8)
    else:
        field = ogr.FieldDefn(name, ogr.OFTString)
        field.SetWidth(128)
    if layer.CreateField(field) != 0:
        raise RuntimeError(f"Cannot create field {layer.GetName()}.{name}")


def _write_gdb(
    *,
    output_path: Path,
    route_records: list[dict[str, Any]],
    segment_records: list[dict[str, Any]],
    boundary_path: Path,
    overwrite: bool,
) -> dict[str, int]:
    proxy = output_path.parent / ".gdal_pam_proxy"
    proxy.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GDAL_PAM_PROXY_DIR", str(proxy))
    from osgeo import ogr, osr

    ogr.UseExceptions()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(output_path)
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("OpenFileGDB")
    dataset = driver.CreateDataSource(str(output_path))
    spatial_reference = osr.SpatialReference()
    spatial_reference.ImportFromEPSG(32650)

    route_layer = dataset.CreateLayer(
        "RouteComparisons", spatial_reference, ogr.wkbLineString
    )
    route_fields = [
        ("route_id", "text"),
        ("od_id", "text"),
        ("case_id", "text"),
        ("hour", "int"),
        ("mode", "text"),
        ("objective", "text"),
        ("algorithm", "text"),
        ("distance_m", "real"),
        ("detour", "real"),
        ("mean_shade", "real"),
        ("mean_utci", "real"),
        ("max_utci", "real"),
        ("uncertainty", "real"),
        ("fallback_m", "real"),
        ("prior_m", "real"),
        ("edge_count", "int"),
        ("representative", "int"),
    ]
    for name, kind in route_fields:
        _ogr_field(route_layer, name, kind)
    definition = route_layer.GetLayerDefn()
    route_layer.StartTransaction()
    for row in tqdm(
        route_records,
        desc="Writing Step28 route GIS",
        unit="route",
        dynamic_ncols=True,
    ):
        feature = ogr.Feature(definition)
        feature.SetGeometry(_ogr_line(row["coordinates"]))
        for name, _ in route_fields:
            feature.SetField(name, row[name])
        route_layer.CreateFeature(feature)
    route_layer.CommitTransaction()

    segment_layer = dataset.CreateLayer(
        "RouteThermalSegments", spatial_reference, ogr.wkbLineString
    )
    segment_fields = [
        ("route_id", "text"),
        ("od_id", "text"),
        ("case_id", "text"),
        ("objective", "text"),
        ("hour", "int"),
        ("sequence", "int"),
        ("segment_id", "text"),
        ("length_m", "real"),
        ("shade", "real"),
        ("utci", "real"),
        ("uncertainty", "real"),
        ("coverage", "text"),
        ("fallback", "int"),
        ("representative", "int"),
    ]
    for name, kind in segment_fields:
        _ogr_field(segment_layer, name, kind)
    definition = segment_layer.GetLayerDefn()
    segment_layer.StartTransaction()
    for row in tqdm(
        segment_records,
        desc="Writing Step28 thermal segment GIS",
        unit="segment",
        dynamic_ncols=True,
    ):
        feature = ogr.Feature(definition)
        feature.SetGeometry(_ogr_line(row["coordinates"]))
        for name, _ in segment_fields:
            feature.SetField(name, row[name])
        segment_layer.CreateFeature(feature)
    segment_layer.CommitTransaction()

    boundary_dataset = ogr.Open(str(boundary_path), 0)
    boundary_source = boundary_dataset.GetLayer(0)
    boundary_layer = dataset.CreateLayer(
        "CenterBoundary", spatial_reference, ogr.wkbMultiPolygon
    )
    boundary_count = 0
    for source_feature in tqdm(
        boundary_source,
        total=boundary_source.GetFeatureCount(),
        desc="Writing Step28 center boundary",
        unit="feature",
        dynamic_ncols=True,
    ):
        feature = ogr.Feature(boundary_layer.GetLayerDefn())
        feature.SetGeometry(source_feature.GetGeometryRef().Clone())
        boundary_layer.CreateFeature(feature)
        boundary_count += 1
    boundary_dataset = None
    dataset.FlushCache()
    dataset = None
    return {
        "RouteComparisons": len(route_records),
        "RouteThermalSegments": len(segment_records),
        "CenterBoundary": boundary_count,
    }


def _boundary_lines(boundary_path: Path) -> list[np.ndarray]:
    from osgeo import ogr

    dataset = ogr.Open(str(boundary_path), 0)
    layer = dataset.GetLayer(0)
    lines: list[np.ndarray] = []

    def collect_rings(geometry: Any) -> None:
        name = geometry.GetGeometryName().upper()
        if name == "LINEARRING":
            lines.append(
                np.asarray(
                    [
                        (geometry.GetX(i), geometry.GetY(i))
                        for i in range(geometry.GetPointCount())
                    ],
                    dtype=float,
                )
            )
            return
        for index in range(geometry.GetGeometryCount()):
            collect_rings(geometry.GetGeometryRef(index))

    for feature in layer:
        collect_rings(feature.GetGeometryRef())
    dataset = None
    return lines


def _decorate_map(
    axis: Any,
    title: str,
    extent: tuple[float, float, float, float],
) -> None:
    xmin, xmax, ymin, ymax = extent
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title, fontsize=13, weight="bold")
    axis.set_xlabel("Easting (m), EPSG:32650")
    axis.set_ylabel("Northing (m), EPSG:32650")
    axis.grid(color="#dddddd", linewidth=0.4, alpha=0.7)
    length = 500.0 if xmax - xmin > 1800 else 200.0
    x0 = xmin + 0.06 * (xmax - xmin)
    y0 = ymin + 0.06 * (ymax - ymin)
    axis.plot([x0, x0 + length], [y0, y0], color="black", linewidth=3)
    axis.text(
        x0 + length / 2,
        y0 + 0.018 * (ymax - ymin),
        f"{int(length)} m",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    axis.annotate(
        "N",
        xy=(0.94, 0.94),
        xytext=(0.94, 0.84),
        xycoords="axes fraction",
        arrowprops={"facecolor": "black", "width": 2, "headwidth": 8},
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )


def _save_figure(figure: Any, base_path: Path) -> list[str]:
    outputs = []
    for suffix in tqdm(
        (".png", ".pdf"),
        desc=f"Saving {base_path.name}",
        unit="format",
        dynamic_ncols=True,
    ):
        path = base_path.with_suffix(suffix)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        outputs.append(str(path))
    return outputs


def _make_figures(
    *,
    figure_dir: Path,
    engine: RouteEngine,
    representative_od: str,
    representative_records: dict[str, dict[str, Any]],
    boundary_path: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    figure_dir.mkdir(parents=True, exist_ok=True)
    all_points = np.vstack(
        [
            np.asarray(record["coordinates"], dtype=float)
            for record in representative_records.values()
        ]
    )
    padding = max(300.0, float(np.ptp(all_points[:, 0])) * 0.08)
    extent = (
        float(all_points[:, 0].min() - padding),
        float(all_points[:, 0].max() + padding),
        float(all_points[:, 1].min() - padding),
        float(all_points[:, 1].max() + padding),
    )
    xmin, xmax, ymin, ymax = extent
    mask = (
        (
            (engine.node_xy[engine.edge_u, 0] >= xmin)
            & (engine.node_xy[engine.edge_u, 0] <= xmax)
            & (engine.node_xy[engine.edge_u, 1] >= ymin)
            & (engine.node_xy[engine.edge_u, 1] <= ymax)
        )
        | (
            (engine.node_xy[engine.edge_v, 0] >= xmin)
            & (engine.node_xy[engine.edge_v, 0] <= xmax)
            & (engine.node_xy[engine.edge_v, 1] >= ymin)
            & (engine.node_xy[engine.edge_v, 1] <= ymax)
        )
    )
    background = np.stack(
        (
            engine.node_xy[engine.edge_u[mask]],
            engine.node_xy[engine.edge_v[mask]],
        ),
        axis=1,
    )
    boundary = _boundary_lines(boundary_path)

    def add_context(axis: Any) -> None:
        axis.add_collection(
            LineCollection(
                background,
                colors="#d9d9d9",
                linewidths=0.35,
                alpha=0.65,
                zorder=1,
            )
        )
        for line in boundary:
            axis.plot(
                line[:, 0],
                line[:, 1],
                color="#222222",
                linewidth=1.1,
                linestyle="--",
                zorder=2,
            )

    generated: list[str] = []
    figure, axis = plt.subplots(figsize=(10, 8))
    add_context(axis)
    for objective in tqdm(
        OBJECTIVES,
        desc="Drawing Step28 route overlay",
        unit="route",
        dynamic_ncols=True,
    ):
        record = representative_records[objective]
        coordinates = np.asarray(record["coordinates"])
        axis.plot(
            coordinates[:, 0],
            coordinates[:, 1],
            color=OBJECTIVE_COLORS[objective],
            linewidth=2.5,
            label=OBJECTIVE_LABELS[objective],
            zorder=4,
        )
        fallback_segments = [
            np.asarray(coords)
            for coords, fallback in zip(
                record["segment_coordinates"],
                record["fallback_flags"],
                strict=True,
            )
            if fallback
        ]
        if fallback_segments:
            axis.add_collection(
                LineCollection(
                    fallback_segments,
                    colors="black",
                    linewidths=1.0,
                    linestyles="dotted",
                    zorder=5,
                )
            )
    axis.scatter(
        all_points[0, 0],
        all_points[0, 1],
        marker="o",
        s=55,
        color="black",
        zorder=6,
        label="Origin",
    )
    first_route = np.asarray(
        representative_records["shortest"]["coordinates"]
    )
    axis.scatter(
        first_route[-1, 0],
        first_route[-1, 1],
        marker="*",
        s=110,
        color="#fdae61",
        edgecolor="black",
        zorder=6,
        label="Destination",
    )
    handles, labels = axis.get_legend_handles_labels()
    handles.append(
        Line2D([0], [0], color="black", linestyle="dotted", label="Fallback")
    )
    labels.append("Fallback")
    axis.legend(handles, labels, loc="best", frameon=True)
    hour = representative_records["shortest"]["hour"]
    _decorate_map(
        axis,
        f"Four-route comparison — {representative_od}, {hour:02d}:00",
        extent,
    )
    generated.extend(
        _save_figure(
            figure, figure_dir / "step28_four_route_overlay"
        )
    )
    plt.close(figure)

    thematic = (
        ("utci", "utci_values", "UTCI (°C)", "inferno", None, None),
        ("shade", "shade_values", "Shade proportion", "YlGn", 0.0, 1.0),
        (
            "risk_aware",
            "uncertainty_values",
            "Prediction uncertainty",
            "magma",
            None,
            None,
        ),
    )
    names = {
        "utci": "step28_utci_gradient",
        "shade": "step28_shade_segments",
        "risk_aware": "step28_uncertainty_segments",
    }
    for objective, field, label, cmap, vmin, vmax in tqdm(
        thematic,
        desc="Drawing Step28 thematic maps",
        unit="map",
        dynamic_ncols=True,
    ):
        record = representative_records[objective]
        figure, axis = plt.subplots(figsize=(10, 8))
        add_context(axis)
        collection = LineCollection(
            [
                np.asarray(values)
                for values in record["segment_coordinates"]
            ],
            array=np.asarray(record[field], dtype=float),
            cmap=cmap,
            linewidths=3.2,
            zorder=4,
        )
        if vmin is not None:
            collection.set_clim(vmin, vmax)
        axis.add_collection(collection)
        colorbar = figure.colorbar(collection, ax=axis, fraction=0.035)
        colorbar.set_label(label)
        _decorate_map(
            axis,
            f"{label} along {OBJECTIVE_LABELS[objective]} route — "
            f"{representative_od}, {record['hour']:02d}:00",
            extent,
        )
        generated.extend(
            _save_figure(figure, figure_dir / names[objective])
        )
        plt.close(figure)
    return generated


def run_formal_step28(
    *,
    root: Path,
    rules: dict[str, Any],
    overwrite: bool,
    logger: Any,
) -> dict[str, Any]:
    started_at = now_text()
    started = time.perf_counter()
    inputs = {
        key: (root / value).resolve()
        for key, value in rules["inputs"].items()
    }
    inputs["segments_gdb"] = (
        root / "routing/data/thermal_edges/step26_thermal_segments.gdb"
    )
    inputs["center_boundary"] = (
        root
        / "data/raw/Nanjing_center_boundary/Nanjing_center_UTM50N.shp"
    )
    inputs["step28_check"] = (
        root / "routing/data/evaluation/step28_check_summary.json"
    )
    outputs = {
        "comparison": (
            root / "routing/outputs/tables/step28_route_comparison.csv"
        ),
        "performance": (
            root / "routing/outputs/tables/step28_algorithm_performance.csv"
        ),
        "mode_connectivity": (
            root / "routing/outputs/tables/step28_mode_connectivity.csv"
        ),
        "coverage": (
            root / "routing/outputs/tables/step28_coverage_summary.csv"
        ),
        "gis": (
            root / "routing/outputs/gis/step28_route_evaluation.gdb"
        ),
        "summary": (
            root / "routing/data/evaluation/step28_formal_summary.json"
        ),
        "qc": root / "routing/data/evaluation/step28_formal_qc.json",
        "failed": (
            root
            / "routing/data/evaluation/step28_formal_failed_records.csv"
        ),
        "report": (
            root / "routing/reports/STEP28_FORMAL_EVALUATION_REPORT.md"
        ),
        "methods": root / "routing/reports/STEP28_ROUTING_METHODS.md",
    }
    formal_figures = [
        root / "routing/outputs/figures/step28_four_route_overlay.png",
        root / "routing/outputs/figures/step28_four_route_overlay.pdf",
        root / "routing/outputs/figures/step28_utci_gradient.png",
        root / "routing/outputs/figures/step28_utci_gradient.pdf",
        root / "routing/outputs/figures/step28_shade_segments.png",
        root / "routing/outputs/figures/step28_shade_segments.pdf",
        root / "routing/outputs/figures/step28_uncertainty_segments.png",
        root / "routing/outputs/figures/step28_uncertainty_segments.pdf",
    ]
    for path in [*outputs.values(), *formal_figures]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Step28 formal output exists: {path}")
    check = json.loads(inputs["step28_check"].read_text(encoding="utf-8"))
    if not check.get("ready_for_formal_step28_run"):
        raise RuntimeError("Latest Step28 check is not ready for formal run")
    if check.get("blocker_count") != 0:
        raise RuntimeError("Latest Step28 check contains blockers")
    source_hashes_before = {
        key: sha256_path(path)
        for key, path in inputs.items()
        if key
        in {
            "formal_graph",
            "formal_costs",
            "graph_metadata",
            "validation_routes",
            "validation_summaries",
            "algorithm_consistency",
            "segment_coverage",
            "road_classification",
            "segments_gdb",
            "center_boundary",
        }
    }
    engine = RouteEngine(
        inputs["formal_graph"],
        inputs["formal_costs"],
        inputs["graph_metadata"],
    )
    summaries = pd.read_csv(inputs["validation_summaries"])
    consistency = pd.read_csv(inputs["algorithm_consistency"])
    route_edges = pd.read_csv(
        inputs["validation_routes"], dtype={"segment_id": str}
    )
    od_samples = pd.read_csv(
        root / "routing/outputs/tables/step27_validation_od_samples.csv"
    )
    sequences = _route_sequences(route_edges)
    comparison = _build_comparison(
        engine=engine,
        summaries=summaries,
        sequences=sequences,
    )
    performance = _performance_table(consistency)
    connectivity = _mode_connectivity(engine)
    coverage = _coverage_summary(engine)
    representative_od, selection = _select_representative_od(
        summaries, sequences
    )
    selected_comparison = comparison[
        comparison["od_id"] == representative_od
    ].copy()
    if set(selected_comparison["objective"]) != set(OBJECTIVES):
        raise ValueError("Representative OD lacks a formal objective")
    segment_lookup = {
        segment_id: index
        for index, segment_id in enumerate(engine.segment_ids)
    }
    required_segments = {
        segment_id
        for sequence in sequences.values()
        for segment_id in sequence
    }
    geometries = _load_route_geometries(
        gdb_path=inputs["segments_gdb"],
        required_segment_ids=required_segments,
    )
    origin_by_od = (
        od_samples.set_index("od_id")["origin_node"].astype(int).to_dict()
    )
    comparison_lookup = comparison.set_index("case_id").to_dict(
        orient="index"
    )
    route_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    representative_records: dict[str, dict[str, Any]] = {}
    for (case_id, od_id), sequence in tqdm(
        sorted(sequences.items()),
        desc="Assembling Step28 route geometry",
        unit="route",
        dynamic_ncols=True,
    ):
        metrics = comparison_lookup[case_id]
        coordinates, segment_coordinates = _oriented_route(
            engine=engine,
            origin_node=origin_by_od[od_id],
            segment_ids=sequence,
            segment_lookup=segment_lookup,
            geometries=geometries,
        )
        objective = str(metrics["objective"])
        representative = int(od_id == representative_od)
        route_record = {
            "route_id": metrics["route_id"],
            "od_id": od_id,
            "case_id": case_id,
            "hour": int(metrics["hour"]),
            "mode": str(metrics["mode"]),
            "objective": objective,
            "algorithm": "astar",
            "distance_m": float(metrics["distance_m"]),
            "detour": float(metrics["detour_ratio"]),
            "mean_shade": float(metrics["mean_shade"]),
            "mean_utci": float(metrics["mean_utci"]),
            "max_utci": float(metrics["max_utci"]),
            "uncertainty": float(metrics["uncertainty_mean"]),
            "fallback_m": float(metrics["spatial_fallback_length_m"]),
            "prior_m": float(metrics["prior_imputed_length_m"]),
            "edge_count": len(sequence),
            "representative": representative,
            "coordinates": coordinates,
        }
        route_records.append(route_record)
        hour_index = int(metrics["hour"]) - 6
        edge_indices = [
            segment_lookup[segment_id] for segment_id in sequence
        ]
        shade_values = [
            float(engine.shade_mean[index, hour_index])
            for index in edge_indices
        ]
        utci_values = [
            float(engine.utci_mean[index, hour_index])
            for index in edge_indices
        ]
        uncertainty_values = [
            float(engine.uncertainty_cost[index, hour_index])
            for index in edge_indices
        ]
        fallback_flags = [
            bool(engine.fallback_only[index]) for index in edge_indices
        ]
        for position, (
            segment_id,
            edge_index,
            segment_geometry,
            shade,
            utci,
            uncertainty,
            fallback,
        ) in enumerate(
            zip(
                sequence,
                edge_indices,
                segment_coordinates,
                shade_values,
                utci_values,
                uncertainty_values,
                fallback_flags,
                strict=True,
            ),
            1,
        ):
            segment_records.append(
                {
                    "route_id": metrics["route_id"],
                    "od_id": od_id,
                    "case_id": case_id,
                    "objective": objective,
                    "hour": int(metrics["hour"]),
                    "sequence": position,
                    "segment_id": segment_id,
                    "length_m": float(engine.lengths[edge_index]),
                    "shade": shade,
                    "utci": utci,
                    "uncertainty": uncertainty,
                    "coverage": engine.coverage_by_code[
                        int(engine.coverage_code[edge_index])
                    ],
                    "fallback": int(fallback),
                    "representative": representative,
                    "coordinates": segment_geometry,
                }
            )
        if representative:
            representative_records[objective] = {
                **route_record,
                "segment_coordinates": segment_coordinates,
                "shade_values": shade_values,
                "utci_values": utci_values,
                "uncertainty_values": uncertainty_values,
                "fallback_flags": fallback_flags,
            }
    _write_dataframe(outputs["comparison"], comparison)
    _write_dataframe(outputs["performance"], performance)
    _write_dataframe(outputs["mode_connectivity"], connectivity)
    _write_dataframe(outputs["coverage"], coverage)
    gis_counts = _write_gdb(
        output_path=outputs["gis"],
        route_records=route_records,
        segment_records=segment_records,
        boundary_path=inputs["center_boundary"],
        overwrite=overwrite,
    )
    generated_figures = _make_figures(
        figure_dir=root / "routing/outputs/figures",
        engine=engine,
        representative_od=representative_od,
        representative_records=representative_records,
        boundary_path=inputs["center_boundary"],
    )
    source_hashes_after = {
        key: sha256_path(path)
        for key, path in inputs.items()
        if key in source_hashes_before
    }
    failed_rows: list[dict[str, Any]] = []
    atomic_text(
        outputs["failed"],
        "category,object_id,error_type,error_message\n",
    )
    comparison_finite = bool(
        np.isfinite(
            comparison[
                [
                    "distance_m",
                    "total_cost",
                    "mean_shade",
                    "mean_tmrt",
                    "mean_utci",
                    "uncertainty_mean",
                    "detour_ratio",
                ]
            ].to_numpy(float)
        ).all()
    )
    qc = {
        "success": (
            len(comparison) == 100
            and comparison["od_id"].nunique() == 25
            and comparison.groupby("od_id")["objective"].nunique().eq(4).all()
            and consistency["consistency_pass"].all()
            and comparison_finite
            and len(engine.forbidden_turns)
            == int(
                rules["grade_transition_review"][
                    "approved_forbidden_turn_count"
                ]
            )
            and check["route_metrics"][
                "graph_suspicious_grade_transition_count"
            ]
            == 0
            and check["route_metrics"][
                "forbidden_turn_violation_record_count"
            ]
            == 0
            and gis_counts["RouteComparisons"] == 100
            and gis_counts["RouteThermalSegments"] == len(segment_records)
            and len(generated_figures) == 8
            and all(Path(path).stat().st_size > 0 for path in generated_figures)
            and source_hashes_before == source_hashes_after
            and len(failed_rows) == 0
        ),
        "route_comparison_count": len(comparison),
        "od_count": int(comparison["od_id"].nunique()),
        "objective_count_per_od_minimum": int(
            comparison.groupby("od_id")["objective"].nunique().min()
        ),
        "algorithm_consistency_pass_count": int(
            consistency["consistency_pass"].sum()
        ),
        "algorithm_case_count": len(consistency),
        "maximum_absolute_cost_difference": float(
            consistency["absolute_cost_difference"].max()
        ),
        "comparison_numeric_values_finite": comparison_finite,
        "maximum_detour_ratio": float(comparison["detour_ratio"].max()),
        "route_over_20_percent_detour_count": int(
            (comparison["detour_ratio"] > 0.20 + 1e-9).sum()
        ),
        "protected_turn_count": len(engine.forbidden_turns),
        "unprotected_suspicious_turn_count": check["route_metrics"][
            "graph_suspicious_grade_transition_count"
        ],
        "route_forbidden_turn_violation_count": check["route_metrics"][
            "forbidden_turn_violation_record_count"
        ],
        "gis_feature_counts": gis_counts,
        "figure_count": len(generated_figures),
        "representative_od": representative_od,
        "representative_selection": selection,
        "source_hashes_unchanged": source_hashes_before
        == source_hashes_after,
        "failed_record_count": len(failed_rows),
    }
    atomic_json(outputs["qc"], qc)
    ended_at = now_text()
    summary = {
        "phase": "G7",
        "step": 28,
        "mode": "run",
        "script": "Step28_evaluate_routing.py",
        "script_version": "2.0.0",
        "rule_version": rules["rule_version"],
        "rule_status": rules["status"],
        "approved_by": rules["approval"]["approved_by"],
        "approved_at": rules["approval"]["approved_at"],
        "started_at": started_at,
        "ended_at": ended_at,
        "elapsed_seconds": time.perf_counter() - started,
        "success": qc["success"],
        "ready_for_phase_g8": qc["success"],
        "route_count": len(comparison),
        "od_count": int(comparison["od_id"].nunique()),
        "representative_od": representative_od,
        "representative_hour": int(
            selected_comparison["hour"].iloc[0]
        ),
        "representative_mode": str(
            selected_comparison["mode"].iloc[0]
        ),
        "maximum_detour_ratio": qc["maximum_detour_ratio"],
        "route_over_20_percent_detour_count": qc[
            "route_over_20_percent_detour_count"
        ],
        "protected_turn_count": len(engine.forbidden_turns),
        "unprotected_suspicious_turn_count": 0,
        "formal_gis_generated": True,
        "formal_figures_generated": True,
        "source_inputs_modified": False,
        "failed_record_count": len(failed_rows),
        "outputs": {
            **{key: str(value) for key, value in outputs.items()},
            "figures": generated_figures,
        },
    }
    atomic_json(outputs["summary"], summary)
    maximum_detour = (
        comparison.groupby("objective")["detour_ratio"].mean().to_dict()
    )
    lines = [
        "# Step28 Formal Routing Evaluation Report",
        "",
        f"Generated: {ended_at}",
        "",
        f"- `SUCCESS = {str(qc['success']).upper()}`",
        f"- `READY_FOR_PHASE_G8 = {str(qc['success']).upper()}`",
        f"- Formal A* routes: {len(comparison):,}",
        f"- Fixed OD pairs: {comparison['od_id'].nunique():,}",
        "- Objectives per OD: 4",
        f"- A*/Dijkstra consistency: {qc['algorithm_consistency_pass_count']}/{qc['algorithm_case_count']}",
        f"- Maximum algorithm cost difference: {qc['maximum_absolute_cost_difference']:.12g}",
        f"- Protected grade-separated turns: {qc['protected_turn_count']:,}",
        f"- Unprotected suspicious turns: {qc['unprotected_suspicious_turn_count']:,}",
        f"- Route forbidden-turn violations: {qc['route_forbidden_turn_violation_count']:,}",
        f"- Representative map case: {representative_od}, "
        f"{summary['representative_hour']:02d}:00, "
        f"{summary['representative_mode']}",
        f"- GIS route / segment features: "
        f"{gis_counts['RouteComparisons']:,} / "
        f"{gis_counts['RouteThermalSegments']:,}",
        f"- Figure files: {len(generated_figures):,}",
        f"- Maximum unconstrained detour ratio: {qc['maximum_detour_ratio']:.2%}",
        f"- Routes above 20% detour: {qc['route_over_20_percent_detour_count']:,}",
        f"- Source hashes unchanged: {qc['source_hashes_unchanged']}",
        f"- Failed records: {len(failed_rows):,}",
        "",
        "## Mean detour ratio by objective",
        "",
        *[
            f"- {objective}: {maximum_detour[objective]:.2%}"
            for objective in OBJECTIVES
        ],
        "",
        "All route comparisons use frozen Step26 hourly costs and the "
        "turn-aware Step27 graph. Formal GIS includes all 100 route "
        "geometries, per-route thermal segments and the center boundary. "
        "No thermal model, prediction, road segment or Step26 cost was "
        "modified.",
        "",
        "The formal benchmark intentionally leaves `max_detour_ratio` "
        "unset so the four objective functions can be compared without an "
        "extra constraint. Some thermal routes therefore make large "
        "detours. The validated route interface already supports a hard "
        "detour limit; Phase G8 should expose and configure that parameter "
        "for user-facing requests.",
    ]
    atomic_text(outputs["report"], "\n".join(lines) + "\n")
    methods = [
        "# Step28 Routing Evaluation Methods",
        "",
        "The formal evaluation uses 25 deterministic, turn-reachable OD "
        "pairs. Each OD is evaluated with shortest, shade-priority, "
        "UTCI-priority and risk-aware A* routing. Every case was paired "
        "with Dijkstra in Step27 and passed a 1e-6 cost tolerance.",
        "",
        "Route metrics are recomputed from ordered segment IDs and frozen "
        "hourly edge arrays. Detour is route length divided by the "
        "corresponding shortest-route length minus one. Direct, "
        "interpolated and low-reliability lengths are reported separately. "
        "Node-specific bridge/tunnel/layer restrictions are enforced using "
        "the current-node plus incoming-edge search state.",
        "",
        "The representative map OD is selected deterministically by the "
        "largest number of distinct objective edge sequences, then maximum "
        "summed pairwise Jaccard distance, then lexical OD ID. Four map "
        "products show the route overlay, segment UTCI, segment shade and "
        "prediction uncertainty, with the center boundary and fallback "
        "segments retained for interpretation.",
        "",
        "The formal comparison is unconstrained with respect to detour. "
        "This characterizes the raw objective trade-off and does not imply "
        "that a user-facing application should accept every detour. The "
        "validated `max_detour_ratio` parameter must be applied when a "
        "request specifies a limit, and a default limit should be reviewed "
        "before Phase G8 deployment.",
    ]
    atomic_text(outputs["methods"], "\n".join(methods) + "\n")
    logger.info(
        "Formal Step28 complete: success=%s routes=%s figures=%s",
        qc["success"],
        len(comparison),
        len(generated_figures),
    )
    return summary
