"""Post-repair QC and focused four-route map for od_025 at 17:00."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from routing.src.route_engine import RouteEngine


def main() -> int:
    engine = RouteEngine(
        ROOT / "routing/data/graph/step27_route_graph.npz",
        ROOT / "routing/data/graph/step27_hourly_costs.npz",
        ROOT / "routing/data/graph/step27_graph_metadata.json",
    )
    origin = (667137.814, 3547670.695)
    destination = (667727.326, 3541300.682)
    styles = {
        "shortest": ("#3561c8", "Shortest"),
        "shade": ("#20a162", "Shade priority"),
        "utci": ("#f03b20", "UTCI priority"),
        "risk_aware": ("#8e017e", "Risk-aware"),
    }
    segment_lookup = {value: index for index, value in enumerate(engine.segment_ids.tolist())}
    routes = []
    for objective in tqdm(styles, total=len(styles), desc="Routing repaired od_025", unit="route", dynamic_ncols=True):
        result = engine.route(origin, destination, 17, mode="walk", objective=objective, algorithm="astar")
        nodes = np.asarray(result["node_indices"], dtype=np.int32)
        original_ids = engine.original_edge_ids[
            np.asarray([segment_lookup[segment] for segment in result["segment_ids"]], dtype=np.int32)
        ].tolist()
        routes.append({
            "objective": objective,
            "distance_m": result["distance_m"],
            "duration_minutes": result["estimated_duration_min"],
            "fallback_retry": result["fallback_retry"],
            "uses_approved_connector": "EXT_GAP_0001" in original_ids,
            "node_indices": nodes,
        })

    all_xy = np.vstack([engine.node_xy[row["node_indices"]] for row in routes])
    padding = 700.0
    low = all_xy.min(axis=0) - padding
    high = all_xy.max(axis=0) + padding
    edge_mask = (
        (engine.node_xy[engine.edge_u, 0] >= low[0]) & (engine.node_xy[engine.edge_u, 0] <= high[0])
        & (engine.node_xy[engine.edge_u, 1] >= low[1]) & (engine.node_xy[engine.edge_u, 1] <= high[1])
    ) | (
        (engine.node_xy[engine.edge_v, 0] >= low[0]) & (engine.node_xy[engine.edge_v, 0] <= high[0])
        & (engine.node_xy[engine.edge_v, 1] >= low[1]) & (engine.node_xy[engine.edge_v, 1] <= high[1])
    )
    local_edges = np.flatnonzero(edge_mask)
    background = np.stack((engine.node_xy[engine.edge_u[local_edges]], engine.node_xy[engine.edge_v[local_edges]]), axis=1)
    figure, axis = plt.subplots(figsize=(13, 10))
    axis.add_collection(LineCollection(background, colors="#dedede", linewidths=0.45, alpha=0.65, zorder=1))
    for row in tqdm(routes, total=len(routes), desc="Plotting repaired od_025", unit="route", dynamic_ncols=True):
        color, label = styles[row["objective"]]
        xy = engine.node_xy[row["node_indices"]]
        axis.plot(xy[:, 0], xy[:, 1], color=color, linewidth=2.0, label=f"{label} ({row['distance_m']/1000:.2f} km)", zorder=3)
    axis.scatter(*origin, c="black", s=70, label="Origin", zorder=5)
    axis.scatter(*destination, c="#f6a21a", marker="*", edgecolors="black", s=180, label="Destination", zorder=5)
    axis.set_xlim(low[0], high[0]); axis.set_ylim(low[1], high[1]); axis.set_aspect("equal", adjustable="box")
    axis.set_title("Four-route comparison after Step24c topology repair — od_025, 17:00")
    axis.set_xlabel("Easting (m), EPSG:32650"); axis.set_ylabel("Northing (m), EPSG:32650")
    axis.grid(alpha=0.12); axis.legend(loc="best")
    output = ROOT / "routing/outputs/figures/od_025_17_four_route_comparison_post_step24c.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(figure)

    table = ROOT / "routing/outputs/tables/step24c_od025_postrepair_qc.csv"
    with table.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["objective", "distance_m", "duration_minutes", "fallback_retry", "uses_approved_connector"])
        writer.writeheader()
        for row in routes:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    success = all(row["uses_approved_connector"] for row in routes) and max(row["distance_m"] for row in routes) < 10000.0
    summary = {
        "success": success,
        "all_routes_use_approved_connector": all(row["uses_approved_connector"] for row in routes),
        "maximum_route_distance_m": max(row["distance_m"] for row in routes),
        "figure": str(output), "table": str(table),
    }
    summary_path = ROOT / "routing/data/topology/step24c_od025_postrepair_qc.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
