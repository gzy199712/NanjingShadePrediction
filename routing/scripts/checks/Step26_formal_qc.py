"""Independent chunked QC for formal Step26 thermal segment costs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routing.src.reporting import atomic_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--costs",
        type=Path,
        default=(
            PROJECT_ROOT
            / "routing/data/thermal_edges/road_hour_thermal_cost.csv"
        ),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=(
            PROJECT_ROOT
            / "routing/data/thermal_edges/thermal_segment_coverage.csv"
        ),
    )
    parser.add_argument(
        "--gdb",
        type=Path,
        default=(
            PROJECT_ROOT
            / "routing/data/thermal_edges/step26_thermal_segments.gdb"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            PROJECT_ROOT
            / "routing/data/thermal_edges/step26_independent_qc.json"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coverage = pd.read_csv(
        args.coverage,
        dtype={"segment_id": str, "original_edge_id": str},
        low_memory=False,
    )
    coverage_types = set(coverage["coverage_type"])
    expected_hours = np.arange(6, 19, dtype=np.int64)
    row_count = 0
    nonfinite_count = 0
    negative_std_count = 0
    shade_out_of_range_count = 0
    invalid_hour_count = 0
    invalid_formal_flag_count = 0
    order_error_count = 0
    coverage_row_counts: dict[str, int] = {}
    first_rows: list[tuple[str, int]] = []
    last_rows: list[tuple[str, int]] = []
    numeric_columns = [
        "length_m",
        "shade_mean",
        "shade_std",
        "tmrt_mean",
        "tmrt_std",
        "utci_mean",
        "utci_std",
        "uncertainty_cost",
    ]
    reader = pd.read_csv(
        args.costs,
        dtype={
            "segment_id": str,
            "original_edge_id": str,
            "source_edge_id": str,
        },
        chunksize=100_000,
        low_memory=False,
    )
    previous_tail: list[tuple[str, int]] = []
    for chunk in tqdm(
        reader,
        desc="Independent Step26 cost QC",
        unit="chunk",
        dynamic_ncols=True,
    ):
        row_count += len(chunk)
        values = chunk[numeric_columns].to_numpy(float)
        nonfinite_count += int((~np.isfinite(values)).sum())
        negative_std_count += int(
            (
                chunk[["shade_std", "tmrt_std", "utci_std"]].to_numpy(float)
                < 0
            ).sum()
        )
        shade_out_of_range_count += int(
            (
                (chunk["shade_mean"].to_numpy(float) < 0)
                | (chunk["shade_mean"].to_numpy(float) > 1)
            ).sum()
        )
        invalid_hour_count += int(
            (~chunk["hour"].isin(expected_hours)).sum()
        )
        formal_flags = (
            chunk["formal_edge_cost"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1"})
        )
        invalid_formal_flag_count += int((~formal_flags).sum())
        for key, value in chunk["coverage_type"].value_counts().items():
            coverage_row_counts[str(key)] = (
                coverage_row_counts.get(str(key), 0) + int(value)
            )
        pairs = list(
            zip(
                chunk["segment_id"].astype(str),
                chunk["hour"].astype(int),
                strict=True,
            )
        )
        combined = previous_tail + pairs
        for index in range(1, len(combined)):
            previous_segment, previous_hour = combined[index - 1]
            segment, hour = combined[index]
            valid = (
                segment == previous_segment and hour == previous_hour + 1
            ) or (
                segment != previous_segment
                and previous_hour == 18
                and hour == 6
            )
            if not valid:
                order_error_count += 1
        previous_tail = combined[-1:]
        if not first_rows:
            first_rows = pairs[:13]
        last_rows = pairs[-13:]
    from osgeo import ogr

    dataset = ogr.Open(str(args.gdb), 0)
    layer = dataset.GetLayerByName("ThermalCostSegments")
    gdb_count = int(layer.GetFeatureCount())
    gdb_invalid_geometry_count = 0
    gdb_max_length_m = 0.0
    for feature in tqdm(
        layer,
        total=gdb_count,
        desc="Independent Step26 GDB QC",
        unit="segment",
        dynamic_ncols=True,
    ):
        geometry = feature.GetGeometryRef()
        if geometry is None or geometry.IsEmpty() or not geometry.IsValid():
            gdb_invalid_geometry_count += 1
            continue
        gdb_max_length_m = max(gdb_max_length_m, float(geometry.Length()))
    expected_rows = len(coverage) * 13
    distance_rule_error_count = int(
        (
            (
                (coverage["coverage_type"] == "spatial_interpolated")
                & (coverage["nearest_point_distance_m"] > 150)
            )
            | (
                (coverage["coverage_type"] == "spatial_interpolated_low")
                & (
                    (coverage["nearest_point_distance_m"] <= 150)
                    | (coverage["nearest_point_distance_m"] > 300)
                )
            )
            | (
                (coverage["coverage_type"] == "spatial_fallback")
                & (
                    (coverage["nearest_point_distance_m"] <= 300)
                    | (coverage["nearest_point_distance_m"] > 500)
                )
            )
            | (
                (coverage["coverage_type"] == "structure_interpolated")
                & (coverage["nearest_point_distance_m"] > 500)
            )
        ).sum()
    )
    expected_coverage_rows = {
        str(key): int(value * 13)
        for key, value in coverage["coverage_type"].value_counts().items()
    }
    qc = {
        "success": all(
            (
                len(coverage) == coverage["segment_id"].nunique(),
                row_count == expected_rows,
                gdb_count == len(coverage),
                nonfinite_count == 0,
                negative_std_count == 0,
                shade_out_of_range_count == 0,
                invalid_hour_count == 0,
                invalid_formal_flag_count == 0,
                distance_rule_error_count == 0,
                order_error_count == 0,
                gdb_invalid_geometry_count == 0,
                gdb_max_length_m <= 50.000001,
                coverage_types == set(coverage_row_counts),
                coverage_row_counts == expected_coverage_rows,
            )
        ),
        "coverage_segment_count": int(len(coverage)),
        "unique_segment_id_count": int(coverage["segment_id"].nunique()),
        "expected_cost_row_count": int(expected_rows),
        "actual_cost_row_count": int(row_count),
        "gdb_feature_count": gdb_count,
        "gdb_invalid_geometry_count": gdb_invalid_geometry_count,
        "gdb_max_segment_length_m": gdb_max_length_m,
        "nonfinite_numeric_value_count": nonfinite_count,
        "negative_std_value_count": negative_std_count,
        "shade_out_of_range_count": shade_out_of_range_count,
        "invalid_hour_count": invalid_hour_count,
        "invalid_formal_flag_count": invalid_formal_flag_count,
        "distance_rule_error_count": distance_rule_error_count,
        "segment_hour_order_error_count": order_error_count,
        "coverage_cost_row_counts": coverage_row_counts,
        "expected_coverage_cost_row_counts": expected_coverage_rows,
        "first_segment_hours": [hour for _, hour in first_rows],
        "last_segment_hours": [hour for _, hour in last_rows],
    }
    atomic_json(args.output, qc)
    print(qc)
    return 0 if qc["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
