"""Step114: extract multiweather labels and build aligned multi-date training tables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data_pipeline/configs/multiweather/step114_multidate_training_tables.yaml"
LABEL_FIELDS = [
    "point_id", "date", "hour", "datetime_local",
    "shade_rate", "shade_nonshade_rate", "shade_mean_raw",
    "shade_valid_pixels", "shade_nodata_pixels", "shade_total_candidate_pixels", "shade_valid_ratio",
    "tmrt_mean", "tmrt_std", "tmrt_min", "tmrt_max", "tmrt_p10", "tmrt_p90",
    "tmrt_valid_pixels", "tmrt_nodata_pixels", "tmrt_total_candidate_pixels", "tmrt_valid_ratio",
    "utci_mean", "utci_std", "utci_min", "utci_max", "utci_p10", "utci_p90",
    "utci_valid_pixels", "utci_nodata_pixels", "utci_total_candidate_pixels", "utci_valid_ratio",
]
DYNAMIC_FIELDS = [
    "date", "hour", "datetime_local", "representative_longitude", "representative_latitude",
    "solar_altitude", "solar_azimuth", "sin_solar_azimuth", "cos_solar_azimuth",
    "sin_solar_altitude", "cos_solar_altitude", "air_temperature", "relative_humidity",
    "wind_speed", "wind_direction", "sin_wind_direction", "cos_wind_direction",
    "air_pressure_kpa", "rainfall", "global_shortwave_radiation",
    "direct_shortwave_radiation", "diffuse_shortwave_radiation",
]
MANIFEST_FIELDS = [
    "sample_id", "point_id", "date", "hour", "static_feature_row", "dino_mean_row",
    "dino_window_row", "dynamic_condition_row", "label_row", "spatial_block_id",
    "overlap_component_id", "dataset_split",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def logger_for(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step114"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def create_zone_rasters(shapefile: Path, groups_csv: Path, reference_grid: Path, zone_dir: Path, group_count: int) -> list[Path]:
    zone_dir.mkdir(parents=True, exist_ok=True)
    paths = [zone_dir / f"outdoor_zone_group_{group}.tif" for group in range(group_count)]
    if all(path.exists() for path in paths):
        return paths
    groups = pd.read_csv(groups_csv, dtype={"point_id": str}).set_index("point_id")["color_group"].astype(int).to_dict()
    source = ogr.Open(str(shapefile), 0)
    if source is None:
        raise RuntimeError(f"OGR cannot open {shapefile}")
    source_layer = source.GetLayer(0); spatial_ref = source_layer.GetSpatialRef()
    memories = []
    for group in range(group_count):
        memory = ogr.GetDriverByName("Memory").CreateDataSource(f"group_{group}")
        layer = memory.CreateLayer(f"group_{group}", srs=spatial_ref, geom_type=ogr.wkbMultiPolygon)
        layer.CreateField(ogr.FieldDefn("ID", ogr.OFTInteger))
        memories.append((memory, layer))
    for feature in tqdm(source_layer, total=source_layer.GetFeatureCount(), desc="Grouping outdoor polygons", unit="polygon", dynamic_ncols=True):
        point_id = str(feature.GetField("point_id")); group = groups[point_id]
        output = ogr.Feature(memories[group][1].GetLayerDefn())
        output.SetField("ID", int(point_id)); output.SetGeometry(feature.GetGeometryRef().Clone())
        if memories[group][1].CreateFeature(output) != 0:
            raise RuntimeError(f"Failed to copy outdoor polygon {point_id}")
    reference = gdal.Open(str(reference_grid), gdal.GA_ReadOnly)
    transform = reference.GetGeoTransform(); projection = reference.GetProjection()
    for group, path in enumerate(tqdm(paths, desc="Rasterizing overlap-safe zones", unit="group", dynamic_ncols=True)):
        driver = gdal.GetDriverByName("GTiff")
        target = driver.Create(str(path), reference.RasterXSize, reference.RasterYSize, 1, gdal.GDT_Int32, options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=IF_SAFER"])
        target.SetGeoTransform(transform); target.SetProjection(projection)
        band = target.GetRasterBand(1); band.Fill(0); band.SetNoDataValue(0)
        error = gdal.RasterizeLayer(target, [1], memories[group][1], options=["ATTRIBUTE=ID"])
        if error != 0:
            raise RuntimeError(f"RasterizeLayer failed for group {group}")
        band.FlushCache(); target.FlushCache(); target = None
    source = None
    return paths


def load_zones(paths: list[Path]) -> tuple[list[np.ndarray], dict[int, int]]:
    arrays = []
    counts: dict[int, int] = {}
    for path in tqdm(paths, desc="Loading zone rasters", unit="group", dynamic_ncols=True):
        dataset = gdal.Open(str(path), gdal.GA_ReadOnly); array = dataset.GetRasterBand(1).ReadAsArray().astype(np.int32, copy=False)
        arrays.append(array)
        ids, values = np.unique(array[array > 0], return_counts=True)
        counts.update({int(point_id): int(count) for point_id, count in zip(ids, values)})
    return arrays, counts


def stats_for_array(array: np.ndarray, zones: list[np.ndarray]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    for zone in zones:
        mask = (zone > 0) & np.isfinite(array)
        ids = zone[mask].astype(np.int32, copy=False); values = array[mask].astype(np.float64, copy=False)
        order = np.argsort(ids, kind="stable"); ids = ids[order]; values = values[order]
        unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
        for point_id, start, count in zip(unique, starts, counts):
            part = values[start:start + count]
            p10, p90 = np.percentile(part, [10, 90], method="linear")
            result[int(point_id)] = {
                "mean": float(np.mean(part)), "std": float(np.std(part, ddof=0)),
                "min": float(np.min(part)), "max": float(np.max(part)),
                "p10": float(p10), "p90": float(p90), "count": int(count),
            }
    return result


def band_stats(path: Path, band: int, zones: list[np.ndarray]) -> dict[int, dict[str, float]]:
    dataset = gdal.Open(str(path), gdal.GA_ReadOnly)
    if dataset is None or dataset.RasterCount < band:
        raise RuntimeError(f"Cannot read band {band} from {path}")
    array = dataset.GetRasterBand(band).ReadAsArray()
    output = stats_for_array(array, zones); dataset = None
    return output


def preflight(config: dict, root: Path, process: Path, logger: logging.Logger) -> dict[str, Any]:
    zone_paths = create_zone_rasters(root / config["outdoor_units"], root / config["overlap_groups"], Path(config["reference_grid"]), process / "zone_rasters", int(config["zone_groups"]))
    zones, counts = load_zones(zone_paths)
    labels = pd.read_csv(root / config["reference_labels"])
    reference = labels.loc[labels.hour == int(config["hours"][0]), ["point_id", "tmrt_total_candidate_pixels"]].copy()
    reference["observed"] = reference.point_id.map(counts).fillna(0).astype(int)
    reference["difference"] = reference.observed - reference.tmrt_total_candidate_pixels.astype(int)
    noon = 12; reference_noon = labels.loc[labels.hour == noon].set_index("point_id")
    comparisons = []
    for source_name, path_key in (("tmrt", "reference_tmrt"), ("utci", "reference_utci")):
        stats = band_stats(root / config[path_key], noon + 1, zones)
        for point_id in tqdm(sorted(stats), desc=f"Cross-checking {source_name}", unit="point", dynamic_ncols=True):
            current = stats[point_id]; expected = reference_noon.loc[point_id]
            comparisons.append({
                "source": source_name, "point_id": point_id,
                "mean_abs": abs(current["mean"] - float(expected[f"{source_name}_mean"])),
                "std_abs": abs(current["std"] - float(expected[f"{source_name}_std"])),
                "min_abs": abs(current["min"] - float(expected[f"{source_name}_min"])),
                "max_abs": abs(current["max"] - float(expected[f"{source_name}_max"])),
                "p10_abs": abs(current["p10"] - float(expected[f"{source_name}_p10"])),
                "p90_abs": abs(current["p90"] - float(expected[f"{source_name}_p90"])),
            })
    comparison = pd.DataFrame(comparisons)
    candidate_exact = float((reference.difference == 0).mean())
    thresholds = {"mean_abs": 0.2, "std_abs": 0.2, "min_abs": 0.5, "max_abs": 0.5, "p10_abs": 0.3, "p90_abs": 0.3}
    comparison["pass"] = True
    for field, threshold in thresholds.items():
        comparison["pass"] &= comparison[field] <= threshold
    comparison.to_csv(process / "reference_method_crosscheck.csv", index=False, encoding="utf-8-sig")
    reference.to_csv(process / "candidate_pixel_crosscheck.csv", index=False, encoding="utf-8-sig")
    summary = {
        "status": "PREFLIGHT_PASS" if candidate_exact >= 0.995 and bool(comparison["pass"].all()) else "PREFLIGHT_FAIL",
        "point_count": len(counts), "candidate_exact_ratio": candidate_exact,
        "candidate_max_absolute_difference": int(reference.difference.abs().max()),
        "boundary_candidate_overrides": reference.loc[reference.difference != 0, "point_id"].astype(int).tolist(),
        "statistics_rows": len(comparison), "statistics_failure_count": int((~comparison["pass"]).sum()),
        "maximum_differences": {field: float(comparison[field].max()) for field in thresholds},
        "zone_rasters": [str(path) for path in zone_paths],
        "method": "GDAL overlap-safe center-cell zonal statistics; NumPy linear percentiles",
    }
    write_json(process / "preflight.json", summary); logger.info("preflight=%s", json.dumps(summary, ensure_ascii=False))
    return summary


def dynamic_table(config: dict, root: Path, output: Path) -> tuple[pd.DataFrame, dict[tuple[str, int], int]]:
    reference = pd.read_csv(root / config["reference_dynamic"]); reference.date = reference.date.astype(str)
    selected = pd.read_csv(root / config["selected_dates"]).date.astype(str).tolist()
    hourly = pd.read_csv(root / config["era5_hourly"]); hourly.date = hourly.date.astype(str)
    solar = reference.set_index("hour")
    rows = [reference]
    for date in selected:
        part = hourly.loc[(hourly.date == date) & hourly.hour.isin(config["hours"])].copy().sort_values("hour")
        if len(part) != len(config["hours"]):
            raise RuntimeError(f"Incomplete hourly weather for {date}")
        values = []
        for _, weather in part.iterrows():
            base = solar.loc[int(weather.hour)]
            wind_radians = math.radians(float(weather.wind_direction))
            values.append({
                "date": date, "hour": int(weather.hour), "datetime_local": f"{date}T{int(weather.hour):02d}:00:00+08:00",
                **{field: base[field] for field in DYNAMIC_FIELDS[3:11]},
                "air_temperature": weather.air_temperature, "relative_humidity": weather.relative_humidity,
                "wind_speed": weather.wind_speed, "wind_direction": weather.wind_direction,
                "sin_wind_direction": math.sin(wind_radians), "cos_wind_direction": math.cos(wind_radians),
                "air_pressure_kpa": weather.air_pressure_kpa, "rainfall": weather.rainfall,
                "global_shortwave_radiation": weather.global_shortwave_radiation,
                "direct_shortwave_radiation": weather.direct_shortwave_radiation,
                "diffuse_shortwave_radiation": weather.diffuse_shortwave_radiation,
            })
        rows.append(pd.DataFrame(values))
    combined = pd.concat(rows, ignore_index=True)[DYNAMIC_FIELDS].sort_values(["date", "hour"]).reset_index(drop=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    return combined, {(str(row.date), int(row.hour)): index for index, row in combined.iterrows()}


def label_row(point_id: int, date: str, hour: int, shade: pd.Series, tmrt: dict, utci: dict, candidate: int) -> dict[str, Any]:
    output = {field: shade[field] for field in LABEL_FIELDS if field.startswith("shade_")}
    output.update({"point_id": point_id, "date": date, "hour": hour, "datetime_local": f"{date}T{hour:02d}:00:00+08:00"})
    for name, stats in (("tmrt", tmrt), ("utci", utci)):
        valid = int(stats["count"]); nodata = int(candidate - valid)
        output.update({
            f"{name}_mean": stats["mean"], f"{name}_std": stats["std"], f"{name}_min": stats["min"],
            f"{name}_max": stats["max"], f"{name}_p10": stats["p10"], f"{name}_p90": stats["p90"],
            f"{name}_valid_pixels": valid, f"{name}_nodata_pixels": nodata,
            f"{name}_total_candidate_pixels": candidate, f"{name}_valid_ratio": valid / candidate if candidate else math.nan,
        })
    return output


def extract_date(config: dict, root: Path, date: str, zones: list[np.ndarray], counts: dict[int, int], shade_lookup: pd.DataFrame, part_path: Path) -> None:
    date_key = date.replace("-", ""); base = root / config["solweig_results"] / date_key
    tmrt_path = base / "Tmrt" / f"TMRT_{date_key}_10m.tif"; utci_path = base / "UTCI" / f"UTCI_{date_key}_10m.tif"
    rows = []
    for band, hour in enumerate(tqdm(config["hours"], desc=f"Extracting {date}", unit="hour", dynamic_ncols=True), start=1):
        tmrt = band_stats(tmrt_path, band, zones); utci = band_stats(utci_path, band, zones)
        for point_id in sorted(counts):
            if point_id not in tmrt or point_id not in utci:
                raise RuntimeError(f"Missing thermal statistics: {date} point {point_id} hour {hour}")
            rows.append(label_row(point_id, date, int(hour), shade_lookup.loc[(point_id, int(hour))], tmrt[point_id], utci[point_id], counts[point_id]))
    pd.DataFrame(rows)[LABEL_FIELDS].sort_values(["point_id", "hour"]).to_csv(part_path, index=False, encoding="utf-8-sig")


def concatenate_labels(parts: list[Path], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True); total = 0
    with destination.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=LABEL_FIELDS); writer.writeheader()
        for path in tqdm(parts, desc="Combining multi-date labels", unit="date", dynamic_ncols=True):
            with path.open("r", newline="", encoding="utf-8-sig") as source:
                for row in csv.DictReader(source): writer.writerow(row); total += 1
    return total


def build_manifest(config: dict, root: Path, labels_path: Path, dynamic_index: dict[tuple[str, int], int], output: Path) -> int:
    base = pd.read_csv(root / config["reference_manifest"], dtype={"point_id": str})
    metadata = base.drop_duplicates("point_id").set_index("point_id")
    count = 0
    with labels_path.open("r", newline="", encoding="utf-8-sig") as source, output.open("w", newline="", encoding="utf-8-sig") as target:
        writer = csv.DictWriter(target, fieldnames=MANIFEST_FIELDS); writer.writeheader()
        for row in tqdm(csv.DictReader(source), total=int(config["expected_points"]) * 14 * len(config["hours"]), desc="Building multi-date manifest", unit="sample", dynamic_ncols=True):
            point_id = str(int(row["point_id"])); date = row["date"]; hour = int(row["hour"]); item = metadata.loc[point_id]
            writer.writerow({
                "sample_id": f"{point_id}_{date.replace('-', '')}_{hour:02d}", "point_id": point_id, "date": date, "hour": hour,
                "static_feature_row": int(item.static_feature_row), "dino_mean_row": int(item.dino_mean_row), "dino_window_row": int(item.dino_window_row),
                "dynamic_condition_row": dynamic_index[(date, hour)], "label_row": count,
                "spatial_block_id": item.spatial_block_id, "overlap_component_id": item.overlap_component_id, "dataset_split": item.dataset_split,
            }); count += 1
    return count


def main() -> int:
    args = parse_args(); config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")); root = Path(config["project_root"])
    process = root / config["process_directory"]; output = root / config["output_directory"]
    process.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True); logger = logger_for(process / "step114.log")
    try:
        summary = preflight(config, root, process, logger)
        if args.mode == "preflight":
            print(json.dumps(summary, ensure_ascii=False, indent=2)); return 0 if summary["status"] == "PREFLIGHT_PASS" else 2
        if not args.approved_by_user: raise RuntimeError("Run mode requires --approved-by-user")
        if summary["status"] != "PREFLIGHT_PASS": raise RuntimeError("Step114 preflight failed")
        zone_paths = [process / "zone_rasters" / f"outdoor_zone_group_{group}.tif" for group in range(int(config["zone_groups"]))]
        zones, counts = load_zones(zone_paths)
        reference_labels = pd.read_csv(root / config["reference_labels"]); shade_lookup = reference_labels.set_index(["point_id", "hour"])
        # Preserve the two formally approved Step10 boundary-zone candidate
        # overrides (points 34 and 21038) while retaining actual valid counts.
        approved_candidates = reference_labels.loc[
            reference_labels.hour == int(config["hours"][0]),
            ["point_id", "tmrt_total_candidate_pixels"],
        ]
        counts.update({int(row.point_id): int(row.tmrt_total_candidate_pixels) for _, row in approved_candidates.iterrows()})
        selected = pd.read_csv(root / config["selected_dates"]).date.astype(str).tolist(); parts_dir = process / "label_parts"; parts_dir.mkdir(exist_ok=True)
        reference_part = parts_dir / "labels_20240729.csv"
        if args.overwrite or not reference_part.exists(): reference_labels[LABEL_FIELDS].to_csv(reference_part, index=False, encoding="utf-8-sig")
        for date in tqdm(selected, desc="Multiweather label dates", unit="date", dynamic_ncols=True):
            part = parts_dir / f"labels_{date.replace('-', '')}.csv"
            if args.overwrite or not part.exists(): extract_date(config, root, date, zones, counts, shade_lookup, part)
        dates = sorted([config["geometry_reference_date"], *selected]); parts = [parts_dir / f"labels_{date.replace('-', '')}.csv" for date in dates]
        dynamic, dynamic_index = dynamic_table(config, root, output / "dynamic_conditions_multidate.csv")
        label_count = concatenate_labels(parts, output / "point_hour_labels_multidate.csv")
        manifest_count = build_manifest(config, root, output / "point_hour_labels_multidate.csv", dynamic_index, output / "training_manifest_multidate.csv")
        expected = int(config["expected_points"]) * len(dates) * len(config["hours"])
        final = {
            "status": "PASS" if label_count == manifest_count == expected else "FAIL", "date_count": len(dates),
            "weather_dates": dates, "point_count": len(counts), "hours_per_date": len(config["hours"]),
            "dynamic_rows": len(dynamic), "label_rows": label_count, "manifest_rows": manifest_count, "expected_rows": expected,
            "shade_geometry_reference_date": config["geometry_reference_date"], "solar_geometry_reference_date": config["geometry_reference_date"],
            "failed_files": 0,
        }
        write_json(output / "summary.json", final); (process / "failed_files.csv").write_text("point_id,filename,error_message\n", encoding="utf-8-sig")
        print(json.dumps(final, ensure_ascii=False, indent=2)); return 0 if final["status"] == "PASS" else 2
    except Exception as exc:
        logger.exception("Step114 failed")
        with (process / "failed_files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]); writer.writeheader(); writer.writerow({"point_id": "", "filename": str(CONFIG_PATH), "error_message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
