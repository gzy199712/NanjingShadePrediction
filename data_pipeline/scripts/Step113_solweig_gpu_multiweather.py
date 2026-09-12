"""Step113: preflight and run fixed-geometry multiweather SOLWEIG-GPU simulations."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
import torch
from tqdm import tqdm
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data_pipeline/configs/multiweather/solweig_gpu_multiweather.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--dates", nargs="*", help="Optional YYYY-MM-DD subset")
    parser.add_argument("--keep-intermediate", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step113")
    logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="a", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def raster_signature(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        return {
            "path": str(path), "width": src.width, "height": src.height,
            "count": src.count, "dtype": src.dtypes[0], "crs": str(src.crs),
            "transform": tuple(src.transform), "bounds": tuple(src.bounds),
        }


def validate_rasters(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    signatures = {name: raster_signature(path) for name, path in paths.items()}
    reference = signatures["building_dsm"]
    for name, item in signatures.items():
        for key in ("width", "height", "crs", "transform", "bounds"):
            if item[key] != reference[key]:
                raise RuntimeError(f"Raster alignment mismatch: {name}.{key}")
    return signatures


def umep_met_file(hourly: pd.DataFrame, weather_date: str, geometry_date: str, hours: list[int], path: Path) -> None:
    part = hourly.loc[(hourly["date"] == weather_date) & hourly["hour"].isin(hours)].sort_values("hour")
    if len(part) != len(hours):
        raise RuntimeError(f"{weather_date}: expected {len(hours)} local hours, found {len(part)}")
    reference = pd.Timestamp(geometry_date)
    day_of_year = int(reference.dayofyear)
    output = pd.DataFrame({
        "%iy": reference.year, "id": day_of_year, "it": part["hour"].astype(int), "imin": 0,
        "Q*": -999.0, "QH": -999.0, "QE": -999.0, "Qs": -999.0, "Qf": -999.0,
        "Wind": part["wind_speed"], "RH": part["relative_humidity"], "Td": part["air_temperature"],
        "press": part["air_pressure_kpa"], "rain": part["rainfall"],
        "Kdn": part["global_shortwave_radiation"], "snow": -999.0, "ldown": -999.0,
        "fcld": -999.0, "wuh": -999.0, "xsmd": -999.0, "lai_hr": -999.0,
        "Kdiff": part["diffuse_shortwave_radiation"], "Kdir": part["direct_shortwave_radiation"],
        "Wd": part["wind_direction"],
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, sep=" ", index=False, float_format="%.6f")


def merge_product(tile_root: Path, prefix: str, output_path: Path, compression: str, reference_path: Path) -> int:
    files = sorted(path for path in tile_root.rglob("*.tif") if path.name.upper().startswith(prefix.upper() + "_"))
    if not files:
        raise RuntimeError(f"No {prefix} tile outputs found under {tile_root}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    sources = [rasterio.open(path) for path in files]
    try:
        with rasterio.open(reference_path) as reference:
            reference_bounds = reference.bounds
            reference_res = reference.res
        merge(
            sources, bounds=reference_bounds, res=reference_res,
            method="first", target_aligned_pixels=False, mem_limit=256,
            dst_path=output_path,
            dst_kwds={
                "driver": "GTiff", "compress": compression, "tiled": True,
                "blockxsize": 512, "blockysize": 512, "BIGTIFF": "YES",
            },
        )
    finally:
        for source in sources:
            source.close()
    return len(files)


def valid_final(path: Path, reference_path: Path, bands: int) -> bool:
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as dst, rasterio.open(reference_path) as reference:
            return (
                dst.count == bands and dst.width == reference.width and dst.height == reference.height
                and dst.crs == reference.crs and dst.transform.almost_equals(reference.transform)
            )
    except rasterio.errors.RasterioError:
        return False


def static_cache_complete(cache: Path, expected_tiles: int = 25) -> bool:
    required = ["Building_DSM", "DEM", "Trees", "Landcover", "walls", "aspect"]
    return all(len(list((cache / name).glob("*.tif"))) == expected_tiles for name in required)


def prepare_static_cache(cache: Path, raster_paths: dict[str, Path], sample_met: Path, config: dict, logger: logging.Logger) -> None:
    if static_cache_complete(cache):
        logger.info("Reusing shared static SOLWEIG cache %s", cache)
        return
    if cache.exists():
        shutil.rmtree(cache)
    cache.mkdir(parents=True, exist_ok=True)
    from solweig_gpu.preprocessor import ppr
    import solweig_gpu.walls_aspect as walls_aspect

    logger.info("Creating shared static raster tiles")
    ppr(
        str(cache), str(raster_paths["building_dsm"]), str(raster_paths["dem"]),
        str(raster_paths["tree_dsm"]), str(raster_paths["landcover"]),
        int(config["tile_size"]), int(config["overlap"]), config["geometry_reference_date"],
        True, own_met_file=str(sample_met), preprocess_dir=str(cache),
    )
    requested_workers = int(config["static_preprocess_workers"])
    original_cpu_count = walls_aspect.os.cpu_count
    # On Windows the package uses cpu_count//2, so report twice the desired limit.
    walls_aspect.os.cpu_count = lambda: requested_workers * 2
    try:
        walls_aspect.run_parallel_processing(str(cache / "Building_DSM"), str(cache / "walls"), str(cache / "aspect"))
    finally:
        walls_aspect.os.cpu_count = original_cpu_count
    if not static_cache_complete(cache):
        raise RuntimeError("Shared static cache is incomplete")
    shutil.rmtree(cache / "metfiles", ignore_errors=True)
    (cache / "cache_manifest.json").write_text(json.dumps({
        "created_at": datetime.now().astimezone().isoformat(), "tile_count": 25,
        "tile_size": config["tile_size"], "overlap": config["overlap"],
        "worker_limit": requested_workers, "source_rasters": {key: str(value) for key, value in raster_paths.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def run_cached_date(cache: Path, work: Path, met_source: Path, geometry_date: str) -> tuple[int, int]:
    from solweig_gpu.utci_process import compute_utci, map_files_by_key

    met_dir = work / "metfiles"; output_dir = work / "output_folder"
    met_dir.mkdir(parents=True, exist_ok=True); output_dir.mkdir(parents=True, exist_ok=True)
    maps = {
        "building": map_files_by_key(str(cache / "Building_DSM"), ".tif"),
        "tree": map_files_by_key(str(cache / "Trees"), ".tif"),
        "dem": map_files_by_key(str(cache / "DEM"), ".tif"),
        "landcover": map_files_by_key(str(cache / "Landcover"), ".tif"),
        "walls": map_files_by_key(str(cache / "walls"), ".tif"),
        "aspect": map_files_by_key(str(cache / "aspect"), ".tif"),
    }
    keys = set.intersection(*(set(item) for item in maps.values()))
    if len(keys) != 25:
        raise RuntimeError(f"Static cache key mismatch: expected 25, got {len(keys)}")
    for key in keys:
        shutil.copyfile(met_source, met_dir / f"metfile_{key}.txt")
    ordered = sorted(keys, key=lambda value: tuple(int(part) for part in value.split("_")))
    for key in tqdm(ordered, desc="SOLWEIG cached GPU tiles", unit="tile", dynamic_ncols=True):
        key_dir = output_dir / key
        complete_tmrt = list(key_dir.glob("TMRT_*.tif"))
        complete_utci = list(key_dir.glob("UTCI_*.tif"))
        if complete_tmrt and complete_utci:
            continue
        key_dir.mkdir(parents=True, exist_ok=True)
        met_data = np.loadtxt(met_dir / f"metfile_{key}.txt", skiprows=1, delimiter=" ")
        compute_utci(
            maps["building"][key], maps["tree"][key], maps["dem"][key],
            maps["walls"][key], maps["aspect"][key], maps["landcover"][key],
            met_data, str(key_dir), key, geometry_date,
            save_tmrt=True, save_svf=False, save_kup=False, save_kdown=False,
            save_lup=False, save_ldown=False, save_shadow=False,
        )
        torch.cuda.empty_cache()
    return (
        len([path for path in output_dir.rglob("*.tif") if path.name.upper().startswith("TMRT_")]),
        len([path for path in output_dir.rglob("*.tif") if path.name.upper().startswith("UTCI_")]),
    )


def main() -> int:
    # SOLWEIG-GPU emits Unicode status symbols; force a safe Windows console.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    output_root = root / config["output_directory"]
    process_root = root / config["process_directory"]
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(process_root / "step113.log")
    failures: list[dict[str, str]] = []
    try:
        selected = pd.read_csv(root / config["selected_dates_csv"], dtype={"date": str})
        dates = selected["date"].tolist()
        if args.dates:
            unknown = sorted(set(args.dates) - set(dates))
            if unknown:
                raise RuntimeError(f"Requested dates not in Step112 selection: {unknown}")
            dates = args.dates
        hourly = pd.read_csv(root / config["hourly_weather_csv"], dtype={"date": str})
        raster_paths = {name: Path(path) for name, path in config["rasters"].items()}
        missing = [str(path) for path in raster_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing rasters: {missing}")
        signatures = validate_rasters(raster_paths)
        if bool(config["require_cuda"]) and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is prohibited")
        gpu = {
            "available": torch.cuda.is_available(),
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3) if torch.cuda.is_available() else 0,
        }
        height = int(signatures["building_dsm"]["height"]); width = int(signatures["building_dsm"]["width"])
        bands = len(config["local_hours"]); date_count = len(dates)
        final_bytes = height * width * bands * 4 * 2 * date_count
        estimate = {
            "date_count": date_count, "hours_per_date": bands,
            "raster_shape": [height, width], "estimated_uncompressed_final_gb": round(final_bytes / 1024**3, 2),
            "estimated_peak_working_gb": round(final_bytes / 1024**3 * 2.5, 2),
            "geometry_reference_date": config["geometry_reference_date"], "weather_dates": dates,
            "gpu": gpu, "raster_signatures": signatures,
            "method": "fixed_20240729_solar_geometry_with_date_specific_ERA5_meteorology",
        }
        process_root.mkdir(parents=True, exist_ok=True)
        (process_root / "preflight.json").write_text(json.dumps(estimate, ensure_ascii=False, indent=2), encoding="utf-8")
        met_root = process_root / "meteorology_fixed_geometry"
        for weather_date in tqdm(dates, desc="Preparing fixed-geometry meteorology", unit="date", dynamic_ncols=True):
            umep_met_file(hourly, weather_date, config["geometry_reference_date"], config["local_hours"], met_root / f"UMEP_{weather_date.replace('-', '')}_geometry_20240729.txt")
        if args.mode == "preflight":
            print(json.dumps({"status": "PREFLIGHT_PASS", **estimate}, ensure_ascii=False, indent=2))
            return 0
        if not args.approved_by_user:
            raise RuntimeError("Run mode requires --approved-by-user")
        static_cache = root / config["static_cache_directory"]
        sample_met = met_root / f"UMEP_{dates[0].replace('-', '')}_geometry_20240729.txt"
        prepare_static_cache(static_cache, raster_paths, sample_met, config, logger)
        records: list[dict[str, object]] = []
        for weather_date in tqdm(dates, desc="SOLWEIG-GPU weather dates", unit="date", dynamic_ncols=True):
            started = datetime.now().astimezone()
            work = process_root / "work" / weather_date.replace("-", "")
            final = output_root / weather_date.replace("-", "")
            try:
                tmrt_final = final / "Tmrt" / f"TMRT_{weather_date.replace('-', '')}_10m.tif"
                utci_final = final / "UTCI" / f"UTCI_{weather_date.replace('-', '')}_10m.tif"
                if valid_final(tmrt_final, raster_paths["building_dsm"], len(config["local_hours"])) and valid_final(utci_final, raster_paths["building_dsm"], len(config["local_hours"])):
                    logger.info("Reusing validated final products for %s", weather_date)
                    records.append({"date": weather_date, "success": True, "tmrt_tiles": 25, "utci_tiles": 25, "reused_final": True, "started_at": started.isoformat(), "ended_at": datetime.now().astimezone().isoformat()})
                    continue
                tile_output = work / "output_folder"
                existing_tmrt = [path for path in tile_output.rglob("*.tif") if path.name.upper().startswith("TMRT_")]
                existing_utci = [path for path in tile_output.rglob("*.tif") if path.name.upper().startswith("UTCI_")]
                if len(existing_tmrt) == 25 and len(existing_utci) == 25:
                    logger.info("Reusing 25 completed TMRT/UTCI tile pairs for %s", weather_date)
                else:
                    run_cached_date(static_cache, work, met_root / f"UMEP_{weather_date.replace('-', '')}_geometry_20240729.txt", config["geometry_reference_date"])
                tmrt_tiles = merge_product(work / "output_folder", "TMRT", tmrt_final, config["output_compression"], raster_paths["building_dsm"])
                utci_tiles = merge_product(work / "output_folder", "UTCI", utci_final, config["output_compression"], raster_paths["building_dsm"])
                if config["cleanup_intermediate_after_merge"] and not args.keep_intermediate:
                    shutil.rmtree(work)
                records.append({"date": weather_date, "success": True, "tmrt_tiles": tmrt_tiles, "utci_tiles": utci_tiles, "started_at": started.isoformat(), "ended_at": datetime.now().astimezone().isoformat()})
            except Exception as exc:
                logger.exception("Failed weather date %s", weather_date)
                failures.append({"point_id": weather_date, "filename": str(work), "error_message": str(exc)})
                records.append({"date": weather_date, "success": False, "error_message": str(exc), "started_at": started.isoformat(), "ended_at": datetime.now().astimezone().isoformat()})
        log_path = output_root / "simulation_log.csv"
        new_log = pd.DataFrame(records)
        if log_path.exists():
            old_log = pd.read_csv(log_path)
            preserved = []
            for _, row in old_log.iterrows():
                replacement = new_log.loc[new_log["date"].astype(str) == str(row["date"])]
                if replacement.empty or (bool(replacement.iloc[-1].get("reused_final", False)) and not bool(row.get("reused_final", False))):
                    preserved.append(row.to_dict())
            preserved_dates = {str(item["date"]) for item in preserved}
            new_log = pd.concat([pd.DataFrame(preserved), new_log.loc[~new_log["date"].astype(str).isin(preserved_dates)]], ignore_index=True)
        new_log.sort_values("date").to_csv(log_path, index=False, encoding="utf-8-sig")
        with (process_root / "failed_files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]); writer.writeheader(); writer.writerows(failures)
        status = "PASS" if not failures else "PARTIAL"
        print(json.dumps({"status": status, "success": len(records) - len(failures), "failed": len(failures)}, ensure_ascii=False, indent=2))
        return 0 if not failures else 2
    except Exception as exc:
        logger.exception("Step113 failed")
        process_root.mkdir(parents=True, exist_ok=True)
        with (process_root / "failed_files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]); writer.writeheader(); writer.writerow({"point_id": "", "filename": str(CONFIG_PATH), "error_message": str(exc)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
