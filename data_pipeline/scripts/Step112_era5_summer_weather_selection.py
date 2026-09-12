"""Step112: download ERA5 and select representative Nanjing summer weather dates."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cdsapi
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "data_pipeline/configs/multiweather/era5_summer_2024_selection.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step112"); logger.handlers.clear(); logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    return logger


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = frame.to_dict("records")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame.columns)); writer.writeheader()
        for row in tqdm(rows, desc=f"Writing {path.name}", unit="row", dynamic_ncols=True):
            writer.writerow(row)


def request_payload(config: dict[str, Any], variables: list[str]) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": ["2024"],
        "month": ["06", "07", "08"],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        "area": config["area_north_west_south_east"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def open_download(path: Path, extract_dir: Path) -> xr.Dataset:
    if zipfile.is_zipfile(path):
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith((".nc", ".netcdf"))]
            if not members:
                raise RuntimeError("ERA5 archive contains no NetCDF files")
            for member in tqdm(members, desc="Extracting ERA5 NetCDF", unit="file", dynamic_ncols=True):
                archive.extract(member, extract_dir)
        datasets = [xr.open_dataset(extract_dir / member) for member in members]
        return xr.merge(datasets, compat="override", join="outer")
    return xr.open_dataset(path)


def open_era5_pair(instant_path: Path, accum_path: Path, extract_dir: Path) -> xr.Dataset:
    """Open the two files required by SOLWEIG-GPU and align them by valid time."""
    instant = open_download(instant_path, extract_dir / "instant")
    accum = open_download(accum_path, extract_dir / "accum")
    return xr.merge([instant, accum], compat="override", join="inner")


def variable(dataset: xr.Dataset, names: list[str]) -> xr.DataArray:
    for name in names:
        if name in dataset.data_vars:
            return dataset[name]
    raise KeyError(f"Missing ERA5 variable; tried {names}; found {list(dataset.data_vars)}")


def saturation_vapour_pressure_hpa(temperature_c: np.ndarray) -> np.ndarray:
    return 6.112 * np.exp((17.67 * temperature_c) / (temperature_c + 243.5))


def dataframe_from_dataset(dataset: xr.Dataset, timezone: str) -> pd.DataFrame:
    time_name = "valid_time" if "valid_time" in dataset.coords else "time"
    spatial_dims = [name for name in ("latitude", "longitude") if name in dataset.dims]

    def values(names: list[str]) -> np.ndarray:
        data = variable(dataset, names)
        if spatial_dims:
            data = data.mean(dim=spatial_dims, skipna=True)
        extra = [dim for dim in data.dims if dim != time_name]
        if extra:
            data = data.mean(dim=extra, skipna=True)
        return np.asarray(data.values, dtype=np.float64).reshape(-1)

    utc = pd.to_datetime(dataset[time_name].values, utc=True)
    local = utc.tz_convert(timezone)
    temperature_c = values(["t2m", "2m_temperature"]) - 273.15
    dewpoint_c = values(["d2m", "2m_dewpoint_temperature"]) - 273.15
    u10 = values(["u10", "10m_u_component_of_wind"])
    v10 = values(["v10", "10m_v_component_of_wind"])
    pressure_kpa = values(["sp", "surface_pressure"]) / 1000.0
    precipitation_mm = values(["tp", "total_precipitation"]) * 1000.0
    global_shortwave = values(["ssrd", "surface_solar_radiation_downwards"]) / 3600.0
    direct_shortwave = values(["fdir", "total_sky_direct_solar_radiation_at_surface"]) / 3600.0
    diffuse_shortwave = np.maximum(global_shortwave - direct_shortwave, 0.0)
    cloud = np.clip(values(["tcc", "total_cloud_cover"]), 0.0, 1.0)
    relative_humidity = np.clip(
        100.0 * saturation_vapour_pressure_hpa(dewpoint_c) / saturation_vapour_pressure_hpa(temperature_c),
        0.0, 100.0,
    )
    wind_speed = np.sqrt(u10 * u10 + v10 * v10)
    wind_direction = (np.degrees(np.arctan2(-u10, -v10)) + 360.0) % 360.0
    frame = pd.DataFrame({
        "datetime_utc": utc, "datetime_local": local, "date": local.strftime("%Y-%m-%d"),
        "hour": local.hour, "air_temperature": temperature_c, "dewpoint_temperature": dewpoint_c,
        "relative_humidity": relative_humidity, "wind_speed": wind_speed, "wind_direction": wind_direction,
        "air_pressure_kpa": pressure_kpa, "rainfall": precipitation_mm,
        "global_shortwave_radiation": np.maximum(global_shortwave, 0.0),
        "direct_shortwave_radiation": np.maximum(direct_shortwave, 0.0),
        "diffuse_shortwave_radiation": diffuse_shortwave, "total_cloud_cover": cloud,
    })
    return frame.sort_values("datetime_local").drop_duplicates("datetime_local").reset_index(drop=True)


def daily_metrics(hourly: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config["selection"]
    start = pd.Timestamp(config["target_period"]["start_local_date"])
    end = pd.Timestamp(config["target_period"]["end_local_date"])
    frame = hourly.loc[pd.to_datetime(hourly.date).between(start, end)].copy()
    frame = frame.loc[frame.hour.isin(selection["evaluation_hours_local"])]
    rows: list[dict[str, Any]] = []
    for date, part in tqdm(frame.groupby("date"), desc="Summarising ERA5 days", unit="day", dynamic_ncols=True):
        if len(part) != len(selection["evaluation_hours_local"]):
            # Manual CDS downloads commonly start at 00:00 UTC, leaving the
            # first local day incomplete in UTC+8. Never rank a partial day.
            continue
        local_date = pd.Timestamp(date)
        rows.append({
            "date": date, "iso_year": int(local_date.isocalendar().year), "iso_week": int(local_date.isocalendar().week),
            "day_count_hours": len(part), "maximum_temperature_c": float(part.air_temperature.max()),
            "mean_temperature_c": float(part.air_temperature.mean()), "minimum_temperature_c": float(part.air_temperature.min()),
            "mean_relative_humidity_percent": float(part.relative_humidity.mean()),
            "mean_wind_speed_m_s": float(part.wind_speed.mean()), "maximum_wind_speed_m_s": float(part.wind_speed.max()),
            "mean_global_shortwave_w_m2": float(part.global_shortwave_radiation.mean()),
            "maximum_global_shortwave_w_m2": float(part.global_shortwave_radiation.max()),
            "shortwave_energy_mj_m2": float(part.global_shortwave_radiation.sum() * 3600.0 / 1e6),
            "precipitation_mm": float(part.rainfall.sum()), "mean_cloud_fraction": float(part.total_cloud_cover.mean()),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No complete local dates contain all 06:00-18:00 hours")
    thresholds = selection["weather_thresholds"]
    regimes = []
    for _, row in tqdm(result.iterrows(), total=len(result), desc="Classifying weather regimes", unit="day", dynamic_ncols=True):
        if row.precipitation_mm >= thresholds["rainy_daily_precipitation_mm"]:
            regime = "hot_rainy"
        elif row.mean_wind_speed_m_s >= thresholds["windy_mean_wind_speed_m_s"]:
            regime = "hot_windy"
        elif row.mean_cloud_fraction >= thresholds["cloudy_mean_cloud_fraction"]:
            regime = "hot_cloudy"
        elif row.mean_relative_humidity_percent >= thresholds["humid_mean_relative_humidity_percent"]:
            regime = "hot_humid"
        elif row.mean_cloud_fraction <= thresholds["clear_mean_cloud_fraction"]:
            regime = "hot_clear"
        else:
            regime = "hot_mixed"
        regimes.append(regime)
    result["weather_regime"] = regimes
    hot_threshold = float(selection["absolute_hot_day_max_temperature_c"])
    result["temperature_pass"] = result.maximum_temperature_c >= hot_threshold
    result["solar_max_pass"] = result.maximum_global_shortwave_w_m2 >= float(selection["minimum_maximum_global_shortwave_w_m2"])
    result["solar_mean_pass"] = result.mean_global_shortwave_w_m2 >= float(selection["minimum_mean_global_shortwave_w_m2"])
    result["rain_pass"] = result.precipitation_mm <= float(selection["maximum_daily_precipitation_mm"])
    result["cloud_pass"] = result.mean_cloud_fraction <= float(selection["maximum_mean_cloud_fraction"])
    result["eligible_clear_hot"] = result[[
        "temperature_pass", "solar_max_pass", "solar_mean_pass", "rain_pass", "cloud_pass"
    ]].all(axis=1)
    for column in ["maximum_temperature_c", "mean_temperature_c", "shortwave_energy_mj_m2", "mean_relative_humidity_percent"]:
        std = float(result[column].std(ddof=0)) or 1.0
        result[f"z_{column}"] = (result[column] - result[column].mean()) / std
    result["heat_score"] = (
        0.45 * result.z_maximum_temperature_c + 0.25 * result.z_mean_temperature_c
        + 0.20 * result.z_shortwave_energy_mj_m2 + 0.10 * result.z_mean_relative_humidity_percent
    )
    result.attrs["hot_threshold_c"] = hot_threshold
    return result


def select_weekly(daily: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config["selection"]; selected_rows: list[pd.Series] = []
    diversity_features = ["maximum_temperature_c", "mean_relative_humidity_percent", "mean_wind_speed_m_s"]
    for (_, week), week_frame in tqdm(daily.groupby(["iso_year", "iso_week"]), desc="Selecting weekly weather dates", unit="week", dynamic_ncols=True):
        eligible = week_frame.loc[week_frame.eligible_clear_hot].sort_values("heat_score", ascending=False)
        fallback = eligible.empty
        candidates = eligible if not fallback else week_frame.sort_values("heat_score", ascending=False)
        chosen = [candidates.iloc[0]]
        # A second date is retained only when it is also clear/hot, at least two
        # days away, and materially expands humidity/wind/temperature coverage.
        if not fallback and len(eligible) > 1 and int(selection["weekly_maximum_dates"]) >= 2:
            anchor_date = pd.Timestamp(chosen[0].date)
            remaining = eligible.loc[
                eligible.date.map(lambda value: abs((pd.Timestamp(value) - anchor_date).days))
                >= int(selection["minimum_separation_days"])
            ].copy()
            if not remaining.empty:
                matrix = week_frame[diversity_features].to_numpy(float)
                scale = np.nanstd(matrix, axis=0); scale[scale < 1e-8] = 1.0
                anchor = chosen[0][diversity_features].to_numpy(float)
                distances = np.linalg.norm((remaining[diversity_features].to_numpy(float) - anchor) / scale, axis=1)
                best = int(np.argmax(distances))
                if float(distances[best]) >= float(selection["second_date_minimum_score_gain"]):
                    chosen.append(remaining.iloc[best])
        for rank, row in enumerate(chosen, start=1):
            row = row.copy(); row["weekly_selection_rank"] = rank
            row["selection_reason"] = "weekly_clear_hot_anchor" if rank == 1 and not fallback else "weekly_best_available_fallback" if rank == 1 else "clear_hot_weather_diversity"
            selected_rows.append(row)
    result = pd.DataFrame(selected_rows).sort_values("date").reset_index(drop=True)
    return result[[column for column in result.columns if not column.startswith("z_")]]


def solweig_csv(hourly: pd.DataFrame, date: str, target: Path) -> None:
    part = hourly.loc[hourly.date.eq(date)].sort_values("hour")
    if len(part) != 24:
        raise RuntimeError(f"{date}: expected 24 local hourly rows, got {len(part)}")
    output = pd.DataFrame({
        "Year": pd.to_datetime(part.date).dt.year, "Month": pd.to_datetime(part.date).dt.month,
        "Day": pd.to_datetime(part.date).dt.day, "Hour": part.hour, "Minute": 0,
        "Wind speed": part.wind_speed, "Wind direction": part.wind_direction,
        "Air trmperature": part.air_temperature, "Relative Humidity": part.relative_humidity,
        "Barometric pressure (kpa)": part.air_pressure_kpa, "Rainfall": part.rainfall,
        "Incoming shortwave radiation": part.global_shortwave_radiation,
        "Direct shortwave radiation (W/m2)": part.direct_shortwave_radiation,
        "Diffuse shortwave radiation": part.diffuse_shortwave_radiation,
    })
    write_csv(target, output)


def main() -> int:
    args = parse_args(); config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")); root = Path(config["project_root"])
    raw = root / config["raw_directory"]; curated = root / config["curated_directory"]; process = root / config["process_directory"]
    raw.mkdir(parents=True, exist_ok=True); curated.mkdir(parents=True, exist_ok=True); logger = setup_logger(process / "step112.log")
    started = datetime.now().astimezone(); failures: list[dict[str, str]] = []
    try:
        credential = Path.home() / ".cdsapirc"
        instant_target = raw / config["instant_download_target"]
        accum_target = raw / config["accum_download_target"]
        checks = {
            "cdsapirc_exists": credential.exists(), "reference_csv_exists": (root / config["reference_meteorology_csv"]).exists(),
            "target_period_days": 62, "download_requested": args.mode == "run",
            "instant_target": str(instant_target), "accum_target": str(accum_target),
            "instant_variables": config["instant_variables"], "accum_variables": config["accum_variables"],
        }
        if not checks["cdsapirc_exists"]: raise RuntimeError("Missing ~/.cdsapirc")
        (process / "preflight.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.mode == "preflight":
            print(json.dumps({"status": "PREFLIGHT_PASS", **checks}, ensure_ascii=False, indent=2)); return 0
        if not args.approved_by_user: raise RuntimeError("Run mode requires --approved-by-user")
        client = None
        request_specs = [
            ("instant", config["instant_variables"], instant_target),
            ("accum", config["accum_variables"], accum_target),
        ]
        for label, variables, target in tqdm(request_specs, desc="Downloading ERA5 groups", unit="group", dynamic_ncols=True):
            payload = request_payload(config, variables)
            (process / f"era5_request_{label}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            if target.exists() and not (args.overwrite or config["overwrite_download"]):
                logger.info("Reusing existing ERA5 %s file %s", label, target)
            else:
                if client is None:
                    client = cdsapi.Client()
                logger.info("Submitting ERA5 %s request to %s", label, config["source_dataset"])
                client.retrieve(config["source_dataset"], payload, str(target))
        dataset = open_era5_pair(instant_target, accum_target, raw / "extracted")
        hourly = dataframe_from_dataset(dataset, config["target_timezone"])
        start = config["target_period"]["start_local_date"]; end = config["target_period"]["end_local_date"]
        hourly = hourly.loc[pd.to_datetime(hourly.date).between(pd.Timestamp(start), pd.Timestamp(end))].reset_index(drop=True)
        write_csv(curated / "era5_hourly_nanjing_202407_202408.csv", hourly)
        daily = daily_metrics(hourly, config); write_csv(curated / "era5_daily_weather_metrics.csv", daily)
        selected = select_weekly(daily, config); write_csv(curated / "selected_representative_weather_dates.csv", selected)
        met_root = curated / "solweig_meteorology"
        for date in tqdm(selected.date.astype(str), desc="Writing SOLWEIG meteorology", unit="date", dynamic_ncols=True):
            solweig_csv(hourly, date, met_root / date.replace("-", "") / f"Rawdata{date.replace('-', '')}.csv")
        weekly_counts = selected.groupby(["iso_year", "iso_week"]).size()
        status = "PASS" if weekly_counts.between(config["selection"]["weekly_minimum_dates"], config["selection"]["weekly_maximum_dates"]).all() else "FAIL"
        summary = {
            "status": status, "started_at": started.isoformat(), "ended_at": datetime.now().astimezone().isoformat(),
            "era5_dataset": config["source_dataset"], "era5_doi": config["source_doi"],
            "hourly_rows": len(hourly), "daily_rows": len(daily), "selected_date_count": len(selected),
            "week_count": int(len(weekly_counts)), "weekly_selected_min": int(weekly_counts.min()), "weekly_selected_max": int(weekly_counts.max()),
            "weather_regime_counts": selected.weather_regime.value_counts().to_dict(),
            "strict_clear_hot_dates": int(selected.eligible_clear_hot.sum()),
            "weekly_fallback_dates": int((selected.selection_reason == "weekly_best_available_fallback").sum()),
            "hot_threshold_c": float(daily.attrs["hot_threshold_c"]),
            "solweig_labels_generated": False, "multiweather_training_authorized": False,
        }
        (curated / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (process / "failed_files.csv").write_text("point_id,filename,error_message\n", encoding="utf-8-sig")
        logger.info("completed %s", json.dumps(summary, ensure_ascii=False)); print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if status == "PASS" else 2
    except Exception as exc:
        logger.exception("Step112 failed"); failures.append({"point_id": "", "filename": str(CONFIG_PATH), "error_message": str(exc)})
        with (process / "failed_files.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"]); writer.writeheader(); writer.writerows(failures)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
