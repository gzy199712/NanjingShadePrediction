"""Auditable UTCI calculation and thermal-stress categorisation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
from tqdm import tqdm
from utci import rh_to_vp, utci_approx


@dataclass(frozen=True)
class UTCIInputQC:
    record_count: int
    valid_count: int
    invalid_count: int
    air_temperature_out_of_range: int
    tmrt_difference_out_of_range: int
    wind_speed_out_of_range: int
    humidity_out_of_range: int
    vapor_pressure_out_of_range: int
    nonfinite_input_count: int
    air_temperature_min: float
    air_temperature_max: float
    tmrt_minus_air_min: float
    tmrt_minus_air_max: float
    wind_speed_min: float
    wind_speed_max: float
    relative_humidity_min: float
    relative_humidity_max: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def implementation_metadata() -> dict[str, Any]:
    return {
        "package": "utci",
        "version": version("utci"),
        "function": "utci.utci_approx",
        "algorithm": "UTCI Version a 0.002 (October 2009), 6th-order polynomial",
        "source_url": "https://pypi.org/project/utci/",
        "reference": (
            "Bröde et al. (2012), Deriving the operational procedure for the "
            "Universal Thermal Climate Index (UTCI), "
            "doi:10.1007/s00484-011-0454-1"
        ),
        "wind_definition": "10 m wind speed in m/s",
        "valid_ranges": {
            "air_temperature_celsius": [-50.0, 50.0],
            "tmrt_minus_air_celsius": [-30.0, 70.0],
            "wind_speed_m_s": [0.5, 17.0],
            "relative_humidity_percent": [0.0, 100.0],
            "vapor_pressure_hpa": [0.0, 50.0],
        },
    }


def calculate_utci(
    air_temperature: np.ndarray,
    mean_radiant_temperature: np.ndarray,
    wind_speed: np.ndarray,
    relative_humidity: np.ndarray,
    *,
    description: str,
) -> tuple[np.ndarray, UTCIInputQC]:
    """Calculate UTCI without extrapolating beyond the documented domain."""
    ta, tmrt, wind, humidity = np.broadcast_arrays(
        np.asarray(air_temperature, dtype=np.float64),
        np.asarray(mean_radiant_temperature, dtype=np.float64),
        np.asarray(wind_speed, dtype=np.float64),
        np.asarray(relative_humidity, dtype=np.float64),
    )
    finite = (
        np.isfinite(ta)
        & np.isfinite(tmrt)
        & np.isfinite(wind)
        & np.isfinite(humidity)
    )
    vapor_pressure = np.full(ta.shape, np.nan, dtype=np.float64)
    finite_positions = np.flatnonzero(finite)
    for index in finite_positions:
        vapor_pressure.flat[index] = rh_to_vp(
            float(ta.flat[index]), float(humidity.flat[index])
        )
    difference = tmrt - ta
    ta_valid = (ta >= -50.0) & (ta <= 50.0)
    difference_valid = (difference >= -30.0) & (difference <= 70.0)
    wind_valid = (wind >= 0.5) & (wind <= 17.0)
    humidity_valid = (humidity >= 0.0) & (humidity <= 100.0)
    vapor_valid = (vapor_pressure >= 0.0) & (vapor_pressure <= 50.0)
    valid = (
        finite
        & ta_valid
        & difference_valid
        & wind_valid
        & humidity_valid
        & vapor_valid
    )
    output = np.full(ta.shape, np.nan, dtype=np.float64)
    valid_positions = np.flatnonzero(valid)
    for index in tqdm(
        valid_positions,
        desc=description,
        unit="record",
        dynamic_ncols=True,
    ):
        output.flat[index] = utci_approx(
            float(ta.flat[index]),
            float(vapor_pressure.flat[index]),
            float(tmrt.flat[index]),
            float(wind.flat[index]),
        )
    qc = UTCIInputQC(
        record_count=int(ta.size),
        valid_count=int(valid.sum()),
        invalid_count=int((~valid).sum()),
        air_temperature_out_of_range=int((finite & ~ta_valid).sum()),
        tmrt_difference_out_of_range=int((finite & ~difference_valid).sum()),
        wind_speed_out_of_range=int((finite & ~wind_valid).sum()),
        humidity_out_of_range=int((finite & ~humidity_valid).sum()),
        vapor_pressure_out_of_range=int((finite & ~vapor_valid).sum()),
        nonfinite_input_count=int((~finite).sum()),
        air_temperature_min=float(np.nanmin(ta)),
        air_temperature_max=float(np.nanmax(ta)),
        tmrt_minus_air_min=float(np.nanmin(difference)),
        tmrt_minus_air_max=float(np.nanmax(difference)),
        wind_speed_min=float(np.nanmin(wind)),
        wind_speed_max=float(np.nanmax(wind)),
        relative_humidity_min=float(np.nanmin(humidity)),
        relative_humidity_max=float(np.nanmax(humidity)),
    )
    return output, qc


def thermal_stress_category(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    labels = np.full(values.shape, "invalid", dtype=object)
    finite = np.isfinite(values)
    labels[finite & (values < -40.0)] = "extreme cold stress"
    labels[finite & (values >= -40.0) & (values < -27.0)] = (
        "very strong cold stress"
    )
    labels[finite & (values >= -27.0) & (values < -13.0)] = (
        "strong cold stress"
    )
    labels[finite & (values >= -13.0) & (values < 0.0)] = (
        "moderate cold stress"
    )
    labels[finite & (values >= 0.0) & (values < 9.0)] = "slight cold stress"
    labels[finite & (values >= 9.0) & (values < 26.0)] = "no thermal stress"
    labels[finite & (values >= 26.0) & (values < 32.0)] = (
        "moderate heat stress"
    )
    labels[finite & (values >= 32.0) & (values < 38.0)] = (
        "strong heat stress"
    )
    labels[finite & (values >= 38.0) & (values < 46.0)] = (
        "very strong heat stress"
    )
    labels[finite & (values >= 46.0)] = "extreme heat stress"
    return labels
