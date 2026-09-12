"""Validated API request models for the local application."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Coordinate(StrictModel):
    crs: Literal["EPSG:4326", "EPSG:32650"]
    longitude: float | None = None
    latitude: float | None = None
    x: float | None = None
    y: float | None = None

    @model_validator(mode="after")
    def validate_coordinate(self) -> "Coordinate":
        if self.crs == "EPSG:4326":
            if self.longitude is None or self.latitude is None:
                raise ValueError("EPSG:4326 requires longitude and latitude")
            if not 118.0 <= self.longitude <= 119.5 or not 31.0 <= self.latitude <= 33.0:
                raise ValueError("Coordinate is outside the supported Nanjing extent")
        else:
            if self.x is None or self.y is None:
                raise ValueError("EPSG:32650 requires x and y")
            if not 500000 <= self.x <= 800000 or not 3400000 <= self.y <= 3700000:
                raise ValueError("Coordinate is outside the supported Nanjing extent")
        return self

    def tool_dict(self) -> dict[str, float | str]:
        if self.crs == "EPSG:4326":
            return {
                "crs": self.crs,
                "longitude": float(self.longitude),
                "latitude": float(self.latitude),
            }
        return {"crs": self.crs, "x": float(self.x), "y": float(self.y)}


class GeocodeRequest(StrictModel):
    query: str = Field(min_length=1, max_length=100)
    limit: int = Field(default=5, ge=1, le=10)


class SnapRequest(StrictModel):
    origin: Coordinate
    destination: Coordinate
    mode: Literal["walk", "bike", "shared"] = "walk"


class RouteRequest(StrictModel):
    origin: Coordinate
    destination: Coordinate
    hour: int = Field(default=14, ge=6, le=18)
    mode: Literal["walk", "bike", "shared"] = "walk"
    objective: Literal["shortest", "shade", "utci"] = "shortest"
    algorithm: Literal["astar", "dijkstra"] = "astar"
    max_detour_ratio: float | None = Field(default=0.20, ge=0, le=1)
    uncertainty_weight: float = Field(default=0.0, ge=0, le=3)

    def tool_dict(self) -> dict:
        return {
            "origin": self.origin.tool_dict(),
            "destination": self.destination.tool_dict(),
            "hour": self.hour,
            "mode": self.mode,
            "objective": self.objective,
            "algorithm": self.algorithm,
            "max_detour_ratio": self.max_detour_ratio,
            "uncertainty_weight": self.uncertainty_weight,
        }


class CompareRequest(RouteRequest):
    objectives: list[Literal["shortest", "shade", "utci"]] = Field(
        default_factory=lambda: ["shortest", "shade", "utci"],
        min_length=1,
    )

    def tool_dict(self) -> dict:
        value = super().tool_dict()
        value["objectives"] = self.objectives
        return value


class SummarizeRequest(StrictModel):
    route_id: str = Field(min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")
    language: Literal["zh-CN", "en"] = "zh-CN"


class ExportRequest(StrictModel):
    route_id: str = Field(min_length=8, max_length=64, pattern=r"^[a-f0-9]+$")
    format: Literal["geojson", "csv", "png", "pdf", "file_geodatabase"]


class VisualLayersRequest(StrictModel):
    bbox: list[float] = Field(min_length=4, max_length=4)
    hour: int = Field(ge=6, le=18)
    max_buildings: int = Field(default=3500, ge=100, le=5000)
    raster_max_dimension: int = Field(default=1200, ge=600, le=2000)
    include_vectors: bool = True

    @model_validator(mode="after")
    def validate_bbox(self) -> "VisualLayersRequest":
        min_x, min_y, max_x, max_y = self.bbox
        if not (min_x < max_x and min_y < max_y):
            raise ValueError("bbox must have positive width and height")
        if not (
            640000 <= min_x <= 700000
            and 640000 <= max_x <= 700000
            and 3520000 <= min_y <= 3580000
            and 3520000 <= max_y <= 3580000
        ):
            raise ValueError("bbox is outside the supported Nanjing extent")
        return self


class PlannerManualGenerationRequest(StrictModel):
    """Optional planner-edited tree mask and explicit feedback consent."""

    confidence: int = Field(default=65, ge=0, le=100)
    hour: int = Field(default=14, ge=6, le=18)
    manual_mask_png_base64: str = Field(min_length=32, max_length=8_000_000)
    feedback_consent: bool = False


class PlannerNoInterventionReviewRequest(StrictModel):
    """Planner decision that the current direction image needs no intervention."""

    reason: str = Field(min_length=5, max_length=500)
    feedback_consent: bool = True
    confidence: int = Field(default=65, ge=0, le=100)


class AssistantParseRequest(StrictModel):
    text: str = Field(min_length=2, max_length=500)


class AssistantRouteMetric(StrictModel):
    objective: Literal["shortest", "shade", "utci", "risk_aware"]
    distance_m: float = Field(ge=0)
    estimated_duration_min: float = Field(ge=0)
    detour_ratio: float = Field(ge=0, le=1)
    mean_shade: float = Field(ge=0, le=1)
    mean_tmrt: float
    mean_utci: float
    uncertainty_mean: float = Field(ge=0)
    reliability_score: float | None = Field(default=None, ge=0, le=100)
    warning_codes: list[str] = Field(default_factory=list, max_length=8)


class AssistantExplainRequest(StrictModel):
    routes: list[AssistantRouteMetric] = Field(min_length=1, max_length=4)
    question: str = Field(
        default="请解释这些路线的距离、遮荫、热舒适与不确定性权衡。",
        min_length=2,
        max_length=300,
    )
