"""FastAPI application for the offline Nanjing thermal-route planner."""

from __future__ import annotations

import csv
import base64
import gzip
import io
import json
import logging
import math
import statistics
import shutil
import subprocess
import threading
import uuid
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from routing.app.backend.api_models import (
    AssistantExplainRequest,
    AssistantParseRequest,
    CompareRequest,
    ExportRequest,
    GeocodeRequest,
    RouteRequest,
    SnapRequest,
    SummarizeRequest,
    VisualLayersRequest,
    PlannerManualGenerationRequest,
    PlannerNoInterventionReviewRequest,
)
from routing.app.backend.error_handlers import install_error_handlers
from routing.app.backend.llm_adapter import (
    explain_route_tradeoffs,
    parse_travel_request,
    status as llm_status,
)
from routing.app.backend.privacy import privacy_middleware, privacy_status
from routing.app.backend.service_adapter import ServiceAdapter
from routing.app.backend.planner_analysis_adapter import PlannerAnalysisAdapter
from routing.app.backend.planner_generation_agent_adapter import PlannerGenerationAgentAdapter
from routing.app.backend.planner_agent_orchestrator import PlannerAgentRuntime, build_assessment_agent_result
from routing.app.backend.planner_agent_tools import PlannerAgentTool, PlannerToolRegistry, tool_result
from routing.app.backend.multidate_contract import (
    model_status as multidate_model_status,
    scenarios as multidate_scenarios,
)


ROOT = Path(__file__).resolve().parents[3]
FRONTEND = ROOT / "routing" / "app" / "frontend"
LOGGER = logging.getLogger("thermal_route_app")
ADAPTER = ServiceAdapter()
PLANNER_ADAPTER = PlannerAnalysisAdapter()
PLANNER_GENERATION_AGENT = PlannerGenerationAgentAdapter()
PLANNER_REALISTIC_GENERATION_LOCK = threading.Lock()
PLANNER_FEEDBACK_LOCK = threading.Lock()
STARTED_AT = time.time()
PLANNER_OUTPUT = ROOT / "outputs/planner_streetscape_decision_package"
PLANNER_PERFORMANCE = ROOT / "outputs/planner_citywide_performance"
# The web application only uses accepted ghost-reduced panoramas.
PLANNER_PANORAMAS = ROOT / "streetscape/data/panorama/stage_04_formal/images"
PLANNER_DIRECTIONS = ROOT / "data/raw/NanjingStreetViewImages_12Directions_pitch0"
PLANNER_PACKAGE = ROOT / "streetscape/data/planning/stage_13_planner_handoff_package"
PLANNER_DUAL_LAYER = ROOT / "streetscape/data/planning/stage_18_planner_dual_layer_handoff/planner_dual_layer_cases.json"
PLANNER_TERMINAL = ROOT / "streetscape/data/planning/stage_22_planner_terminal_web_handoff/v2_terminal_cases.json"
PLANNER_VALIDATION_FIGURE = ROOT / "streetscape/figures/stage_24_terminal_benchmark/stage_24_terminal_benchmark_figure.png"
PLANNER_VALIDATION_PDF = ROOT / "streetscape/figures/stage_24_terminal_benchmark/stage_24_terminal_benchmark_figure.pdf"
PLANNER_VALIDATION_SOURCE = ROOT / "streetscape/figures/stage_24_terminal_benchmark/source_data.csv"
PLANNER_CITYWIDE_FIGURE_DIR = ROOT / "streetscape/figures/stage_33_citywide_terminal_planning"
PLANNER_CITYWIDE_FIGURE = PLANNER_CITYWIDE_FIGURE_DIR / "stage_33_citywide_terminal_planning.png"
PLANNER_CITYWIDE_PDF = PLANNER_CITYWIDE_FIGURE_DIR / "stage_33_citywide_terminal_planning.pdf"
PLANNER_CITYWIDE_SOURCE = PLANNER_CITYWIDE_FIGURE_DIR / "source_data_8975.csv.gz"
PLANNER_CITYWIDE_ELIGIBLE = PLANNER_CITYWIDE_FIGURE_DIR / "eligible_points_5.csv"
PLANNER_CITYWIDE_SUMMARY = PLANNER_CITYWIDE_FIGURE_DIR / "summary.json"
PLANNER_FIGURES = ROOT / "streetscape/figures/stage_12_panorama_sector_evidence_cards"
PLANNER_PANORAMAS = ROOT / "streetscape/data/panorama/stage_04_formal/images"
PLANNER_CITYWIDE = ROOT / "streetscape/data/planning/stage_32_citywide_final_decisions/planner_web_decisions.csv.gz"
PLANNER_FINAL_DECISIONS = ROOT / "streetscape/data/planning/stage_32_citywide_final_decisions/citywide_final_planning_decisions.csv.gz"
UPLOAD_RUNTIME = ROOT / "routing/app/runtime_uploads"
PLANNER_FEEDBACK = ROOT / "data/training/planner_human_feedback"
LEAF_OFF_MONTHS = {11, 12, 1, 2}
SVF_GRADE_LIMITS = {"good_max": 0.15, "fair_max": 0.25, "poor_max": 0.35}
EXPECTED_HEADINGS = tuple(range(0, 360, 30))
PLANNER_POINT_EDGE_MAPPING = ROOT / "routing/data/point_edge_mapping/point_edge_mapping.csv"
PLANNER_ROAD_CLASSIFICATION = ROOT / "routing/data/candidate_network/road_thermal_classification.csv"
IMAGE2_TEACHER_BENCHMARK = ROOT / "streetscape/data/generation/stage_50_image2_benchmark"
PLANNER_READY_IMAGE2_PACKAGE = ROOT / "streetscape/data/generation/stage_69_planner_ready_generation_package/planner_ready_manifest.csv"


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


@lru_cache(maxsize=64)
def _image2_teacher_reference(point_id: str, heading: int) -> dict[str, Any] | None:
    """Return a benchmark teacher image only for a frozen matching view."""
    packaged = _planner_ready_image2_row(point_id, heading)
    if packaged:
        image_path = Path(packaged["image2_reference"])
        if image_path.is_file():
            return {
                "model": "Image2 teacher reference",
                "quality_tier": "external_teacher_reference",
                "image_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
                "benchmark_id": packaged["pair_id"],
                "source": "human_imported_external_image",
                "external_request_made": False,
                "claim_boundary": "人工外部生成参考图；不是现场实测，也不是因果热效益验证。",
            }
    if not IMAGE2_TEACHER_BENCHMARK.is_dir():
        return None
    for metadata_path in IMAGE2_TEACHER_BENCHMARK.glob("shade_edit_*/analysis_summary.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("point_id")) != str(point_id) or int(metadata.get("heading", -1)) != int(heading):
            continue
        image_path = metadata_path.parent / "gpt-image-2.png"
        if not image_path.is_file():
            return None
        return {
            "model": "Image 2 teacher reference",
            "quality_tier": "external_teacher_reference",
            "image_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            "benchmark_id": metadata.get("benchmark_id"),
            "source": "human_imported_external_image",
            "external_request_made": False,
            "claim_boundary": "人工外部生成参考图；不是现场实测，也不是因果热效益验证。",
        }
    return None


@lru_cache(maxsize=1)
def _planner_ready_image2_index() -> dict[tuple[str, int], dict[str, str]]:
    if not PLANNER_READY_IMAGE2_PACKAGE.is_file():
        return {}
    with PLANNER_READY_IMAGE2_PACKAGE.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            (str(row["point_id"]), int(float(row["heading"]))): row
            for row in csv.DictReader(stream)
        }


def _planner_ready_image2_row(point_id: str, heading: int) -> dict[str, str] | None:
    return _planner_ready_image2_index().get((str(point_id), int(heading)))


def _planner_ready_headings(point_id: str) -> list[int]:
    return sorted(heading for pid, heading in _planner_ready_image2_index() if pid == str(point_id))


def _planner_ready_teacher_generation(point_id: str, heading: int, hour: int) -> dict[str, Any] | None:
    row = _planner_ready_image2_row(point_id, heading)
    if not row:
        return None
    teacher = Path(row["image2_reference"])
    guide = Path(row["planner_guidance_overlay"])
    card = Path(row["planner_generation_card"])
    if not teacher.is_file():
        return None
    image_payload = base64.b64encode(teacher.read_bytes()).decode("ascii")
    guide_payload = base64.b64encode(guide.read_bytes()).decode("ascii") if guide.is_file() else None
    result = {
        "generation_available": True,
        "automatic_acceptance": True,
        "planner_review_required": False,
        "agent_state": "IMAGE2_TEACHER_REFERENCE_AVAILABLE",
        "agent_explanation": "该南京点位命中已导入并完成本地验收的外部教师样本；本次不发起外部请求。",
        "realistic_tree_png_base64": image_payload,
        "overlay_png_base64": guide_payload,
        "selected_seed": row["pair_id"],
        "generator": "Image2 teacher reference",
        "scenario_qc": {
            "quality_tier": "external_teacher_reference",
            "compact_mask_ratio": float(row.get("compact_mask_ratio") or 0),
            "compact_precision_to_teacher": float(row.get("compact_precision_to_teacher") or 0),
            "compact_recall_to_teacher": float(row.get("compact_recall_to_teacher") or 0),
        },
        "model_alternatives": [{
            "model": "Image2 teacher reference",
            "quality_tier": "external_teacher_reference",
            "automatic_acceptance": True,
            "score": 1.0,
            "image_png_base64": image_payload,
        }],
        "model_comparison": [],
        "point_id": point_id,
        "heading": heading,
        "hour": hour,
        "teacher_pair_id": row["pair_id"],
        "planner_generation_card": str(card) if card.is_file() else None,
        "material_source_priority": "Image2_teacher_reference_from_500_pair_distillation_package",
        "full_gpu_verified": False,
        "cpu_fallback": False,
        "external_request_made": False,
        "source": "human_imported_external_image",
        "claim_boundary": "人工外部生成参考图；不是现场实测，也不是因果热效益验证。",
        "feedback_saved": False,
        "feedback_usage": "session_only_not_persisted",
    }
    return attach_reference_thermal_estimate(
        {
            **result,
            "shade_optimization_need": "需要评估",
            "suggestion_regions": [{"type": "canopy_target", "pixel_ratio": float(row.get("compact_mask_ratio") or 0)}],
        },
        hour,
        point_id,
    )


@lru_cache(maxsize=1)
def planner_osm_road_contexts() -> dict[str, dict[str, Any]]:
    edge_attributes: dict[str, dict[str, Any]] = {}
    with PLANNER_ROAD_CLASSIFICATION.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            edge_attributes[str(row["edge_id"])] = row
    contexts: dict[str, dict[str, Any]] = {}
    with PLANNER_POINT_EDGE_MAPPING.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if not _csv_bool(row.get("formal_match", True)):
                continue
            edge_id = str(row["edge_id"])
            edge = edge_attributes.get(edge_id, {})
            fclass = str(edge.get("fclass") or row.get("fclass") or "").lower()
            motor_only = _csv_bool(edge.get("motor_vehicle_only")) or str(row.get("mode_network", "")) == "motor_vehicle_only"
            active_classes = {"footway", "cycleway", "pedestrian", "path", "steps", "living_street", "track"}
            contexts[str(row["point_id"])] = {
                "source": "formal_osm_point_edge_mapping",
                "edge_id": edge_id,
                "fclass": fclass,
                "road_name": str(edge.get("name") or ""),
                "maxspeed": str(edge.get("maxspeed") or ""),
                "oneway": str(edge.get("oneway") or ""),
                "bridge": _csv_bool(edge.get("bridge") or row.get("bridge")),
                "tunnel": _csv_bool(edge.get("tunnel") or row.get("tunnel")),
                "grade_separated": _csv_bool(row.get("grade_separated")),
                "motor_vehicle_only": motor_only,
                "thermal_comfort_relevant": _csv_bool(edge.get("thermal_comfort_relevant")) or str(row.get("mode_network", "")) == "thermal_comfort_relevant",
                "active_mobility_class": fclass in active_classes,
                "classification_basis": str(edge.get("classification_basis") or ""),
                "match_confidence": str(row.get("match_confidence") or "unknown"),
                "distance_m": float(row.get("distance_m") or 0),
            }
    return contexts


def planner_osm_road_context(point_id: str) -> dict[str, Any]:
    return planner_osm_road_contexts().get(point_id, {
        "source": "no_formal_osm_match", "fclass": "unknown",
        "motor_vehicle_only": False, "thermal_comfort_relevant": False,
        "active_mobility_class": False, "match_confidence": "none",
    })


@lru_cache(maxsize=1)
def planner_point_metadata() -> dict[str, dict[str, Any]]:
    source = ROOT / "data/training_data/static/point_static_metadata_8975.csv"
    result: dict[str, dict[str, Any]] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            result[str(row["point_id"])] = {
                "year": int(row["year"]), "month": int(row["month"]),
                "longitude": float(row["x"]), "latitude": float(row["y"]),
            }
    return result


@lru_cache(maxsize=2048)
def planner_direction_records(point_id: str) -> tuple[dict[str, Any], ...]:
    if not point_id.isdigit():
        raise ValueError("Invalid point_id")
    records: list[dict[str, Any]] = []
    for path in PLANNER_DIRECTIONS.glob(f"{point_id}_*.png"):
        parts = path.stem.split("_")
        if len(parts) != 8:
            continue
        heading = int(float(parts[3]))
        if heading not in EXPECTED_HEADINGS:
            continue
        records.append({
            "point_id": point_id, "heading": heading, "image_path": str(path.resolve()),
            "filename": path.name, "longitude": float(parts[1]), "latitude": float(parts[2]),
            "capture_date": parts[5], "month": int(parts[5][4:6]),
        })
    records.sort(key=lambda item: item["heading"])
    if len(records) != 12 or tuple(item["heading"] for item in records) != EXPECTED_HEADINGS:
        raise ValueError(f"Point {point_id} does not have a complete 12-direction source set")
    return tuple(records)


@lru_cache(maxsize=24)
def planner_reference_weather(hour: int) -> dict[str, Any]:
    source = ROOT / "data/training_data/multiweather/step112_era5_weather_selection/solweig_meteorology/20240804/Rawdata20240804.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        row = next(item for item in csv.DictReader(stream) if int(item["Hour"]) == hour)
    return {
        "scenario_date": "2024-08-04", "hour": hour,
        "daily_maximum_air_temperature_c": 38.3856,
        "air_temperature_c": round(float(row["Air trmperature"]), 2),
        "relative_humidity_percent": round(float(row["Relative Humidity"]), 2),
        "wind_speed_m_s": round(float(row["Wind speed"]), 2),
        "incoming_shortwave_radiation_w_m2": round(float(row["Incoming shortwave radiation"]), 2),
        "weather_source": "ERA5 Nanjing center representative clear-hot day",
    }


def attach_reference_thermal_estimate(
    result: dict[str, Any], hour: int = 14, known_point_id: str | None = None,
    longitude: float | None = None, latitude: float | None = None,
) -> dict[str, Any]:
    points = planner_points_data(hour)["points"]
    by_id = {str(item["point_id"]): item for item in points}
    reference = result.get("self_supervised_reference") or {}
    if known_point_id:
        reference_id = known_point_id
    elif longitude is not None and latitude is not None:
        reference_id = min(
            planner_point_metadata(),
            key=lambda point_id: (planner_point_metadata()[point_id]["longitude"] - longitude) ** 2 + (planner_point_metadata()[point_id]["latitude"] - latitude) ** 2,
        )
    else:
        reference_id = str(reference.get("nearest_morphology_point_id", ""))
    point = by_id.get(reference_id)
    if point is None:
        point = points[0]; reference_id = str(point["point_id"])
    perf = point.get("performance_estimate") or {}
    eligible = [
        item.get("performance_estimate") or {} for item in points
        if (item.get("performance_estimate") or {}).get("status") == "estimated_standardized_intervention"
    ]
    need = str(result.get("shade_optimization_need", ""))
    if need == "通常不需要":
        shade_gain = delta_tmrt = delta_utci = 0.0
        benefit_source = "directional_no_intervention_gate"
    elif perf.get("status") == "estimated_standardized_intervention":
        shade_gain = float(perf["effective_shade_gain"])
        delta_tmrt = float(perf["estimated_delta_tmrt_c"])
        delta_utci = float(perf["estimated_delta_utci_c"])
        benefit_source = "matched_point_model_response" if known_point_id else "nearest_morphology_point_model_response"
    elif not eligible:
        return {
            **result,
            "thermal_reference_estimate": None,
            "quantified_thermal_benefit_available": False,
        }
    else:
        shade_gain = statistics.median(float(item["effective_shade_gain"]) for item in eligible)
        delta_tmrt = statistics.median(float(item["estimated_delta_tmrt_c"]) for item in eligible)
        delta_utci = statistics.median(float(item["estimated_delta_utci_c"]) for item in eligible)
        benefit_source = "citywide_eligible_intervention_median_response"
    canopy_ratio = sum(
        float(item.get("pixel_ratio", 0)) for item in result.get("suggestion_regions", [])
        if item.get("type") in {"canopy_target", "facility_review"}
    )
    scale = 0.0 if need == "通常不需要" else max(.45, min(1.0, canopy_ratio / .08 if canopy_ratio else .45))
    shade_gain *= scale; delta_tmrt *= scale; delta_utci *= scale
    weather = planner_reference_weather(hour)
    estimate = {
        "estimate_type": "nanjing_standardized_intervention_response" if known_point_id else "nanjing_dinov2_morphology_transfer_reference",
        "reference_point_id": reference_id,
        "reference_similarity": reference.get("nearest_morphology_similarity"),
        "scenario_weather": weather,
        "reference_baseline_tmrt_c": round(float(point["tmrt"]), 2),
        "reference_baseline_utci_c": round(float(point["utci"]), 2),
        "estimated_shade_gain": round(shade_gain, 4),
        "estimated_delta_tmrt_c": round(delta_tmrt, 2),
        "estimated_delta_utci_c": round(delta_utci, 2),
        "estimated_after_tmrt_c": round(float(point["tmrt"]) + delta_tmrt, 2),
        "estimated_after_utci_c": round(float(point["utci"]) + delta_utci, 2),
        "benefit_source": benefit_source,
        "spatial_status": "known_nanjing_point" if known_point_id else ("user_coordinate_nearest_nanjing_point" if longitude is not None and latitude is not None else "unknown_upload_transferred_from_similar_nanjing_morphology"),
        "claim_boundary": (
            "训练模型对标准化遮阴干预的预评估，不是生成图像的实测或因果验证。"
            if known_point_id
            else "相似城市形态与典型晴热天气下的参考估计，不是上传地点的实测或坐标级模拟。"
        ),
    }
    return {**result, "thermal_reference_estimate": estimate, "quantified_thermal_benefit_available": True}


@lru_cache(maxsize=1)
def planner_visual_review() -> dict[str, dict[str, str]]:
    source = PLANNER_OUTPUT / "visual_review.csv"
    if not source.is_file():
        return {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        return {str(row["point_id"]): row for row in csv.DictReader(stream)}


@lru_cache(maxsize=1)
def planner_citywide_decisions() -> dict[str, dict[str, str]]:
    if not PLANNER_CITYWIDE.is_file():
        return {}
    with gzip.open(PLANNER_CITYWIDE, "rt", encoding="utf-8-sig", newline="") as stream:
        return {str(row["point_id"]): row for row in csv.DictReader(stream)}


def warning_codes(route: dict[str, Any]) -> list[str]:
    warnings = ["MULTIDATE_HOT_WEATHER_SCENARIO"]
    if route.get("fallback_retry") or route.get("spatial_fallback_length_m", 0) > 0:
        warnings.append("FALLBACK_USED")
    if route.get("prior_imputed_length_m", 0) > 0:
        warnings.append("PRIOR_IMPUTED_USED")
    if route.get("center_outside_length_m", 0) > 0:
        warnings.append("OUTSIDE_CENTER")
    if route.get("reliability_score", 100) < 45:
        warnings.append("LOW_RELIABILITY")
    return warnings


def public_route(route: dict[str, Any], geometry: dict[str, Any] | None = None) -> dict:
    distance = float(route.get("distance_m", 0))
    fallback = float(route.get("spatial_fallback_length_m", 0))
    prior = float(route.get("prior_imputed_length_m", 0))
    uncertainty_percentile = max(
        0.0, min(100.0, float(route.get("uncertainty_percentile", 50.0)))
    )
    coverage_reliability = 1.0 - min(1.0, (fallback + prior) / max(distance, 1.0))
    reliability_score = max(
        0.0,
        min(100.0, 0.70 * (100.0 - uncertainty_percentile) + 30.0 * coverage_reliability),
    )
    reliability_grade = (
        "high" if reliability_score >= 75 else "medium" if reliability_score >= 50 else "low"
    )
    route_for_warnings = {**route, "reliability_score": reliability_score}
    value = {
        key: route.get(key)
        for key in (
            "route_id",
            "found",
            "hour",
            "mode",
            "objective",
            "algorithm",
            "distance_m",
            "estimated_duration_min",
            "total_cost",
            "detour_ratio",
            "mean_shade",
            "shade_exposure",
            "mean_tmrt",
            "max_tmrt",
            "mean_utci",
            "max_utci",
            "uncertainty_mean",
            "uncertainty_max",
            "uncertainty_percentile",
            "spatial_fallback_length_m",
            "prior_imputed_length_m",
            "center_inside_length_m",
            "center_outside_length_m",
            "origin_snap_distance_m",
            "destination_snap_distance_m",
            "fallback_retry",
            "detour_limit_satisfied",
            "interface_runtime_seconds",
        )
    }
    value.update(
        {
            "direct_length_m": max(distance - fallback - prior, 0.0),
            "interpolated_length_m": 0.0,
            "low_reliability_length_m": fallback + prior,
            "reliability_score": reliability_score,
            "reliability_grade": reliability_grade,
            "fallback_length_m": fallback,
            "route_found": bool(route.get("found")),
            "warning_codes": warning_codes(route_for_warnings),
            "segment_count": len(route.get("segment_ids", [])),
            "segment_ids": route.get("segment_ids", []),
            "geometry": geometry,
        }
    )
    return value


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    ADAPTER.start()
    app.state.load_seconds = time.perf_counter() - started
    yield
    PLANNER_ADAPTER.close()
    ADAPTER.close()


app = FastAPI(
    title="南京热舒适路径规划",
    version="current",
    docs_url=None,
    redoc_url=None,
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)
app.middleware("http")(privacy_middleware)
install_error_handlers(app)


@app.get("/api/health")
def health() -> dict[str, Any]:
    worker = ADAPTER.health()
    return {
        "status": "ok",
        "interface": "current_multidate_directional_planner",
        **worker,
        "external_services_enabled": False,
        "privacy_mode": privacy_status()["mode"],
        "supported_hours": list(range(6, 19)),
        "supported_modes": ["walk", "bike", "shared"],
        "supported_objectives": ["shortest", "shade", "utci"],
        "default_model_id": "nanjing-thermal-multidate-current",
        "default_scenario_date": "2024-08-04",
        "planner_streetscape_module": True,
        "planner_analysis_unit": "single_perspective_direction_view",
        "planner_existing_point_direction_count": 12,
        "planner_self_supervised_reference": "DINOv2_71800_direction_windows",
        "planner_depth_model": "Depth Anything V2 Metric Outdoor Small CUDA",
        "service_uptime_seconds": time.time() - STARTED_AT,
        "application_load_seconds": getattr(app.state, "load_seconds", None),
        "local_assistant": llm_status(),
    }


@app.get("/api/assistant/status")
def assistant_status() -> dict[str, Any]:
    return llm_status()


@app.post("/api/assistant/parse")
def assistant_parse(request: AssistantParseRequest) -> dict[str, Any]:
    return parse_travel_request(request.text.strip())


@app.post("/api/assistant/explain")
def assistant_explain(request: AssistantExplainRequest) -> dict[str, Any]:
    routes = [route.model_dump() for route in request.routes]
    return {
        "explanation": explain_route_tradeoffs(routes, request.question),
        "model": llm_status()["model"],
        "local_only": True,
        "coordinates_sent_to_model": False,
        "conversation_persisted": False,
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "default_hour": 14,
        "hour_min": 6,
        "hour_max": 18,
        "default_mode": "walk",
        "default_objective": "shortest",
        "default_algorithm": "astar",
        "default_max_detour_ratio": 0.20,
        "default_uncertainty_weight": 0.0,
        "scenario_date": "2024-08-04",
        "model_id": "nanjing-thermal-multidate",
        "external_services_enabled": False,
        "external_tiles_enabled": False,
        "shared_topology_disclosure": (
            "现有道路属性不足以构建完全独立的步行和骑行拓扑；三种模式当前共享批准的主动交通候选网络。"
        ),
        "scenario_disclosure": (
            "当前路径成本采用多日期模型的2024-08-04极端高温晴天情景；模型训练覆盖2024年7—8月"
            "10个训练天气日，并在独立日期与独立点位上验证。建筑、DSM与显示阴影仍冻结为2024-07-29。"
        ),
        "privacy": privacy_status(),
    }


@app.get("/api/model/status")
def model_status() -> dict[str, Any]:
    """Expose model integration readiness."""
    return multidate_model_status()


@app.get("/api/scenarios")
def scenarios() -> dict[str, Any]:
    """List frozen weather scenarios and their routing-cache readiness."""
    return multidate_scenarios()


@app.post("/api/geocode")
def geocode(request: GeocodeRequest) -> dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise ValueError("Schema validation failed: query is empty")
    result = ADAPTER.tool("geocode_place", {"query": query, "limit": request.limit})
    candidates = []
    for item in result["candidates"]:
        if item["match_type"] == "fuzzy" and float(item["match_score"]) < 0.45:
            continue
        coordinate = item["coordinate"]
        candidates.append(
            {
                "candidate_id": item["candidate_id"],
                "display_name": item["display_name"],
                "feature_type": item["category"],
                "subtype": item["subtype"],
                "longitude": coordinate["longitude"],
                "latitude": coordinate["latitude"],
                "x": item["projected_coordinate"]["x"],
                "y": item["projected_coordinate"]["y"],
                "municipality": "南京市",
                "district": None,
                "match_score": item["match_score"],
                "match_type": item["match_type"],
            }
        )
    return {
        "query": query,
        "candidate_count": len(candidates),
        "ambiguity_flag": len(candidates) > 1,
        "selection_required": len(candidates) > 1,
        "candidates": candidates,
        "external_request_made": False,
    }


@app.post("/api/snap")
def snap(request: SnapRequest) -> dict[str, Any]:
    result = ADAPTER.tool(
        "snap_origin_destination",
        {
            "origin": request.origin.tool_dict(),
            "destination": request.destination.tool_dict(),
        },
    )
    if (
        float(result["origin_snap_distance_m"]) > 100
        or float(result["destination_snap_distance_m"]) > 100
    ):
        raise ValueError("snap pair exceeds within 100 m constraint")
    return {
        **result,
        "mode": request.mode,
        "same_component": True,
        "reachable": True,
        "warning_codes": [],
    }


@app.post("/api/route")
def route(request: RouteRequest) -> dict[str, Any]:
    result = ADAPTER.tool("route", request.tool_dict())
    if not result.get("found"):
        raise ValueError("no_path returned by formal tool")
    geometry = ADAPTER.geometry(result)
    return public_route(result, geometry)


@app.post("/api/compare")
def compare(request: CompareRequest) -> dict[str, Any]:
    result = ADAPTER.tool("compare_routes", request.tool_dict())
    routes = []
    for route_result in result["routes"]:
        geometry = ADAPTER.geometry(route_result) if route_result.get("found") else None
        routes.append(public_route(route_result, geometry))
    return {
        "route_count": len(routes),
        "all_found": all(route["route_found"] for route in routes),
        "routes": routes,
        "shared_topology_disclosed": True,
        "warning_codes": sorted(
            {code for route in routes for code in route["warning_codes"]}
        ),
    }


@app.post("/api/summarize")
def summarize(request: SummarizeRequest) -> dict[str, Any]:
    result = ADAPTER.route_result(request.route_id)
    summary = ADAPTER.tool(
        "summarize_route",
        {"route_result": result, "language": request.language},
    )
    summary["scenario_disclosure"] = (
        "当前路线采用多日期模型的2024-08-04极端高温晴天情景；建筑、DSM和阴影显示几何"
        "继续使用2024-07-29。模型适用于南京相近夏季晴热天气，不代表跨季节或跨城市验证。"
    )
    summary["warning_codes"] = warning_codes(result)
    return summary


@app.post("/api/export")
def export(request: ExportRequest) -> dict[str, Any]:
    route_result = ADAPTER.route_result(request.route_id)
    result = ADAPTER.export(route_result, request.format)
    path = Path(result["output_path"])
    if not path.exists():
        raise RuntimeError("export output does not exist")
    return {
        "route_id": request.route_id,
        "format": request.format,
        "filename": path.name,
        "download_url": None if path.is_dir() else f"/api/download/{path.name}",
        "local_path": str(path),
        "temporary": True,
        "contains_location_in_filename": False,
    }


@app.post("/api/visual-layers")
def visual_layers(request: VisualLayersRequest) -> dict[str, Any]:
    result = ADAPTER.visual_layers(
        request.bbox, request.hour, request.max_buildings,
        request.raster_max_dimension, request.include_vectors,
    )
    return {
        **result,
        "scenario_date": "2024-07-29",
        "source": "SOLWEIG HourlyShade2m + BuildingHeightVector",
        "display_only": True,
        "external_request_made": False,
    }


@lru_cache(maxsize=1)
def planner_pilot_cases_data() -> list[dict[str, Any]]:
    """Read additive pilot cases without replacing citywide data."""
    source = PLANNER_PACKAGE / "v2_pilot_cases.json"
    if not source.is_file():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    dual_layer: dict[str, dict[str, Any]] = {}
    if PLANNER_DUAL_LAYER.is_file():
        dual_payload = json.loads(PLANNER_DUAL_LAYER.read_text(encoding="utf-8"))
        dual_layer = {str(row["point_id"]): row for row in dual_payload.get("cases", [])}
    rows: list[dict[str, Any]] = []
    for raw in payload.get("cases", []):
        point_id = str(raw["point_id"])
        panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
        rows.append({
            **raw,
            **dual_layer.get(point_id, {}),
            "pipeline": raw.get("pipeline", "planner_visual_perception_pilot"),
            "case_key": f"pilot:{point_id}",
            "capture_month": planner_point_metadata().get(point_id, {}).get("month"),
            "phenology_review": False,
            "existing_tree_row_detected": True,
            "card_url": f"/api/planner/pilot-cases/{point_id}/card",
            "panorama_url": f"/api/planner/pilot-cases/{point_id}/panorama" if panorama else None,
        })
    return rows


@lru_cache(maxsize=1)
def planner_terminal_cases_data() -> list[dict[str, Any]]:
    """Read the additive fixed-30 terminal layer without replacing earlier pilot data."""
    if not PLANNER_TERMINAL.is_file():
        return []
    payload = json.loads(PLANNER_TERMINAL.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for raw in payload.get("cases", []):
        point_id = str(raw["point_id"])
        rows.append({
            **raw,
            "case_key": f"terminal:{point_id}",
            "pipeline": "planner_visual_perception_terminal",
            "capture_month": planner_point_metadata().get(point_id, {}).get("month"),
            "phenology_review": False,
            "existing_tree_row_detected": raw.get("terminal_tone") == "maintain",
            "card_url": f"/api/planner/terminal-cases/{point_id}/panorama",
            "panorama_url": f"/api/planner/terminal-cases/{point_id}/panorama",
        })
    return rows


def planner_cases_data() -> list[dict[str, Any]]:
    source = PLANNER_OUTPUT / "planner_recommendations.csv"
    rows: list[dict[str, Any]] = []
    if not source.is_file():
        return planner_pilot_cases_data() + planner_terminal_cases_data()
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            point_id = str(row["point_id"])
            metadata = planner_point_metadata().get(point_id, {})
            review = planner_visual_review().get(point_id)
            retain_existing = bool(
                review and review.get("decision") == "no_new_tree_required"
            )
            panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
            rows.append(
                {
                    "case_key": f"reviewed:{point_id}",
                    "pipeline": "planner_reviewed_case",
                    "point_id": point_id,
                    "priority": "无需新增/现状成长" if retain_existing else row["priority"],
                    "recommended_action": "保留连续现有树列，等待季节恢复或树木生长" if retain_existing else row["recommended_action"],
                    "mean_effective_shade_gain": 0.0 if retain_existing else float(row["mean_effective_shade_gain"]),
                    "estimated_mean_delta_tmrt_c": 0.0 if retain_existing else float(row["estimated_mean_delta_tmrt_c"]),
                    "estimated_mean_delta_utci_c": 0.0 if retain_existing else float(row["estimated_mean_delta_utci_c"]),
                    "planner_recommendation": "人工复核已确认连续树木排列；低叶量来自幼树或落叶状态，不新增乔木，后续只复核生长与季节性树冠恢复。" if retain_existing else row["planner_recommendation"],
                    "evidence_note": "人工视觉库存优先于低GVI自动判断：existing tree row retained." if retain_existing else row["evidence_note"],
                    "capture_month": metadata.get("month"),
                    "phenology_review": retain_existing or metadata.get("month") in LEAF_OFF_MONTHS,
                    "existing_tree_row_detected": retain_existing,
                    "decision_stage": "stage_2_phenology_gate" if retain_existing or metadata.get("month") in LEAF_OFF_MONTHS else "stage_4_optimization",
                    "decision_status": "暂停优化：落叶季、树种差异或现有树列复核" if retain_existing or metadata.get("month") in LEAF_OFF_MONTHS else "正式审核案例：已进入优化与量化阶段",
                    "optimization_eligible": not (retain_existing or metadata.get("month") in LEAF_OFF_MONTHS),
                    "card_url": f"/api/planner/cases/{point_id}/card",
                    "panorama_url": f"/api/planner/cases/{point_id}/panorama" if panorama else None,
                    "evidence_level": "formal_reviewed_case",
                    "performance_available": True,
                }
            )
    return rows + planner_pilot_cases_data() + planner_terminal_cases_data()


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _optional_float(row: dict[str, str], key: str) -> float | None:
    value = str(row.get(key, "")).strip()
    return float(value) if value else None


@lru_cache(maxsize=13)
def planner_performance_by_hour(hour: int) -> dict[str, dict[str, Any]]:
    source = PLANNER_PERFORMANCE / "point_hour_benefits.csv.gz"
    if not source.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["hour"]) != hour:
                continue
            result[str(row["point_id"])] = {
                "scenario_date": row["scenario_date"],
                "status": row["performance_status"],
                "shade_before": _optional_float(row, "shade_pred_mean"),
                "shade_benchmark": _optional_float(row, "shade_benchmark_q75"),
                "effective_shade_gain": _optional_float(row, "effective_shade_gain"),
                "estimated_delta_tmrt_c": _optional_float(row, "estimated_delta_tmrt_c"),
                "delta_tmrt_response_seed_std_c": _optional_float(row, "delta_tmrt_response_seed_std_c"),
                "estimated_delta_utci_c": _optional_float(row, "estimated_delta_utci_c"),
                "delta_utci_response_seed_std_c": _optional_float(row, "delta_utci_response_seed_std_c"),
                "estimate_method": row["estimate_method"],
                "method_boundary": row["method_boundary"],
            }
    return result


@lru_cache(maxsize=13)
def planner_points_data(hour: int) -> dict[str, Any]:
    if hour < 6 or hour > 18:
        raise ValueError("hour must be 6-18")
    semantic_path = ROOT / "data/training_data/static/semantic_features_8975.csv"
    prediction_path = ROOT / "routing/data/multidate/scenarios/2024-08-04/point_predictions.csv.gz"
    semantic: dict[str, dict[str, float]] = {}
    with semantic_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            semantic[str(row["point_id"])] = {
                "svf": float(row["SVF"]),
                "gvi": float(row["GVI"]),
                "bvi": float(row["BVI"]),
                "road_ratio": float(row["class_0_ratio"]),
                "sidewalk_ratio": float(row["class_1_ratio"]),
                "person_ratio": float(row["class_11_ratio"]),
                "active_mobility_ratio": sum(float(row[f"class_{index}_ratio"]) for index in (12, 18)),
            }
    predictions: list[dict[str, Any]] = []
    with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["hour"]) != hour:
                continue
            point_id = str(row["point_id"])
            features = semantic[point_id]
            predictions.append(
                {
                    "point_id": point_id,
                    "x": float(row["utm_x"]),
                    "y": float(row["utm_y"]),
                    "shade": float(row["shade_pred_mean"]),
                    "shade_std": float(row["shade_pred_std"]),
                    "tmrt": float(row["tmrt_pred_mean"]),
                    "utci": float(row["utci_pred_mean"]),
                    **features,
                }
            )
    shade_thresholds = {
        "q25": _quantile([row["shade"] for row in predictions], .25),
        "q50": _quantile([row["shade"] for row in predictions], .50),
        "q75": _quantile([row["shade"] for row in predictions], .75),
    }
    formal = {row["point_id"]: row for row in planner_cases_data()}
    performance = planner_performance_by_hour(hour)
    counts = {"good": 0, "fair": 0, "poor": 0, "critical": 0, "seasonal": 0}
    metadata = planner_point_metadata()
    reviews = planner_visual_review()
    citywide_results = planner_citywide_decisions()
    for row in predictions:
        osm_context = planner_osm_road_context(row["point_id"])
        point_metadata = metadata.get(row["point_id"], {})
        capture_month = point_metadata.get("month")
        review = reviews.get(row["point_id"])
        existing_tree_row = bool(
            review and review.get("decision") == "no_new_tree_required"
        )
        leaf_off_risk = capture_month in LEAF_OFF_MONTHS or existing_tree_row
        if leaf_off_risk:
            grade, label = "seasonal", "落叶/树种复核"
        elif row["svf"] <= SVF_GRADE_LIMITS["good_max"]:
            grade, label = "good", "遮荫较好"
        elif row["svf"] <= SVF_GRADE_LIMITS["fair_max"]:
            grade, label = "fair", "遮荫一般"
        elif row["svf"] <= SVF_GRADE_LIMITS["poor_max"]:
            grade, label = "poor", "遮荫较差"
        else:
            grade, label = "critical", "遮荫不足"
        counts[grade] += 1
        # A road-centre panorama frequently under-segments both pavements.  The
        # scene label is therefore a probabilistic, post-shade assessment and is
        # not a hard gate unless the motor-only evidence is unusually strong.
        dynamic_active = row["person_ratio"] >= .00002 or row["active_mobility_ratio"] >= .00002
        formal_osm_match = osm_context.get("source") == "formal_osm_point_edge_mapping"
        strong_motor_only = bool(osm_context.get("motor_vehicle_only")) or (
            not formal_osm_match and row["road_ratio"] >= .28 and row["sidewalk_ratio"] < .005
            and row["bvi"] + row["gvi"] < .05 and not dynamic_active
        )
        if strong_motor_only:
            space_type, public_likelihood = "机动车专用或快速路", "较低"
        elif bool(osm_context.get("active_mobility_class")) or dynamic_active:
            space_type, public_likelihood = "公共步行骑行空间", "较高"
        elif bool(osm_context.get("thermal_comfort_relevant")):
            space_type, public_likelihood = "城市混合道路空间", "可能"
        elif row["sidewalk_ratio"] >= .018:
            space_type, public_likelihood = "公共步行骑行空间", "较高"
        else:
            space_type, public_likelihood = "城市混合道路空间", "可能"
        public_space_score = max(0.02, min(0.98,
            .28 + 4.0 * row["sidewalk_ratio"] + (0.42 if dynamic_active else 0)
            + 1.2 * (row["bvi"] + row["gvi"])
            + (0.32 if osm_context.get("active_mobility_class") else 0)
            + (0.14 if osm_context.get("thermal_comfort_relevant") else 0)
            - (0.55 if strong_motor_only else 0)
        ))
        if strong_motor_only:
            public_space_score = min(public_space_score, .24)
        scene_review = space_type != "公共步行骑行空间"
        if leaf_off_risk:
            decision_stage = "stage_2_phenology_gate"
            decision_status = "暂停优化：落叶季或树种差异复核"
            optimization_eligible = False
            action = "保留并识别现有连续树列；低叶量不得直接解释为缺树，建议在展叶季复核树冠覆盖。"
        elif grade == "good":
            decision_stage = "stage_1_shade_need"
            decision_status = "提前结束：SVF较低，无需新增遮荫"
            optimization_eligible = False
            action = "保持现有树冠与遮挡条件，重点进行养护和连续性检查。"
        elif strong_motor_only:
            decision_stage = "stage_3_space_typology"
            decision_status = "遮荫评估完成：较可能为机动车专用或快速路"
            optimization_eligible = False
            action = "保留遮荫诊断；该点较可能属于机动车专用空间，不作为步行骑行优化重点。"
        elif row["gvi"] < .14:
            decision_stage = "stage_4_optimization"
            decision_status = "进入优化阶段：连续乔木优先"
            optimization_eligible = True
            action = "优先沿人行道规划连续高大乔木林荫带。"
        elif row["gvi"] < .28:
            decision_stage = "stage_4_optimization"
            decision_status = "进入优化阶段：连接树冠缺口"
            optimization_eligible = True
            action = "连接现有树冠缺口，形成连续乔木遮荫。"
        else:
            decision_stage = "stage_4_optimization"
            decision_status = "进入优化阶段：核查树冠覆盖"
            optimization_eligible = True
            action = "现有绿量不低，重点调整树冠连续性及其对人行道的覆盖。"
        performance_estimate = performance.get(row["point_id"])
        if leaf_off_risk:
            performance_estimate = {
                "status": "not_applicable_phenology_review",
                "method_boundary": "Leaf-off or mixed-species phenology review required before intervention quantification.",
            }
        elif grade == "good":
            performance_estimate = {
                "status": "no_intervention_needed",
                "effective_shade_gain": 0.0,
                "estimated_delta_tmrt_c": 0.0,
                "delta_tmrt_response_seed_std_c": 0.0,
                "estimated_delta_utci_c": 0.0,
                "delta_utci_response_seed_std_c": 0.0,
                "method_boundary": "Low-SVF no-intervention gate.",
            }
        row.update(
            {
                "grade": grade,
                "grade_label": label,
                "scene_review_required": scene_review,
                "space_type_assessment": space_type,
                "public_walk_cycle_likelihood": public_likelihood,
                "public_space_score": round(public_space_score, 4),
                "space_type_is_probabilistic": True,
                "osm_road_context": osm_context,
                "capture_year": point_metadata.get("year"),
                "capture_month": capture_month,
                "leaf_off_risk": leaf_off_risk,
                "existing_tree_row_detected": existing_tree_row,
                "phenology_note": "早春/落叶季或混合树种可能导致GVI低估，不能据此新增树木。" if leaf_off_risk else "展叶季语义指标可用于遮荫初筛。",
                "decision_stage": decision_stage,
                "decision_status": decision_status,
                "optimization_eligible": optimization_eligible,
                "recommended_action": action,
                "formal_reviewed": row["point_id"] in formal,
                "formal_case": formal.get(row["point_id"]),
                "performance_estimate": performance_estimate,
                "panorama_url": f"/api/planner/points/{row['point_id']}/panorama",
                "teacher_reference_headings": _planner_ready_headings(row["point_id"]),
            }
        )
        citywide_row = citywide_results.get(row["point_id"])
        if citywide_row:
            decision_class = citywide_row["final_decision_class"]
            frozen_eligible = str(citywide_row.get("optimization_eligible_final", "")).lower() == "true"
            interactive_eligible = optimization_eligible
            if decision_class in {
                "MOTOR_VEHICLE_ONLY_EXCLUDED", "NO_INTERVENTION_LOW_SVF",
                "NO_NEW_TREE_EXISTING_SHADE_OR_CANOPY", "NO_INTERVENTION_COVERED_INFRASTRUCTURE",
                "EXISTING_TREE_PHENOLOGY_REVIEW", "AUTO_PLANNING_ABSTAIN_PHENOLOGY_UNCERTAIN",
            }:
                interactive_eligible = False
            elif decision_class in {
                "EXISTING_TREE_ROW_COVERAGE_REVIEW", "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW",
            }:
                # Existing trees are not a reason to suppress a future mature-crown
                # or gap-completion scenario when SVF still shows a shade deficit.
                interactive_eligible = grade not in {"good", "seasonal"} and not strong_motor_only
            elif decision_class in {"TREE_PRIORITY_CANDIDATE", "ACTIONABLE_VISUAL_ADVICE"}:
                interactive_eligible = True
            actions = {
                "MOTOR_VEHICLE_ONLY_EXCLUDED": "机动车专用道路不进入步行骑行遮荫优化。",
                "NO_INTERVENTION_LOW_SVF": "SVF已较低，无需新增遮荫；保持现有树冠和遮挡条件。",
                "PENDING_GPU_VISUAL_LOCALIZATION": "已通过道路类型与SVF门槛，等待GPU语义、深度和人行空间几何定位后再发布具体优化位置。",
                "NO_NEW_TREE_EXISTING_SHADE_OR_CANOPY": "保留现有连续树冠或遮挡条件，不新增树木。",
                "NO_INTERVENTION_COVERED_INFRASTRUCTURE": "现有覆盖设施已提供遮挡，不新增树木。",
                "AUTO_PLANNING_ABSTAIN_INSUFFICIENT_VISIBLE_WALKABLE": "可见人行空间证据不足，自动弃权，不强行提出种植位置。",
                "ACTIONABLE_VISUAL_ADVICE": "优先补齐连续高大乔木林荫带；人工遮阳仅作规划师补充复核。",
            }
            actions.update({
                "EXISTING_TREE_ROW_COVERAGE_REVIEW": "已有多视角树木证据；重点复核树冠连续性。SVF仍有缺口时可生成树冠成熟或缺口补齐情景，而不是默认新增整株树木。",
                "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW": "已有树木证据但绿量偏低；优先生成幼树冠幅成熟与连续林荫情景，辅助评估生长和养护效果。",
                "EXISTING_TREE_PHENOLOGY_REVIEW": "落叶或物候不确定影像；结合多视角树木证据复核现状树列，不因低GVI自动新增树木。",
                "AUTO_PLANNING_ABSTAIN_PHENOLOGY_UNCERTAIN": "物候不确定且现状树木证据不足，自动规划弃权。",
                "TREE_PRIORITY_CANDIDATE": "未发现可靠现状树木证据且可步行空间稳定可见；进入树木优先候选定位，人工遮阳仅作补充复核。",
            })
            row.update({
                "decision_stage": citywide_row["final_decision_stage"],
                "decision_status": citywide_row["final_decision_status"],
                "recommended_action": actions.get(decision_class, row["recommended_action"]),
                "optimization_eligible": interactive_eligible,
                "frozen_optimization_eligible": frozen_eligible,
                "interactive_scenario_eligible": interactive_eligible,
                "scenario_generation_mode": (
                    "existing_tree_maturation_or_gap_completion"
                    if decision_class in {"EXISTING_TREE_ROW_COVERAGE_REVIEW", "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW"}
                    else "new_tree_crown_priority" if interactive_eligible else "no_generation_recommended"
                ),
                "scene_review_required": decision_class == "MOTOR_VEHICLE_ONLY_EXCLUDED",
                "leaf_off_risk": int(citywide_row["month"]) in LEAF_OFF_MONTHS,
                "march_phenology_uncertain": str(citywide_row.get("march_phenology_uncertain", "")).lower() == "true",
                "planner_citywide_class": decision_class,
                "planner_citywide_basis": citywide_row["final_decision_basis"],
                "high_resolution_gpu_required": False,
                "gpu_localization_complete": str(citywide_row.get("gpu_localization_complete", "")).lower() == "true",
                "existing_tree_evidence": None if citywide_row.get("existing_tree_evidence", "") == "" else str(citywide_row.get("existing_tree_evidence", "")).lower() == "true",
                "walkable_direction_count": int(float(citywide_row["walkable_direction_count"])) if citywide_row.get("walkable_direction_count", "") else None,
                "tree_evidence_direction_count": int(float(citywide_row["tree_evidence_direction_count"])) if citywide_row.get("tree_evidence_direction_count", "") else None,
                "walkable_tree_cooccurrence_proxy": float(citywide_row["walkable_tree_cooccurrence_proxy"]) if citywide_row.get("walkable_tree_cooccurrence_proxy", "") else None,
                "season_aware_tree_evidence_count": int(float(citywide_row["season_aware_tree_evidence_count"])) if citywide_row.get("season_aware_tree_evidence_count", "") else None,
            })
            # Do not let a missing pavement mask terminate the workflow.  The
            # Multi-view evidence remains visible, but shade need and a
            # provisional intervention class are still reported.
            if decision_class == "AUTO_PLANNING_ABSTAIN_INSUFFICIENT_VISIBLE_WALKABLE":
                rescued_eligible = grade not in {"good", "seasonal"}
                row.update({
                    "planner_citywide_class": "POTENTIAL_ACTIVE_MOBILITY_SHADE_REVIEW",
                    "decision_stage": "stage_3_space_typology",
                    "decision_status": "遮荫评估已完成；道路中心视角下人行空间可见性不足，保留为潜在步行骑行优化点",
                    "recommended_action": "保留为城市混合道路空间候选，继续评估树冠优化；不因道路中心视角下人行道像素偏少而终止。",
                    "optimization_eligible": rescued_eligible,
                    "interactive_scenario_eligible": rescued_eligible,
                    "scenario_generation_mode": "new_tree_crown_priority" if rescued_eligible else "no_generation_recommended",
                    "space_type_assessment": "城市混合道路空间",
                    "public_walk_cycle_likelihood": "可能",
                })
        decision_class = row.get("planner_citywide_class", "")
        if grade == "good" or decision_class in {"NO_INTERVENTION_LOW_SVF", "NO_INTERVENTION_COVERED_INFRASTRUCTURE", "NO_NEW_TREE_EXISTING_SHADE_OR_CANOPY"}:
            optimization_class, optimization_label = "no_intervention", "无需新增遮荫"
        elif leaf_off_risk or decision_class in {"EXISTING_TREE_PHENOLOGY_REVIEW", "EXISTING_YOUNG_OR_SPARSE_TREE_GROWTH_REVIEW"}:
            optimization_class, optimization_label = "tree_growth", "现有树木生长/养护"
        elif strong_motor_only or decision_class == "MOTOR_VEHICLE_ONLY_EXCLUDED":
            optimization_class, optimization_label = "motor_only_review", "机动车专用空间候选"
        elif row["gvi"] < .22:
            optimization_class, optimization_label = "tree_priority", "连续乔木优先"
        elif row["bvi"] >= .14:
            optimization_class, optimization_label = "artificial_supplement", "乔木优先·人工遮阳补充"
        else:
            optimization_class, optimization_label = "canopy_gap", "连接现有树冠缺口"
        perf = row.get("performance_estimate") or {}
        gain = perf.get("effective_shade_gain")
        target_svf = min(row["svf"], .25) if optimization_class != "no_intervention" else row["svf"]
        target_gvi = max(row["gvi"], .20) if optimization_class in {"tree_priority", "canopy_gap", "tree_growth"} else row["gvi"]
        visual_change_target = min(.25, max(0.0, row["svf"] - target_svf) + .35 * max(0.0, target_gvi - row["gvi"]))
        row.update({
            "optimization_class": optimization_class,
            "optimization_class_label": optimization_label,
            "optimization_metrics": {
                "before": {"svf": row["svf"], "shade": row["shade"], "gvi": row["gvi"], "tmrt": row["tmrt"], "utci": row["utci"]},
                "target_or_after": {
                    "svf_screening_target": target_svf,
                    "shade": None if gain is None else min(1.0, row["shade"] + gain),
                    "gvi_screening_target": target_gvi,
                    "tmrt": None if perf.get("estimated_delta_tmrt_c") is None else row["tmrt"] + perf["estimated_delta_tmrt_c"],
                    "utci": None if perf.get("estimated_delta_utci_c") is None else row["utci"] + perf["estimated_delta_utci_c"],
                },
                "visual_change_ratio_target": visual_change_target,
                "target_note": "SVF/GVI为规划筛选目标；图像改变比例为建议叠加区目标，选择点位后由GPU语义叠加结果更新。",
            },
        })
    optimization_counts: dict[str, int] = {}
    for row in predictions:
        key = str(row["optimization_class"])
        optimization_counts[key] = optimization_counts.get(key, 0) + 1
    return {
        "scenario_date": "2024-08-04",
        "hour": hour,
        "point_count": len(predictions),
        "thresholds": {"svf": SVF_GRADE_LIMITS, "shade_context": shade_thresholds},
        "grade_counts": counts,
        "optimization_counts": optimization_counts,
        "classification_method": "svf_primary_absolute_thresholds_with_hourly_shade_context_and_phenology_gate",
        "points": predictions,
    }


@app.get("/api/planner/cases")
def planner_cases() -> dict[str, Any]:
    rows = planner_cases_data()
    return {"case_count": len(rows), "cases": rows, "tree_priority": True}


@app.get("/api/planner/terminal-cases")
def planner_terminal_cases() -> dict[str, Any]:
    rows = planner_terminal_cases_data()
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["terminal_label"])
        counts[label] = counts.get(label, 0) + 1
    return {"case_count": len(rows), "status_counts": counts, "cases": rows,
            "citywide_v1_replaced": False, "terminal_decisions": True}


@app.get("/api/planner/validation")
def planner_validation() -> dict[str, Any]:
    rows = planner_terminal_cases_data()
    counts: dict[str, int] = {}
    for row in rows:
        label = str(row["terminal_label"])
        counts[label] = counts.get(label, 0) + 1
    return {
        "status": "validated_fixed_benchmark",
        "point_count": len(rows),
        "status_counts": counts,
        "core_conclusion": "多模态视觉证据将固定30点收敛为四种可解释终态；证据不足时主动弃权，而非强制生成干预方案。",
        "method_chain": ["北向无重影全景", "语义与深度感知", "公共空间门槛", "SVF与树冠诊断", "终态决策"],
        "claim_boundary": "固定30点技术验证；终态分类不是施工设计，主动弃权不表示人行空间不存在。",
        "figure_url": "/api/planner/validation/figure",
        "pdf_url": "/api/planner/validation/pdf",
        "source_data_url": "/api/planner/validation/source-data",
        "citywide_v1_replaced": False,
    }


@app.get("/api/planner/validation/figure")
def planner_validation_figure() -> FileResponse:
    if not PLANNER_VALIDATION_FIGURE.is_file():
        raise ValueError("Validation figure is unavailable")
    return FileResponse(PLANNER_VALIDATION_FIGURE)


@app.get("/api/planner/validation/pdf")
def planner_validation_pdf() -> FileResponse:
    if not PLANNER_VALIDATION_PDF.is_file():
        raise ValueError("Validation PDF is unavailable")
    return FileResponse(PLANNER_VALIDATION_PDF, filename="nanjing_planner_terminal_validation.pdf")


@app.get("/api/planner/validation/source-data")
def planner_validation_source_data() -> FileResponse:
    if not PLANNER_VALIDATION_SOURCE.is_file():
        raise ValueError("Validation source data is unavailable")
    return FileResponse(PLANNER_VALIDATION_SOURCE, filename="nanjing_planner_terminal_source_data.csv")


@app.get("/api/planner/citywide")
def planner_citywide_summary() -> dict[str, Any]:
    required = [PLANNER_CITYWIDE_SUMMARY, PLANNER_CITYWIDE_FIGURE,
                PLANNER_CITYWIDE_PDF, PLANNER_CITYWIDE_SOURCE,
                PLANNER_CITYWIDE_ELIGIBLE]
    if not all(path.is_file() for path in required):
        raise ValueError("Citywide terminal visualization is unavailable")
    summary = json.loads(PLANNER_CITYWIDE_SUMMARY.read_text(encoding="utf-8"))
    with PLANNER_CITYWIDE_ELIGIBLE.open("r", encoding="utf-8-sig", newline="") as stream:
        eligible_ids = [str(row["point_id"]) for row in csv.DictReader(stream)]
    counts = summary["decision_group_counts"]
    return {
        "status": "citywide_terminal_complete",
        "point_count": int(summary["point_count"]),
        "status_counts": {
            "现状树木 / 物候复核": int(counts["existing_tree_review"]),
            "维持现状 / 无需干预": int(counts["maintain_or_no_intervention"]),
            "排除 / 主动弃权": int(counts["excluded_or_abstained"]),
            "树木优先候选": int(counts["optimization_eligible"]),
        },
        "eligible_point_ids": eligible_ids,
        "core_conclusion": "多模态证据将全市8,975点收敛为维持、复核、排除/弃权与5个树木优先候选，而不是对高SVF点统一生成干预。",
        "method_chain": ["8,975点全市筛查", "6,014点流式CUDA定位", "多视角树木与人行证据", "5个规划师复核候选"],
        "claim_boundary": "5点是进入规划师复核的候选，不是自动施工选址；树木与人行方向均为多视角代理证据。",
        "figure_url": "/api/planner/citywide/figure",
        "pdf_url": "/api/planner/citywide/pdf",
        "source_data_url": "/api/planner/citywide/source-data",
        "eligible_points_url": "/api/planner/citywide/eligible-points",
        "fixed_benchmark_url": "/api/planner/validation",
    }


@app.get("/api/planner/citywide/figure")
def planner_citywide_figure() -> FileResponse:
    if not PLANNER_CITYWIDE_FIGURE.is_file():
        raise ValueError("Citywide figure is unavailable")
    return FileResponse(PLANNER_CITYWIDE_FIGURE)


@app.get("/api/planner/citywide/pdf")
def planner_citywide_pdf() -> FileResponse:
    if not PLANNER_CITYWIDE_PDF.is_file():
        raise ValueError("Citywide PDF is unavailable")
    return FileResponse(PLANNER_CITYWIDE_PDF, filename="nanjing_8975_terminal_planning_summary.pdf")


@app.get("/api/planner/citywide/source-data")
def planner_citywide_source_data() -> FileResponse:
    if not PLANNER_CITYWIDE_SOURCE.is_file():
        raise ValueError("Citywide source data is unavailable")
    return FileResponse(PLANNER_CITYWIDE_SOURCE,
                        filename="nanjing_8975_terminal_planning_source_data.csv.gz",
                        media_type="application/gzip")


@app.get("/api/planner/citywide/eligible-points")
def planner_citywide_eligible_points() -> FileResponse:
    if not PLANNER_CITYWIDE_ELIGIBLE.is_file():
        raise ValueError("Citywide eligible-point data is unavailable")
    return FileResponse(PLANNER_CITYWIDE_ELIGIBLE,
                        filename="nanjing_tree_priority_candidates_5.csv")


@app.get("/api/planner/performance-report")
def planner_performance_report() -> FileResponse:
    source = PLANNER_PERFORMANCE / "point_benefit_summary.csv"
    if not source.is_file():
        raise ValueError("Citywide planner performance report is unavailable")
    return FileResponse(source, filename="nanjing_8975_point_shade_performance_report.csv")


@app.get("/api/planner/final-decisions")
def planner_final_decisions() -> FileResponse:
    if not PLANNER_FINAL_DECISIONS.is_file():
        raise ValueError("Citywide final planning decisions are unavailable")
    return FileResponse(
        PLANNER_FINAL_DECISIONS,
        filename="nanjing_8975_final_planning_decisions.csv.gz",
    )


@app.get("/api/planner/points")
def planner_points(hour: int = 14) -> dict[str, Any]:
    return planner_points_data(hour)


@app.get("/api/planner/points/{point_id}/directions")
def planner_point_directions(point_id: str) -> dict[str, Any]:
    records = planner_direction_records(point_id)
    return {
        "point_id": point_id,
        "direction_count": len(records),
        "headings": [item["heading"] for item in records],
        "directions": [
            {
                **{key: item[key] for key in ("heading", "filename", "capture_date", "longitude", "latitude")},
                "image_url": f"/api/planner/points/{point_id}/directions/{item['heading']}/image",
                "analysis_url": f"/api/planner/points/{point_id}/directions/{item['heading']}/analyze",
            }
            for item in records
        ],
        "fusion_panorama_url": f"/api/planner/points/{point_id}/panorama",
        "analysis_unit": "independent_perspective_direction_view",
    }


@app.get("/api/planner/points/{point_id}/directions/{heading}/image")
def planner_point_direction_image(point_id: str, heading: int) -> FileResponse:
    record = next((item for item in planner_direction_records(point_id) if item["heading"] == heading), None)
    if record is None:
        raise ValueError("Direction heading is unavailable")
    return FileResponse(Path(record["image_path"]))


@app.post("/api/planner/points/{point_id}/directions/{heading}/analyze")
def planner_point_direction_analysis(point_id: str, heading: int, confidence: int = 65, hour: int = 14) -> dict[str, Any]:
    record = next((item for item in planner_direction_records(point_id) if item["heading"] == heading), None)
    if record is None:
        raise ValueError("Direction heading is unavailable")
    intent = "独立分析方位图；识别道路尽头并排除，在道路左右两侧按透视关系生成树冠优先遮荫方案"
    result = PLANNER_ADAPTER.analyze(
        Path(record["image_path"]), intent, record["month"], max(0, min(100, int(confidence))),
        planner_osm_road_context(point_id),
    )
    result = attach_reference_thermal_estimate(result, hour, point_id)
    agent = build_assessment_agent_result(result, f"{point_id}:{heading}")
    teacher = _planner_ready_image2_row(point_id, heading)
    return {
        **result, **agent, "point_id": point_id, "heading": heading,
        "source": "original_direction_view", "generation_invoked": False,
        "external_request_made": False,
        "teacher_reference_available": teacher is not None,
        "teacher_pair_id": teacher["pair_id"] if teacher else None,
    }


def _direction_image_coordinates(image_path: Path) -> tuple[float, float] | None:
    parts = image_path.stem.split("_")
    try:
        longitude, latitude = float(parts[1]), float(parts[2])
    except (IndexError, TypeError, ValueError):
        return None
    return longitude, latitude


def _coordinate_distance_m(first: tuple[float, float], second: tuple[float, float]) -> float:
    mean_latitude = math.radians((first[1] + second[1]) / 2.0)
    east = (second[0] - first[0]) * 111_320.0 * math.cos(mean_latitude)
    north = (second[1] - first[1]) * 110_540.0
    return math.hypot(east, north)


def _planner_reference_direction(
    result: dict[str, Any], image_path: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Resolve the geographically nearest good-shade donor after same-image lookup."""
    references = (result.get("self_supervised_reference") or {}).get("good_shade_references") or []
    source_coordinates = _direction_image_coordinates(image_path)
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for reference in references:
        reference_point = str(reference.get("point_id", ""))
        if not reference_point.isdigit():
            continue
        target_heading = float(reference.get("reference_azimuth_deg", 0)) % 360
        try:
            records = planner_direction_records(reference_point)
        except Exception:
            continue
        record = min(records, key=lambda item: min(abs(item["heading"] - target_heading), 360 - abs(item["heading"] - target_heading)))
        path = Path(record["image_path"])
        if path.is_file():
            reference_coordinates = _direction_image_coordinates(path)
            distance = (
                _coordinate_distance_m(source_coordinates, reference_coordinates)
                if source_coordinates is not None and reference_coordinates is not None else float("inf")
            )
            candidates.append((distance, path, {
                **reference, "resolved_heading": record["heading"], "filename": path.name,
                "longitude": reference_coordinates[0] if reference_coordinates else None,
                "latitude": reference_coordinates[1] if reference_coordinates else None,
                "geographic_distance_m": round(distance, 1) if math.isfinite(distance) else None,
                "material_reference_policy": "current_image_first_then_nearest_coordinate_good_shade",
            }))
    if candidates:
        _, path, metadata = min(candidates, key=lambda item: item[0])
        return path, metadata
    return None, None


def _decode_planner_manual_mask(
    encoded: str, analysis: dict[str, Any], image_path: Path,
) -> tuple[bytes, dict[str, Any]]:
    """Validate a browser-edited mask and reapply the immutable road-end guard."""
    payload_text = encoded.split(",", 1)[-1].strip()
    try:
        payload = base64.b64decode(payload_text, validate=True)
    except Exception as error:
        raise ValueError(f"人工编辑区域不是有效PNG：{error}") from error
    task = UPLOAD_RUNTIME / f"manual_mask_{uuid.uuid4().hex}"
    task.mkdir(parents=True, exist_ok=True)
    try:
        edited_path = task / "edited.png";edited_path.write_bytes(payload)
        command = [
            str(Path(r"D:\miniconda3\envs\gptthermalcomfort\python.exe")),
            str(ROOT / "routing/app/backend/planner_manual_mask_worker.py"),
            "--image", str(image_path), "--edited-mask", str(edited_path),
            "--output", str(task / "sanitized.png"),
        ]
        protected_payload = analysis.get("road_end_protected_mask_png_base64")
        if protected_payload:
            protected_path = task / "protected.png"
            protected_path.write_bytes(base64.b64decode(str(protected_payload)))
            command.extend(["--protected-mask", str(protected_path)])
        completed = subprocess.run(
            command, cwd=str(ROOT), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        last_line = next((line for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")), "")
        result = json.loads(last_line) if last_line else {"status": "FAIL", "error": completed.stderr[-1000:]}
        if completed.returncode != 0 or result.get("status") != "PASS":
            raise ValueError(str(result.get("error") or "人工编辑区域校验失败"))
        return (task / "sanitized.png").read_bytes(), {key: value for key, value in result.items() if key != "status"}
    finally:
        shutil.rmtree(task, ignore_errors=True)


def _persist_planner_feedback(
    image_path: Path, automatic_mask: bytes, edited_mask: bytes,
    generated_path: Path, metadata: dict[str, Any],
) -> str:
    """Persist an opt-in preference example; this does not train online."""
    feedback_id = f"feedback_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    target = PLANNER_FEEDBACK / feedback_id
    with PLANNER_FEEDBACK_LOCK:
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(image_path, target / f"source{image_path.suffix.lower() or '.png'}")
        (target / "automatic_mask.png").write_bytes(automatic_mask)
        (target / "planner_edited_mask.png").write_bytes(edited_mask)
        shutil.copy2(generated_path, target / "accepted_tree_scenario.png")
        (target / "feedback.json").write_text(
            json.dumps({
                **metadata,
                "feedback_id": feedback_id,
                "feedback_type": "accepted_manual_tree_generation",
                "consent": "opt_in_rlhf_preference_learning_candidate",
                "online_training_performed": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return feedback_id


def _persist_planner_no_intervention_feedback(
    image_path: Path, automatic_mask: bytes | None, reason: str, metadata: dict[str, Any],
) -> str:
    """Persist an explicit planner preference that no shade intervention is needed."""
    feedback_id = f"feedback_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    target = PLANNER_FEEDBACK / feedback_id
    decision = {
        "decision": "no_shade_optimization_required",
        "reason": reason,
        "submitted_by": "planner_review",
    }
    with PLANNER_FEEDBACK_LOCK:
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(image_path, target / f"source{image_path.suffix.lower() or '.png'}")
        if automatic_mask:
            (target / "automatic_mask.png").write_bytes(automatic_mask)
        (target / "planner_decision.json").write_text(
            json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        (target / "feedback.json").write_text(
            json.dumps({
                **metadata, **decision, "feedback_id": feedback_id,
                "feedback_type": "no_intervention_natural_language_reason",
                "consent": "opt_in_rlhf_preference_learning_candidate",
                "online_training_performed": False,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return feedback_id


def _no_intervention_review_response(
    image_path: Path, analysis: dict[str, Any], reason: str, feedback_consent: bool,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    reason = reason.strip()
    if len(reason) < 5:
        raise ValueError("请用至少5个字符说明无需优化的理由")
    feedback_id = None
    if feedback_consent:
        mask = analysis.get("proposal_mask_png_base64")
        automatic_mask = base64.b64decode(str(mask)) if mask else None
        feedback_id = _persist_planner_no_intervention_feedback(
            image_path, automatic_mask, reason, metadata,
        )
    return {
        "review_status": "NO_INTERVENTION_CONFIRMED",
        "decision": "no_shade_optimization_required",
        "reason": reason,
        "feedback_saved": feedback_id is not None,
        "feedback_id": feedback_id,
        "feedback_verification_url": f"/api/planner/feedback/{feedback_id}" if feedback_id else None,
        "feedback_usage": "future_RLHF_or_preference_learning_candidate_only" if feedback_id else "session_only_not_persisted",
        "online_training_performed": False,
    }


IMAGE2_GENERATION_PROMPT = (
    "Add a continuous band of mature, broad, lush tree canopy in the proposed "
    "optimization region shown by the planning overlay. Match the existing "
    "street trees, camera perspective, depth, scale, daylight, shadows and "
    "local vegetation. Preserve roads, buildings, vehicles, people, signs, "
    "wires and all pixels outside the proposed region as closely as possible."
)


def _manual_external_generation_package(
    image_path: Path, analysis: dict[str, Any], point_id: str, heading: int,
    confidence: int, hour: int, manual_mask_png_base64: str | None = None,
) -> dict[str, Any]:
    candidate_available = bool(
        analysis.get("generation_candidate_available", analysis.get("optimization_eligible"))
    )
    if not candidate_available:
        raise ValueError("当前方位属于无需新增遮荫、物候复核或机动车专用排除情形，不建议生成干预情景")
    if not analysis.get("directional_spatial_anchor_available") and not manual_mask_png_base64:
        raise ValueError("当前方位仍有遮荫需求，但自动空间锚点不足；请先使用“微调生成区域”在道路两侧圈定树冠范围")
    return {
        "generation_available": False,
        "image2_request_available": True,
        "manual_external_generation_package_available": True,
        "automatic_acceptance": False,
        "planner_review_required": False,
        "agent_state": "HUMAN_EXTERNAL_GENERATION_PACKAGE_READY",
        "agent_explanation": "本地 Agent 已准备源图、规划叠加、掩膜和提示材料；用户可选择人工使用外部图像工具，程序不会发起外部请求。",
        "source_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
        "planning_overlay_png_base64": analysis.get("overlay_png_base64"),
        "proposal_mask_png_base64": manual_mask_png_base64 or analysis.get("proposal_mask_png_base64"),
        "prompt": IMAGE2_GENERATION_PROMPT,
        "generator": "human_operated_external_image_tool",
        "source": "local_planning_package_for_human_external_generation",
        "claim_boundary": "外部生成结果必须重新上传并通过本地验收；不是现场实测，也不是因果热效益验证。",
        "external_request_made": False,
        "local_diffusion_disabled": True,
        "point_id": point_id,
        "heading": heading,
        "hour": hour,
        "confidence": confidence,
        "manual_mask_used": bool(manual_mask_png_base64),
    }


def _external_candidate_acceptance(
    source_path: Path, candidate_path: Path,
    source: dict[str, Any], candidate: dict[str, Any],
) -> dict[str, Any]:
    source_rgb = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.int16)
    candidate_rgb = np.asarray(Image.open(candidate_path).convert("RGB"), dtype=np.int16)
    if source_rgb.shape != candidate_rgb.shape:
        return tool_result(
            "generation_acceptance", "FAIL",
            observations={"automatic_acceptance": False, "same_dimensions": False},
            error="candidate_dimensions_do_not_match_source",
        )
    mask_payload = source.get("proposal_mask_png_base64")
    if not mask_payload:
        return tool_result(
            "generation_acceptance", "FAIL",
            observations={"automatic_acceptance": False, "proposal_mask_available": False},
            error="source_has_no_reliable_proposal_mask",
        )
    height, width = source_rgb.shape[:2]
    mask = np.asarray(
        Image.open(io.BytesIO(base64.b64decode(str(mask_payload)))).convert("L")
        .resize((width, height), Image.Resampling.NEAREST),
    ) > 0
    if not mask.any():
        return tool_result(
            "generation_acceptance", "FAIL",
            observations={"automatic_acceptance": False, "proposal_mask_available": False},
            error="source_proposal_mask_is_empty",
        )
    difference = np.max(np.abs(candidate_rgb - source_rgb), axis=2)
    outside_max = int(difference[~mask].max()) if (~mask).any() else 0
    changed_inside = float((difference[mask] > 8).mean())
    source_ratios = source.get("ratios") or {}
    candidate_ratios = candidate.get("ratios") or {}
    vegetation_gain = float(candidate_ratios.get("vegetation", 0.0)) - float(source_ratios.get("vegetation", 0.0))
    building_gain = float(candidate_ratios.get("building", 0.0)) - float(source_ratios.get("building", 0.0))
    sky_change = float(candidate.get("svf", 0.0)) - float(source.get("svf", 0.0))
    minimum_vegetation_gain = max(.0015, min(.012, float(mask.mean()) * .08))
    gates = {
        "source_generation_candidate": bool(source.get("generation_candidate_available", source.get("optimization_eligible"))),
        "same_dimensions": True,
        "preserve_outside_mask": outside_max == 0,
        "target_region_fill": changed_inside >= .55,
        "vegetation_gain": vegetation_gain >= minimum_vegetation_gain,
        "building_fidelity": building_gain <= max(.004, vegetation_gain * .50),
        "sky_geometry": sky_change <= .05,
    }
    accepted = all(gates.values())
    metrics = {
        "outside_mask_max_difference": outside_max,
        "changed_fraction_inside_mask": round(changed_inside, 4),
        "vegetation_ratio_gain": round(vegetation_gain, 4),
        "minimum_vegetation_ratio_gain": round(minimum_vegetation_gain, 4),
        "building_ratio_gain": round(building_gain, 4),
        "directional_sky_view_change": round(sky_change, 4),
    }
    return tool_result(
        "generation_acceptance", "PASS" if accepted else "FAIL",
        observations={"automatic_acceptance": accepted, "acceptance_gates": gates, "acceptance_metrics": metrics},
        artifacts={"candidate_filename": candidate_path.name, "width": width, "height": height},
        warnings=[] if accepted else ["human_imported_candidate_rejected_and_hidden"],
        error=None if accepted else "automatic_quality_gates_failed",
    )


def _generate_realistic_tree_scenario(
    image_path: Path, analysis: dict[str, Any], point_id: str, heading: int,
    confidence: int, hour: int, manual_mask_png_base64: str | None = None,
    feedback_consent: bool = False,
) -> dict[str, Any]:
    candidate_available = bool(
        analysis.get("generation_candidate_available", analysis.get("optimization_eligible"))
    )
    if not candidate_available:
        raise ValueError("当前方位属于无需新增遮荫、物候复核或机动车专用排除情形，不建议生成干预情景")
    if not analysis.get("directional_spatial_anchor_available") and not manual_mask_png_base64:
        raise ValueError("当前方位仍有遮荫需求，但自动空间锚点不足；请先使用“微调生成区域”在道路两侧圈定树冠范围")
    automatic_mask_payload = base64.b64decode(str(analysis["proposal_mask_png_base64"]))
    mask_payload = automatic_mask_payload
    manual_metadata: dict[str, Any] = {"manual_mask_used": False, "road_end_guard_reapplied": False}
    if manual_mask_png_base64:
        mask_payload, manual_metadata = _decode_planner_manual_mask(
            manual_mask_png_base64, analysis, image_path,
        )
    reference_path, reference = _planner_reference_direction(analysis, image_path)
    task = UPLOAD_RUNTIME / f"realistic_tree_{uuid.uuid4().hex}"
    task.mkdir(parents=True, exist_ok=True)
    try:
        mask_path = task / "proposal_mask.png"; mask_path.write_bytes(mask_payload)
        vegetation_mask_payload = analysis.get("vegetation_mask_png_base64")
        vegetation_mask_path = task / "source_vegetation_mask.png"
        if vegetation_mask_payload:
            vegetation_mask_path.write_bytes(base64.b64decode(str(vegetation_mask_payload)))
        semantic_label_path = task / "source_semantic_labels.png"
        semantic_label_payload = analysis.get("semantic_label_png_base64")
        if semantic_label_payload:
            semantic_label_path.write_bytes(base64.b64decode(str(semantic_label_payload)))
        protected_mask_path = task / "road_end_protected_mask.png"
        protected_mask_payload = analysis.get("road_end_protected_mask_png_base64")
        if protected_mask_payload:
            protected_mask_path.write_bytes(base64.b64decode(str(protected_mask_payload)))
        reference_vegetation_mask_path = task / "reference_vegetation_mask.png"
        if reference_path is not None:
            try:
                reference_point = str((reference or {}).get("point_id", ""))
                reference_analysis = PLANNER_ADAPTER.analyze(
                    reference_path, "为扩散编辑素材库提取邻近优质街景树冠语义掩膜",
                    None, confidence,
                    planner_osm_road_context(reference_point) if reference_point.isdigit() else None,
                )
                reference_mask_payload = reference_analysis.get("vegetation_mask_png_base64")
                if reference_mask_payload:
                    reference_vegetation_mask_path.write_bytes(base64.b64decode(str(reference_mask_payload)))
            except Exception as error:
                LOGGER.warning("Nearby crown semantic extraction failed: %s", error)
        PLANNER_ADAPTER.close()
        # Only the locally validated BrushNet editor is exposed.  The retired
        # PowerPaint-v1/Paint-by-Example/ControlNet candidates produced pasted
        # crowns or non-vegetation structures and must not enter the web result.
        worker_specs = ((
            "powerpaint_v2_1",
            Path(r"D:\miniconda3\envs\gptthermalcomfort-powerpaint\python.exe"),
            ROOT / "routing/app/backend/planner_powerpaint_worker.py",
        ),)
        workers: list[dict[str, Any]] = []
        worker_failures: list[dict[str, str]] = []
        for worker_id, python_path, script_path in worker_specs:
            command = [
                str(python_path), str(script_path), "--image", str(image_path),
                "--mask", str(mask_path), "--output-directory", str(task / worker_id),
            ]
            if vegetation_mask_path.is_file():
                command.extend(["--vegetation-mask", str(vegetation_mask_path)])
            if semantic_label_path.is_file():
                command.extend(["--semantic-labels", str(semantic_label_path)])
            if protected_mask_path.is_file():
                command.extend(["--protected-mask", str(protected_mask_path)])
            if reference_path is not None:
                command.extend(["--reference", str(reference_path)])
            if reference_vegetation_mask_path.is_file():
                command.extend(["--reference-vegetation-mask", str(reference_vegetation_mask_path)])
            completed = subprocess.run(
                command, cwd=str(ROOT), capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            last_line = next((line for line in reversed(completed.stdout.splitlines()) if line.strip().startswith("{")), "")
            worker = json.loads(last_line) if last_line else {"status": "FAIL", "error": completed.stderr[-1000:]}
            if completed.returncode == 0 and worker.get("status") == "PASS":
                worker["worker_id"] = worker_id
                workers.append(worker)
            else:
                worker_failures.append({"worker_id": worker_id, "error": str(worker.get("error") or "worker failed")})
        if not workers:
            no_region = bool(worker_failures) and all(
                "No reliable crown proposal region" in item.get("error", "")
                for item in worker_failures
            )
            return {
                "generation_available": False, "automatic_acceptance": False,
                "agent_state": "NO_PUBLISHABLE_CROWN_REGION" if no_region else "MODEL_RUNTIME_FAILED",
                "agent_explanation": (
                    "道路尽头和不可编辑街道对象保护后，本方位没有剩余的可发布树冠区域；保留规划指导结果。"
                    if no_region else "本地扩散编辑模型本次均未成功运行；不发布未经验证的现实情景。"
                ),
                "model_failures": worker_failures,
                "model_comparison": [], "full_gpu_verified": True,
                "external_request_made": False, **manual_metadata,
            }
        candidates = []
        baseline_vegetation = float((analysis.get("ratios") or {}).get("vegetation", 0.0))
        baseline_sky = float(analysis.get("svf", 0.0))
        baseline_building = float((analysis.get("ratios") or {}).get("building", 0.0))
        mask_ratio = max(float(worker.get("mask_ratio", 0.0)) for worker in workers)
        minimum_vegetation_gain = max(.0015, min(.012, mask_ratio * .08))
        proposal_array = np.asarray(
            Image.open(io.BytesIO(mask_payload)).convert("L").resize((512, 512), Image.Resampling.NEAREST)
        ) > 0
        baseline_labels = None
        if semantic_label_payload:
            baseline_labels = np.asarray(
                Image.open(io.BytesIO(base64.b64decode(str(semantic_label_payload)))).convert("L")
                .resize((512, 512), Image.Resampling.NEAREST)
            )
        baseline_target_vegetation = (
            float((baseline_labels[proposal_array] == 8).mean())
            if baseline_labels is not None and proposal_array.any() else 0.0
        )
        variants = [
            {
                **variant,
                "generator_model": worker["model"],
                "worker_id": worker["worker_id"],
                "conditioning_method": worker.get("conditioning_method"),
                "peak_vram_gib": worker.get("peak_vram_gib"),
            }
            for worker in workers for variant in worker["variants"]
        ]
        for variant in variants:
            path = Path(variant["path"])
            assessed = PLANNER_ADAPTER.analyze(
                path, "自动验收树木优先现实情景；检查树木增加、天空减少和原场景保真",
                None, confidence, planner_osm_road_context(point_id),
            )
            vegetation_gain = float(assessed["ratios"]["vegetation"]) - baseline_vegetation
            building_gain = float(assessed["ratios"]["building"]) - baseline_building
            svf_change = float(assessed["svf"]) - baseline_sky
            assessed_labels_payload = assessed.get("semantic_label_png_base64")
            assessed_labels = (
                np.asarray(
                    Image.open(io.BytesIO(base64.b64decode(str(assessed_labels_payload)))).convert("L")
                    .resize((512, 512), Image.Resampling.NEAREST)
                ) if assessed_labels_payload else None
            )
            target_vegetation_after = (
                float((assessed_labels[proposal_array] == 8).mean())
                if assessed_labels is not None and proposal_array.any() else 0.0
            )
            target_vegetation_gain = target_vegetation_after - baseline_target_vegetation
            tree_semantic_pass = (
                target_vegetation_after >= .35
                and target_vegetation_gain >= .08
                and vegetation_gain >= minimum_vegetation_gain
            )
            building_hallucination_pass = building_gain <= max(.004, vegetation_gain * .50)
            target_region_fill_pass = float(variant["changed_fraction_inside_mask"]) >= .55
            accepted = (
                int(variant["outside_mask_max_difference"]) == 0
                and target_region_fill_pass
                and tree_semantic_pass and building_hallucination_pass and svf_change <= .05
            )
            review_candidate = (
                int(variant["outside_mask_max_difference"]) == 0
                and float(variant["changed_fraction_inside_mask"]) >= .25
                and (
                    (target_vegetation_after >= .20 and target_vegetation_gain >= .03)
                    or vegetation_gain >= .0015
                )
                and building_gain <= .025
                and svf_change <= .10
            )
            candidates.append({
                **variant, "vegetation_gain": round(vegetation_gain, 4),
                "target_vegetation_fraction_before": round(baseline_target_vegetation, 4),
                "target_vegetation_fraction_after": round(target_vegetation_after, 4),
                "target_vegetation_gain": round(target_vegetation_gain, 4),
                "building_gain": round(building_gain, 4),
                "directional_sky_view_change": round(svf_change, 4),
                "minimum_required_vegetation_gain": round(minimum_vegetation_gain, 4),
                "tree_semantic_pass": tree_semantic_pass,
                "building_hallucination_pass": building_hallucination_pass,
                "target_region_fill_pass": target_region_fill_pass,
                "automatic_acceptance": accepted,
                "planner_review_candidate": review_candidate,
                "quality_tier": "automatic_pass" if accepted else "planner_review" if review_candidate else "safety_reject",
                "score": vegetation_gain + float(variant["changed_fraction_inside_mask"]) * .025 - max(0.0, building_gain) * 2.0 - max(0.0, svf_change),
            })
        best_by_model: dict[str, dict[str, Any]] = {}
        for item in candidates:
            model = str(item.get("generator_model", "local editor"))
            if model not in best_by_model or float(item.get("score", -99)) > float(best_by_model[model].get("score", -99)):
                best_by_model[model] = item
        model_alternatives = []
        for model, item in best_by_model.items():
            image_path = Path(str(item.get("path", "")))
            if not image_path.is_file():
                continue
            model_alternatives.append({
                "model": model,
                "quality_tier": item.get("quality_tier"),
                "automatic_acceptance": bool(item.get("automatic_acceptance")),
                "score": float(item.get("score", -99)),
                "image_png_base64": base64.b64encode(image_path.read_bytes()).decode("ascii"),
            })
        teacher_reference = _image2_teacher_reference(str(point_id), int(heading))
        if teacher_reference:
            model_alternatives.insert(0, teacher_reference)
        accepted = [item for item in candidates if item["automatic_acceptance"]]
        reviewable = [item for item in candidates if item["planner_review_candidate"]]
        if not accepted and not reviewable:
            return {
                "generation_available": False, "automatic_acceptance": False,
                "agent_state": "REALISTIC_SCENARIO_REJECTED",
                "agent_explanation": "多模型候选未通过最低安全门槛（掩膜外保真、树冠证据、建筑误生成与天空异常）；继续保留规划指导图。",
                "candidate_qc": candidates, "reference_tree_case": reference,
                "model_failures": worker_failures, "model_alternatives": model_alternatives,
                "model_comparison": [
                    {
                        key: item.get(key) for key in (
                            "worker_id", "generator_model", "generation_method", "seed",
                            "automatic_acceptance", "changed_fraction_inside_mask",
                            "vegetation_gain", "building_gain", "directional_sky_view_change",
                            "foliage_gain_inside_mask", "tree_semantic_pass",
                            "building_hallucination_pass", "target_region_fill_pass",
                            "planner_review_candidate", "quality_tier", "score",
                        )
                    } for item in candidates
                ],
                "material_source_priority": "current_image_most_obvious_crown_then_nearest_coordinate_good_shade",
                "material_source_method": workers[0].get("conditioning_method"),
                "full_gpu_verified": True, "external_request_made": False, **manual_metadata,
            }
        automatic_pass = bool(accepted)
        best = max(accepted if automatic_pass else reviewable, key=lambda item: item["score"])
        best_path = Path(best["path"])
        payload = base64.b64encode(best_path.read_bytes()).decode("ascii")
        feedback_id = None
        if feedback_consent and manual_mask_png_base64:
            feedback_id = _persist_planner_feedback(
                image_path, automatic_mask_payload, mask_payload, best_path,
                {
                    "point_id": point_id, "heading": heading, "hour": hour,
                    "confidence": confidence, "scenario_qc": best,
                    "generator": best["generator_model"],
                    "review_status": "automatic_pass" if automatic_pass else "planner_review_required",
                    **manual_metadata,
                },
            )
        return {
            "generation_available": True, "automatic_acceptance": automatic_pass,
            "planner_review_required": not automatic_pass,
            "agent_state": "REALISTIC_TREE_SCENARIO_ACCEPTED" if automatic_pass else "REALISTIC_SCENARIO_REVIEW_REQUIRED",
            "agent_explanation": (
                "多模型条件修补已通过自动树冠语义、比例、建筑与边界保真验收。"
                if automatic_pass else
                "候选已通过空间安全与最低树冠证据检查，但未满足全部自动发布指标；现作为规划师复核候选显示，不宣称为正式验收结果。"
            ),
            "realistic_tree_png_base64": payload, "selected_seed": best["seed"],
            "scenario_qc": best, "candidate_qc": candidates,
            "reference_tree_case": reference, "generator": best["generator_model"],
            "model_failures": worker_failures, "model_alternatives": model_alternatives,
            "model_comparison": [
                {
                    key: item.get(key) for key in (
                        "worker_id", "generator_model", "generation_method", "seed",
                        "automatic_acceptance", "changed_fraction_inside_mask",
                        "vegetation_gain", "building_gain", "directional_sky_view_change",
                        "foliage_gain_inside_mask", "planner_review_candidate",
                        "quality_tier", "score",
                    )
                } for item in candidates
            ],
            "material_source_priority": "current_image_most_obvious_crown_then_nearest_coordinate_good_shade",
            "material_source_method": best.get("conditioning_method"),
            "peak_vram_gib": max(float(worker.get("peak_vram_gib", 0.0)) for worker in workers), "full_gpu_verified": True,
            "cpu_fallback": False, "external_request_made": False,
            "point_id": point_id, "heading": heading, "hour": hour,
            **manual_metadata,
            "feedback_saved": feedback_id is not None,
            "feedback_id": feedback_id,
            "feedback_verification_url": f"/api/planner/feedback/{feedback_id}" if feedback_id else None,
            "feedback_usage": "future_RLHF_or_preference_learning_candidate_only" if feedback_id else "session_only_not_persisted",
        }
    finally:
        shutil.rmtree(task, ignore_errors=True)


@app.post("/api/planner/points/{point_id}/directions/{heading}/generate-realistic")
def planner_point_direction_realistic(point_id: str, heading: int, confidence: int = 65, hour: int = 14) -> dict[str, Any]:
    record = next((item for item in planner_direction_records(point_id) if item["heading"] == heading), None)
    if record is None:
        raise ValueError("Direction heading is unavailable")
    confidence = max(0, min(100, int(confidence)))
    teacher = _planner_ready_teacher_generation(point_id, heading, hour)
    if teacher:
        return teacher
    with PLANNER_REALISTIC_GENERATION_LOCK:
        analysis = PLANNER_ADAPTER.analyze(
            Path(record["image_path"]),
            "树木优先现实情景：排除道路尽头，仅编辑道路两侧已确认的树冠候选区",
            record["month"], confidence, planner_osm_road_context(point_id),
        )
        return _manual_external_generation_package(
            Path(record["image_path"]), analysis, point_id, heading, confidence, hour,
        )


@app.post("/api/planner/points/{point_id}/directions/{heading}/generate-realistic-edited")
def planner_point_direction_realistic_edited(
    point_id: str, heading: int, request: PlannerManualGenerationRequest,
) -> dict[str, Any]:
    record = next((item for item in planner_direction_records(point_id) if item["heading"] == heading), None)
    if record is None:
        raise ValueError("Direction heading is unavailable")
    teacher = _planner_ready_teacher_generation(point_id, heading, request.hour)
    if teacher:
        return {**teacher, "manual_mask_used": bool(request.manual_mask_png_base64)}
    with PLANNER_REALISTIC_GENERATION_LOCK:
        analysis = PLANNER_ADAPTER.analyze(
            Path(record["image_path"]),
            "规划师已编辑树冠边界：保留道路尽头安全禁区，仅在确认后的道路两侧生成真实树木",
            record["month"], request.confidence, planner_osm_road_context(point_id),
        )
        return _manual_external_generation_package(
            Path(record["image_path"]), analysis, point_id, heading,
            request.confidence, request.hour, request.manual_mask_png_base64,
        )


@app.post("/api/planner/points/{point_id}/directions/{heading}/review-no-intervention")
def planner_point_direction_no_intervention_review(
    point_id: str, heading: int, request: PlannerNoInterventionReviewRequest,
) -> dict[str, Any]:
    record = next((item for item in planner_direction_records(point_id) if item["heading"] == heading), None)
    if record is None:
        raise ValueError("Direction heading is unavailable")
    image_path = Path(record["image_path"])
    analysis = PLANNER_ADAPTER.analyze(
        image_path,
        "规划师复核为无需遮荫优化；保留自动证据供偏好学习，不执行图像生成",
        record["month"], request.confidence, planner_osm_road_context(point_id),
    )
    return _no_intervention_review_response(
        image_path, analysis, request.reason, request.feedback_consent,
        {"point_id": point_id, "heading": heading, "review_scope": "direction", "confidence": request.confidence},
    )


@app.post("/api/planner/points/{point_id}/review-no-intervention")
def planner_point_no_intervention_review(
    point_id: str, request: PlannerNoInterventionReviewRequest,
) -> dict[str, Any]:
    """Persist an explicit point-level review from the fused 12-direction panorama."""
    if not point_id.isdigit():
        raise ValueError("Invalid point_id")
    records = planner_direction_records(point_id)
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None or not records:
        raise ValueError("Point panorama is unavailable for planner review")
    return _no_intervention_review_response(
        panorama, {}, request.reason, request.feedback_consent,
        {
            "point_id": point_id, "heading": None, "review_scope": "point_panorama",
            "direction_count": len(records), "confidence": request.confidence,
        },
    )


@app.get("/api/planner/feedback/{feedback_id}")
def planner_feedback_verification(feedback_id: str) -> dict[str, Any]:
    if not feedback_id.startswith("feedback_") or not all(character.isalnum() or character == "_" for character in feedback_id):
        raise ValueError("Invalid feedback id")
    target = PLANNER_FEEDBACK / feedback_id
    metadata_path = target / "feedback.json"
    if not metadata_path.is_file():
        raise ValueError("Feedback record is unavailable")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    feedback_type = metadata.get("feedback_type", "accepted_manual_tree_generation")
    if feedback_type == "no_intervention_natural_language_reason":
        expected = ["planner_decision.json", "feedback.json"]
    else:
        expected = ["automatic_mask.png", "planner_edited_mask.png", "accepted_tree_scenario.png", "feedback.json"]
    source_present = any(target.glob("source.*"))
    files_present = {name: (target / name).is_file() for name in expected}
    return {
        "status": "verified" if source_present and all(files_present.values()) else "incomplete",
        "feedback_id": feedback_id,
        "feedback_type": feedback_type,
        "decision": metadata.get("decision"),
        "reason": metadata.get("reason"),
        "consent": metadata.get("consent"),
        "online_training_performed": False,
        "source_present": source_present,
        "files_present": files_present,
        "created_at": metadata.get("created_at"),
    }


@app.post("/api/planner/points/{point_id}/directions/analyze-all")
def planner_point_all_directions(point_id: str, confidence: int = 65, hour: int = 14) -> dict[str, Any]:
    records = planner_direction_records(point_id)
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None:
        raise ValueError("Point panorama is unavailable for decision fusion")
    intent = "12个方位图独立判断道路尽头、道路两侧、人行骑行证据和已有遮荫设施，再融合为全景决策总览"
    result = PLANNER_ADAPTER.analyze_directions(
        list(records), panorama, intent, records[0]["month"], max(0, min(100, int(confidence))),
        planner_osm_road_context(point_id),
    )
    result["directions"] = [
        attach_reference_thermal_estimate(item, hour, point_id) for item in result["directions"]
    ]
    point_level = planner_points_data(hour)["points"]
    point = next(item for item in point_level if str(item["point_id"]) == point_id)
    continuity = result.get("side_shade_continuity") or {}
    agent = build_assessment_agent_result(result, f"{point_id}:all-directions")
    return {
        **result, **agent, "point_id": point_id, "hour": hour,
        "point_level_svf": point["svf"], "svf_good_reference": .30,
        "osm_road_context": planner_osm_road_context(point_id),
        "local_directional_gap_overrides_point_average": bool(
            continuity.get("local_gap_direction_count", 0) and point["svf"] <= .30
        ),
        "source": "twelve_original_direction_views_with_fused_panorama_overview",
        "external_request_made": False,
    }


@app.get("/api/planner/points/{point_id}/panorama")
def planner_point_panorama(point_id: str) -> FileResponse:
    if not point_id.isdigit():
        raise ValueError("Invalid point_id")
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None:
        raise ValueError("Point panorama is unavailable")
    return FileResponse(panorama)


@app.post("/api/planner/points/{point_id}/analyze")
def planner_point_analysis(point_id: str, confidence: int = 65) -> dict[str, Any]:
    """Compatibility endpoint: run 12 direction views and fuse only for display."""
    return planner_point_all_directions(point_id, confidence, 14)


@app.post("/api/planner/points/{point_id}/overlay")
def planner_point_overlay(point_id: str, confidence: int = 65) -> dict[str, Any]:
    """Compatibility endpoint: return the north-facing source-view overlay."""
    return planner_point_direction_analysis(point_id, 0, confidence, 14)


@app.get("/api/planner/cases/{point_id}/card")
def planner_case_card(point_id: str) -> FileResponse:
    if point_id not in {row["point_id"] for row in planner_cases_data()}:
        raise ValueError("Unknown planner case")
    return FileResponse(PLANNER_OUTPUT / "points" / point_id / "planner_decision_card.png")


@app.get("/api/planner/cases/{point_id}/panorama")
def planner_case_panorama(point_id: str) -> FileResponse:
    if point_id not in {row["point_id"] for row in planner_cases_data()}:
        raise ValueError("Unknown planner case")
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None:
        raise ValueError("Planner panorama is unavailable")
    return FileResponse(panorama)


@app.get("/api/planner/pilot-cases/{point_id}/card")
def planner_case_card(point_id: str) -> FileResponse:
    cases = {row["point_id"]: row for row in planner_pilot_cases_data()}
    if point_id not in cases:
        raise ValueError("Unknown Planner pilot case")
    source = PLANNER_FIGURES / cases[point_id]["card_filename"]
    if not source.is_file():
        raise ValueError("Planner evidence card is unavailable")
    return FileResponse(source)


@app.get("/api/planner/pilot-cases/{point_id}/panorama")
def planner_case_panorama(point_id: str) -> FileResponse:
    if point_id not in {row["point_id"] for row in planner_pilot_cases_data()}:
        raise ValueError("Unknown Planner pilot case")
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None:
        raise ValueError("Planner panorama is unavailable")
    return FileResponse(panorama)


@app.get("/api/planner/terminal-cases/{point_id}/panorama")
def planner_terminal_case_panorama(point_id: str) -> FileResponse:
    if point_id not in {row["point_id"] for row in planner_terminal_cases_data()}:
        raise ValueError("Unknown Terminal planner case")
    panorama = next(PLANNER_PANORAMAS.glob(f"{point_id}_*_north_pano.png"), None)
    if panorama is None:
        raise ValueError("Terminal panorama is unavailable")
    return FileResponse(panorama)


@app.post("/api/planner/analyze-upload")
async def planner_analyze_upload(
    image: UploadFile = File(...), planner_intent: str = Form(""),
    capture_month: int = Form(0),
    confidence: int = Form(65),
    longitude: str = Form(""), latitude: str = Form(""),
    scenario_hour: int = Form(14),
    generate_realistic: int = Form(0),
    manual_mask_png_base64: str = Form(""),
    feedback_consent: int = Form(0),
) -> dict[str, Any]:
    if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("仅支持PNG、JPEG或WebP街景图像")
    payload = await image.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("上传图片不能超过20 MB")
    suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[image.content_type]
    UPLOAD_RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = UPLOAD_RUNTIME / f"{uuid.uuid4().hex}{suffix}"
    try:
        temporary.write_bytes(payload)
        month = capture_month if 1 <= capture_month <= 12 else None
        intent = planner_intent.strip()[:500]
        confidence = max(0, min(100, int(confidence)))
        result = PLANNER_ADAPTER.analyze(temporary, intent, month, confidence)
        PLANNER_ADAPTER.close()
        agent = PLANNER_GENERATION_AGENT.analyze_and_generate(
            temporary, intent, month, result
        )
        realistic: dict[str, Any] = {}
        if bool(generate_realistic) and not agent.get("generation_available"):
            with PLANNER_REALISTIC_GENERATION_LOCK:
                realistic = _manual_external_generation_package(
                    temporary, result, "upload", 0, confidence,
                    max(6, min(18, int(scenario_hour))),
                    manual_mask_png_base64.strip() or None,
                )
        lon = float(longitude) if longitude.strip() else None
        lat = float(latitude) if latitude.strip() else None
        if (lon is None) != (lat is None):
            raise ValueError("经度和纬度必须同时填写，或同时留空")
        merged = attach_reference_thermal_estimate(
            {**result, **agent, **realistic}, max(6, min(18, int(scenario_hour))),
            longitude=lon, latitude=lat,
        )
        return {
            **merged, "upload_persisted": False, "external_request_made": False,
            "analysis_unit": "panorama" if result.get("panorama_detected") else "single_perspective_direction_view",
            "coordinate_assumption_used": lon is None,
        }
    finally:
        temporary.unlink(missing_ok=True)


@app.post("/api/planner/validate-external-generation")
async def planner_validate_external_generation(
    source_image: UploadFile = File(...), generated_image: UploadFile = File(...),
    planner_intent: str = Form(""), capture_month: int = Form(0),
    confidence: int = Form(65), scenario_hour: int = Form(14),
) -> dict[str, Any]:
    supported = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}
    if source_image.content_type not in supported or generated_image.content_type not in supported:
        raise ValueError("源图和人工生成候选仅支持PNG、JPEG或WebP")
    source_payload = await source_image.read(20 * 1024 * 1024 + 1)
    candidate_payload = await generated_image.read(20 * 1024 * 1024 + 1)
    if max(len(source_payload), len(candidate_payload)) > 20 * 1024 * 1024:
        raise ValueError("单张图片不能超过20 MB")
    UPLOAD_RUNTIME.mkdir(parents=True, exist_ok=True)
    task_id = uuid.uuid4().hex
    source_path = UPLOAD_RUNTIME / f"{task_id}_source{supported[source_image.content_type]}"
    candidate_path = UPLOAD_RUNTIME / f"{task_id}_candidate{supported[generated_image.content_type]}"
    try:
        source_path.write_bytes(source_payload)
        candidate_path.write_bytes(candidate_payload)
        month = capture_month if 1 <= capture_month <= 12 else None
        confidence = max(0, min(100, int(confidence)))
        with PLANNER_REALISTIC_GENERATION_LOCK:
            try:
                source = PLANNER_ADAPTER.analyze(source_path, planner_intent.strip()[:500], month, confidence)
                candidate = PLANNER_ADAPTER.analyze(
                    candidate_path, "本地验收人工导入的外部生成候选", month, confidence,
                )
            finally:
                PLANNER_ADAPTER.close()
            acceptance = lambda _state: _external_candidate_acceptance(source_path, candidate_path, source, candidate)
            overlay = lambda _state: tool_result("planning_overlay", "PASS", artifacts={
                "overlay_available": bool(source.get("overlay_png_base64")),
            })
            runtime = PlannerAgentRuntime(PlannerToolRegistry([
                PlannerAgentTool("generation_acceptance", "Locally validate a human-imported candidate.", {}, {}, acceptance, requires_gpu=True),
                PlannerAgentTool("planning_overlay", "Retain the deterministic planning overlay.", {}, {}, overlay),
            ]), max_steps=3, timeout_seconds=1800)
            agent = runtime.run({
                "task_id": task_id, "goal": "validate_generation",
                "selected_action": "generation_acceptance",
                "observations": {"candidate_source": "human_imported_external_image"},
            })
        response = attach_reference_thermal_estimate(
            {
                **source, **agent,
                "source": "human_imported_external_image",
                "external_request_made": False,
                "planner_review_required": not agent["automatic_acceptance"],
                "claim_boundary": "人工外部生成候选；不是现场实测，也不是因果热效益验证。",
            }, max(6, min(18, int(scenario_hour))),
        )
        if agent["automatic_acceptance"]:
            response["generated_panorama_png_base64"] = base64.b64encode(candidate_payload).decode("ascii")
        return response
    finally:
        source_path.unlink(missing_ok=True)
        candidate_path.unlink(missing_ok=True)


@app.post("/api/planner/review-upload-no-intervention")
async def planner_review_upload_no_intervention(
    image: UploadFile = File(...), reason: str = Form(...),
    feedback_consent: int = Form(1), confidence: int = Form(65),
) -> dict[str, Any]:
    if image.content_type not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError("仅支持PNG、JPEG或WebP街景图像")
    payload = await image.read(20 * 1024 * 1024 + 1)
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("上传图片不能超过20 MB")
    suffix = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}[image.content_type]
    UPLOAD_RUNTIME.mkdir(parents=True, exist_ok=True)
    temporary = UPLOAD_RUNTIME / f"review_{uuid.uuid4().hex}{suffix}"
    try:
        temporary.write_bytes(payload)
        confidence = max(0, min(100, int(confidence)))
        analysis = PLANNER_ADAPTER.analyze(
            temporary, "规划师复核为无需遮荫优化；不执行图像生成", None, confidence,
        )
        if analysis.get("panorama_detected"):
            raise ValueError("请上传单一视角方位图，而不是360度全景图")
        return _no_intervention_review_response(
            temporary, analysis, reason, bool(feedback_consent),
            {"point_id": "upload", "heading": None, "confidence": confidence},
        )
    finally:
        temporary.unlink(missing_ok=True)


@app.get("/api/download/{filename}")
def download(filename: str) -> FileResponse:
    if Path(filename).name != filename or not filename.startswith("route_"):
        raise ValueError("Invalid export filename")
    path = ROOT / "routing" / "outputs" / "application" / filename
    if not path.is_file():
        raise ValueError("Export file not found")
    return FileResponse(path, filename=filename)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "templates" / "index.html")


app.mount("/static", StaticFiles(directory=FRONTEND / "static"), name="static")
app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")
