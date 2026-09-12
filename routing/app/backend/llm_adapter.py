"""Local-only Ollama adapter for bounded natural-language assistance.

The adapter never receives coordinates, route geometry, segment identifiers, or
raw graph data. Request content is held only for the duration of a call.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import Literal


OLLAMA_BASE_URL = os.getenv("THERMAL_ROUTE_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("THERMAL_ROUTE_LLM_MODEL", "qwen3:8b-q4_K_M")
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}


class ParsedTravelIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: Literal["single_route", "compare_routes"] = "compare_routes"
    origin_query: str | None = Field(default=None, max_length=100)
    destination_query: str | None = Field(default=None, max_length=100)
    hour: int | None = Field(default=None, ge=6, le=18)
    mode: Literal["walk", "bike", "shared"] | None = None
    objective: Literal["shortest", "shade", "utci", "risk_aware"] | None = None
    max_detour_ratio: float | None = Field(default=None, ge=0, le=1)
    uncertainty_weight: float | None = Field(default=None, ge=0, le=3)


PARSE_SCHEMA = ParsedTravelIntent.model_json_schema()

PARSE_SYSTEM_PROMPT = """你是南京热舒适路径规划器的参数解析器，只负责把中文出行需求转成JSON。
不得规划路线、不得虚构地点、不得选择地名候选。严格遵守：
1. intent仅为single_route或compare_routes；“比较/四条/几种路线”用compare_routes。
2. mode仅为walk、bike、shared；步行=walk，骑行= bike，步骑均可=shared。
3. objective仅为shortest、shade、utci、risk_aware；阴凉/遮荫=shade，热舒适/低热应激=utci，
   稳健/规避不确定性=risk_aware，最短/最快=shortest。
4. hour必须为6到18的整数；下午2点=14。未提及就填null，禁止猜测。
5. 起终点保留用户写法，仅去掉“从、到、去”等连接词；未提及填null。
6. 最大绕行率若写20%则为0.2；未提及填null。不确定性权重未提及填null。
7. 只输出符合给定schema的JSON对象，不要解释。
示例：“下午2点从新街口步行到鼓楼，比较四类路线”应得到：
{"intent":"compare_routes","origin_query":"新街口","destination_query":"鼓楼","hour":14,
"mode":"walk","objective":null,"max_detour_ratio":null,"uncertainty_weight":null}"""


class LocalLLMError(RuntimeError):
    """A safe local-model failure suitable for the API error handler."""


def _assert_local_url() -> None:
    parsed = urlparse(OLLAMA_BASE_URL)
    if parsed.scheme != "http" or parsed.hostname not in ALLOWED_HOSTS:
        raise LocalLLMError("本地模型地址必须是127.0.0.1、localhost或::1。")


def _post(path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    _assert_local_url()
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LocalLLMError(
            "本地Qwen3暂时不可用；核心地名搜索与路线规划仍可直接使用。"
        ) from exc


def status() -> dict[str, Any]:
    _assert_local_url()
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags", timeout=3
        ) as response:
            data = json.loads(response.read().decode("utf-8"))
        names = [item.get("name") for item in data.get("models", [])]
        return {
            "available": OLLAMA_MODEL in names,
            "runtime": "Ollama",
            "model": OLLAMA_MODEL,
            "endpoint": "localhost",
            "conversation_persistence": False,
            "external_requests_enabled": False,
        }
    except Exception:
        return {
            "available": False,
            "runtime": "Ollama",
            "model": OLLAMA_MODEL,
            "endpoint": "localhost",
            "conversation_persistence": False,
            "external_requests_enabled": False,
        }


def parse_travel_request(text: str) -> dict[str, Any]:
    response = _post(
        "/api/chat",
        {
            "model": OLLAMA_MODEL,
            "stream": False,
            "think": False,
            "format": PARSE_SCHEMA,
            "keep_alive": "10m",
            "options": {
                "temperature": 0,
                "num_ctx": 2048,
                "num_predict": 180,
                "seed": 31,
            },
            "messages": [
                {"role": "system", "content": PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        }
    )
    raw = response.get("message", {}).get("content", "")
    try:
        parsed = ParsedTravelIntent.model_validate_json(raw)
    except (ValidationError, ValueError) as exc:
        raise LocalLLMError("本地模型未能生成有效参数，请换一种简短说法重试。") from exc

    value = parsed.model_dump()
    missing = [
        field
        for field in ("origin_query", "destination_query", "hour", "mode")
        if value[field] is None
    ]
    if value["intent"] == "single_route" and value["objective"] is None:
        missing.append("objective")
    return {
        **value,
        "missing_fields": missing,
        "ready_for_geocoding": not any(
            field in missing for field in ("origin_query", "destination_query")
        ),
        "explicit_confirmation_required": True,
        "model": OLLAMA_MODEL,
        "local_only": True,
    }


def explain_route_tradeoffs(routes: list[dict[str, Any]], question: str) -> str:
    safe_payload = [
        {
            key: route[key]
            for key in (
                "objective",
                "distance_m",
                "estimated_duration_min",
                "detour_ratio",
                "mean_shade",
                "mean_tmrt",
                "mean_utci",
                "uncertainty_mean",
                "warning_codes",
            )
        }
        for route in routes
    ]
    response = _post(
        "/api/chat",
        {
            "model": OLLAMA_MODEL,
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": 0.1,
                "num_ctx": 2048,
                "num_predict": 300,
                "seed": 31,
            },
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你只解释给定路线汇总指标，不计算路线、不推荐唯一最佳路线。"
                        "用简洁中文比较距离、遮荫、Tmrt、UTCI和不确定性；"
                        "明确选择取决于用户偏好，不得添加未提供的数值或地点。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "routes": safe_payload},
                        ensure_ascii=False,
                    ),
                },
            ],
        },
    )
    text = str(response.get("message", {}).get("content", "")).strip()
    if not text:
        raise LocalLLMError("本地模型没有返回解释。")
    return text
