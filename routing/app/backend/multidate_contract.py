"""Read-only contract for the frozen multi-date model integration."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = (
    ROOT / "routing/data/application/multidate_integration_contract.json"
)


@lru_cache(maxsize=1)
def load_contract() -> dict[str, Any]:
    if not CONTRACT_PATH.is_file():
        return {
            "status": "not_initialized",
            "model_id": None,
            "route_cost_status": "not_built",
            "routing_ready": False,
            "message": "Multi-date routing integration has not been initialized.",
        }
    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Invalid multi-date integration contract")
    return value


def model_status() -> dict[str, Any]:
    contract = load_contract()
    return {
        "status": contract.get("status", "unknown"),
        "interface": contract.get("interface"),
        "model_id": contract.get("model_id"),
        "cuda_required": contract.get("cuda_required", True),
        "cuda_preflight_passed": contract.get("cuda_preflight_passed", False),
        "route_cost_status": contract.get("route_cost_status", "not_built"),
        "cost_ready_scenarios": contract.get("cost_ready_scenarios", []),
        "routing_ready": contract.get("routing_ready", False),
        "next_action": contract.get("next_action", "build_route_costs"),
        "message": contract.get("message"),
    }


def scenarios() -> dict[str, Any]:
    contract = load_contract()
    dates = contract.get("supported_scenarios", [])
    ready = set(map(str, contract.get("cost_ready_scenarios", [])))
    return {
        "model_id": contract.get("model_id"),
        "dates": dates,
        "scenario_status": [
            {
                "date": str(date),
                "cost_ready": str(date) in ready,
                "routing_enabled": bool(contract.get("routing_ready", False))
                and str(date) in ready,
            }
            for date in dates
        ],
        "date_count": len(dates),
        "hours": contract.get("supported_hours", list(range(6, 19))),
        "shadow_geometry_date": contract.get("shadow_geometry_date"),
        "routing_ready": contract.get("routing_ready", False),
        "route_cost_status": contract.get("route_cost_status", "not_built"),
        "disclosure": (
            "天气情景来自2024年7—8月南京晴热日；建筑、DSM与阴影几何继续冻结为"
            "2024-07-29。仅完成成本缓存的情景才允许参与路径计算。"
        ),
    }
