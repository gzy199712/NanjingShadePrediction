"""Auditable planning and tool routing for the local multimodal planner Agent.

The module exposes concise decision summaries and tool calls.  It deliberately
does not persist or return hidden model reasoning.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "routing/configs/planner_agent_tools.json"


def load_agent_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def parse_planner_intent(text: str) -> dict[str, Any]:
    normalized = " ".join(text.strip().split())[:500]
    wants_facility = any(word in normalized for word in ("遮阳棚", "雨棚", "人工遮阳", "廊架"))
    wants_tree = any(word in normalized for word in ("乔木", "树冠", "林荫", "种树", "树木"))
    facility_first = any(word in normalized for word in ("人工遮阳优先", "遮阳棚优先", "只用遮阳棚", "不种树"))
    assess_only = any(word in normalized for word in ("只评估", "不要生成", "仅分析"))
    constraints: list[str] = []
    lexicon = {
        "道路南侧": ("南侧", "道路南"), "道路北侧": ("北侧", "道路北"),
        "保持机动车道": ("不占机动车道", "保留车道", "保持车道"),
        "连续树冠": ("连续", "连片", "林荫带"),
        "高大乔木": ("高大", "乔木", "大树冠"),
        "优先人行道": ("人行道", "步行"), "兼顾骑行": ("骑行", "自行车"),
    }
    for label, words in lexicon.items():
        if any(word in normalized for word in words):
            constraints.append(label)
    return {
        "raw_intent": normalized,
        "objective": "assess_only" if assess_only else "shade_intervention",
        "preferred_intervention": "facility_supplement" if facility_first or (wants_facility and not wants_tree) else "tree_first",
        "constraints": constraints or ["连续乔木优先", "保持道路与建筑结构"],
        "output_contract": ["evidence_overlay", "auditable_tool_trace", "automatic_quality_verification"],
    }


def discover_tools() -> dict[str, dict[str, Any]]:
    cfg = load_agent_config()
    paths = {
        "mask2former_semantic_overlay": ROOT / "models/facebook_mask2former-swin-large-cityscapes-semantic",
        "qwen3_vl_scene_consensus": ROOT / "models/Qwen_Qwen3-VL-2B-Instruct",
        "multimodal_rag_case_retrieval": ROOT / "streetscape/data/agent/stage_42c_multimodal_rag_index/case_index.csv.gz",
        "spherical_perspective_windows": ROOT / "streetscape/scripts/stage_44_panorama_perspective_windows.py",
        "semantic_depth_25d_layout": ROOT / "streetscape/scripts/stage_45_tree_first_semantic_25d_layout.py",
        "powerpaint_tree_editor": ROOT / "artifacts/retired/failed_experiment_dependencies/models_counterfactual/powerpaint-v1",
        "translucent_planning_overlay": ROOT / "routing/app/backend/planner_analysis_worker.py",
        "semantic_geometry_acceptance": ROOT / "streetscape/scripts/stage_47_automatic_generation_acceptance.py",
    }
    return {
        item["name"]: {
            **item,
            "available": paths[item["name"]].exists(),
            "local_path_checked": str(paths[item["name"]].relative_to(ROOT)),
        }
        for item in cfg["tools"]
    }


def build_agent_plan(intent: dict[str, Any], semantic: dict[str, Any]) -> dict[str, Any]:
    tools = discover_tools()
    motor_only = bool(semantic.get("high_speed_road_suspected"))
    shade_need = str(semantic.get("shade_optimization_need", "需复核"))
    leaf_off = bool(semantic.get("leaf_off_risk"))
    panorama = bool(semantic.get("panorama_detected"))
    requested = intent["preferred_intervention"]
    if intent["objective"] == "assess_only" or shade_need == "通常不需要":
        action = "overlay_and_report"
        reason = "用户仅要求评估，或SVF表明无需新增遮荫。"
    elif motor_only:
        action = "overlay_and_space_type_review"
        reason = "先保留遮荫诊断，再将机动车专用属性作为概率性空间类型输出。"
    elif leaf_off:
        action = "overlay_and_tree_growth_projection"
        reason = "落叶期低绿量不能解释为缺树，优先显示现有树列生长目标。"
    elif requested == "facility_supplement":
        action = "tree_first_then_facility_review"
        reason = "尊重自然语言需求，但仍执行项目既定的乔木优先约束。"
    elif panorama:
        action = "directional_source_required"
        reason = "全景图仅作为方向结果融合总览，不再直接承担方案推断或生成。"
    else:
        action = "directional_perspective_plan"
        reason = "单方位透视图是正式判断单位；先排除道路尽头，再在左右两侧生成可审计遮荫方案。"
    selected = ["mask2former_semantic_overlay", "qwen3_vl_scene_consensus", "multimodal_rag_case_retrieval", "translucent_planning_overlay"]
    if action == "directional_perspective_plan":
        selected += ["semantic_depth_25d_layout", "semantic_geometry_acceptance"]
    return {
        "controller": "bounded_observe_plan_act_verify_replan",
        "city_profile": "nanjing",
        "intent": intent,
        "selected_action": action,
        "decision_summary": reason,
        "selected_tools": selected,
        "tool_availability": {name: tools[name]["available"] for name in selected},
        "max_replans": int(load_agent_config()["max_replans"]),
        "hidden_reasoning_exposed": False,
    }


def replan_record(from_action: str, reason: str, fallback: str, attempt: int) -> dict[str, Any]:
    return {
        "phase": "replan",
        "attempt": attempt,
        "from_action": from_action,
        "failure_evidence": reason,
        "fallback_action": fallback,
        "status": "SAFE_FALLBACK",
    }
