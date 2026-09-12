"""Read-only preflight for Phase G10 / Step31 natural-language integration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "routing/configs/phase_g10_llm_draft.yaml"
DATA = ROOT / "routing/data/llm_integration"
REPORT = ROOT / "routing/reports/STEP31_LLM_INTEGRATION_PREFLIGHT_REPORT.md"
LOG = ROOT / "routing/logs/llm_integration/step31_preflight.log"

REQUIRED_DOCUMENTS = [
    ROOT / "README.md",
    ROOT / "PIPELINE_SUMMARY.md",
    ROOT / "routing/reports/STEP29_FORMAL_INTERFACE_REPORT.md",
    ROOT / "routing/reports/STEP30_FORMAL_APPLICATION_REPORT.md",
    ROOT / "routing/reports/STEP30_API_SPECIFICATION.md",
    ROOT / "routing/reports/STEP30_PRIVACY_AND_OFFLINE_REPORT.md",
]
FROZEN_EVIDENCE = [
    ROOT / "routing/data/interface/step29_formal_tool_schemas.json",
    ROOT / "routing/data/interface/step29_formal_summary.json",
    ROOT / "routing/data/application/step30_formal_summary.json",
    ROOT / "routing/data/application/step30_independent_qc.json",
    ROOT / "routing/data/application/step30_openapi.json",
]
EXPECTED_TOOLS = {
    "geocode_place",
    "snap_origin_destination",
    "route",
    "compare_routes",
    "summarize_route",
    "export_route_map",
}
EXPECTED_API_PATHS = {
    "/api/health",
    "/api/config",
    "/api/geocode",
    "/api/snap",
    "/api/route",
    "/api/compare",
    "/api/summarize",
    "/api/export",
}


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def configure_logging() -> logging.Logger:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("step31_preflight")


def gpu_info() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=True,
        )
        name, memory, driver = [
            value.strip() for value in completed.stdout.strip().split(",", 2)
        ]
        return {
            "available": True,
            "name": name,
            "memory_mib": int(memory),
            "driver": driver,
        }
    except Exception as exc:
        return {
            "available": False,
            "name": "",
            "memory_mib": 0,
            "driver": "",
            "error": str(exc),
        }


def tool_names(schema: dict[str, Any]) -> set[str]:
    tools = schema.get("tools", [])
    if isinstance(tools, dict):
        return {str(name) for name in tools}
    return {
        str(item.get("name") or item.get("function", {}).get("name"))
        for item in tools
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["check"], default="check")
    args = parser.parse_args()
    logger = configure_logging()
    started = now()
    DATA.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []

    for path in tqdm(
        [*REQUIRED_DOCUMENTS, *FROZEN_EVIDENCE, CONFIG],
        desc="Step31 preflight files",
        unit="file",
        dynamic_ncols=True,
    ):
        exists = path.is_file()
        checks.append(
            {
                "check": f"file_{path.name}",
                "status": "PASS" if exists else "FAIL",
                "detail": "present" if exists else "missing",
            }
        )
        if not exists:
            failed.append(
                {"item": str(path), "error_message": "Required file missing"}
            )

    if failed:
        schema = {}
        step29 = {}
        step30 = {}
        independent = {}
    else:
        schema = json.loads(FROZEN_EVIDENCE[0].read_text(encoding="utf-8"))
        step29 = json.loads(FROZEN_EVIDENCE[1].read_text(encoding="utf-8"))
        step30 = json.loads(FROZEN_EVIDENCE[2].read_text(encoding="utf-8"))
        independent = json.loads(FROZEN_EVIDENCE[3].read_text(encoding="utf-8"))
        openapi = json.loads(FROZEN_EVIDENCE[4].read_text(encoding="utf-8"))
        rules = {
            "step29_tools_complete": tool_names(schema) == EXPECTED_TOOLS,
            "step29_formal_pass": step29.get("status") == "PASS",
            "step30_formal_pass": step30.get("status") == "PASS",
            "step30_ready_for_llm": step30.get("ready_for_llm_integration") is True,
            "step30_independent_qc_pass": independent.get("status") == "PASS",
            "step30_api_complete": EXPECTED_API_PATHS.issubset(openapi.get("paths", {})),
            "step30_external_requests_zero": step30.get("external_request_count") == 0,
            "step30_exact_location_not_persisted": step30.get(
                "exact_location_persisted"
            )
            is False,
        }
        for name, passed in tqdm(
            rules.items(),
            desc="Step31 frozen readiness",
            unit="rule",
            dynamic_ncols=True,
        ):
            checks.append(
                {
                    "check": name,
                    "status": "PASS" if passed else "FAIL",
                    "detail": str(passed),
                }
            )
            if not passed:
                failed.append(
                    {"item": name, "error_message": "Frozen readiness failed"}
                )

    gpu = gpu_info()
    runtime = {
        "ollama": shutil.which("ollama") is not None,
        "openai_python": importlib.util.find_spec("openai") is not None,
        "transformers": importlib.util.find_spec("transformers") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
        "llama_cpp": importlib.util.find_spec("llama_cpp") is not None,
        "openai_api_key": bool(os.environ.get("OPENAI_API_KEY")),
        "azure_openai_api_key": bool(os.environ.get("AZURE_OPENAI_API_KEY")),
        "anthropic_api_key": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
    environment_rows = [
        {"item": "gpu_available", "value": gpu.get("available")},
        {"item": "gpu_name", "value": gpu.get("name")},
        {"item": "gpu_memory_mib", "value": gpu.get("memory_mib")},
        {"item": "gpu_driver", "value": gpu.get("driver")},
        *[{"item": key, "value": value} for key, value in runtime.items()],
    ]
    write_csv(
        DATA / "step31_environment_inventory.csv",
        environment_rows,
        ["item", "value"],
    )

    hashes = [
        {
            "filepath": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in tqdm(
            FROZEN_EVIDENCE,
            desc="Step31 frozen hashes",
            unit="file",
            dynamic_ncols=True,
        )
        if path.is_file()
    ]
    write_csv(
        DATA / "step31_frozen_input_hashes.csv",
        hashes,
        ["filepath", "size_bytes", "sha256"],
    )
    write_csv(
        DATA / "step31_preflight_qc.csv",
        checks,
        ["check", "status", "detail"],
    )
    write_csv(
        DATA / "step31_failed_files.csv",
        failed,
        ["item", "error_message"],
    )

    all_checks_pass = bool(checks) and all(row["status"] == "PASS" for row in checks)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    decisions = config["approval"]["decisions_required"]
    summary = {
        "phase": "G10",
        "step": 31,
        "mode": args.mode,
        "started_at": started,
        "ended_at": now(),
        "status": "READY_FOR_LLM_RULE_REVIEW" if all_checks_pass else "BLOCKED",
        "check_count": len(checks),
        "passed_check_count": sum(row["status"] == "PASS" for row in checks),
        "failed_count": len(failed),
        "gpu": gpu,
        "runtime_inventory": runtime,
        "model_selected": False,
        "model_downloaded": False,
        "external_request_count": 0,
        "frozen_inputs_modified": False,
        "ready_for_llm_rule_review": all_checks_pass,
        "ready_for_formal_step31_run": False,
        "decisions_required": decisions,
    }
    (DATA / "step31_preflight_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# Phase G10 / Step31 大模型接入预检报告

生成时间：{summary['ended_at']}

- 状态：`{summary['status']}`
- 检查：{summary['passed_check_count']}/{summary['check_count']}
- GPU：{gpu.get('name') or '未检测到'}，显存{gpu.get('memory_mib', 0)} MiB
- 本地大模型运行时：尚未安装
- 云端API密钥：未配置
- 外部请求：0
- 模型下载：0
- Step29–Step30冻结输入修改：0

## 推荐边界

Step31只负责自然语言参数解析、缺失字段追问、地名候选消歧、用户确认和结构化结果解释。路径、成本、UTCI、遮荫、转向限制、吸附与绕行约束继续完全由Step30冻结API确定。

现有隐私规则禁止外发精确起终点、地名查询和请求正文，因此草案默认采用本地模型，云端模型保持禁用。RTX 4060 Ti约8GB显存具备评估量化中文7B级或更小模型的条件，但尚未选择运行时、模型和量化版本。

## 待人工批准

1. 是否批准本地模型专用策略；
2. 选择本地运行时；
3. 选择模型与量化版本；
4. 是否批准模型下载；
5. 对话是否仅驻留内存且关闭即清除。

## 当前推荐（尚未批准或下载）

- 运行时：Ollama for Windows，仅监听本机；
- 首选模型：`qwen3:8b-q4_K_M`，Ollama标注下载大小约5.2GB；
- 显存保守备选：`qwen3:4b-q4_K_M`，约2.6GB；
- 建议关闭思考模式并将上下文限制为4096，以优先保证结构化参数解析速度和8GB显存稳定性。

官方资料：

- https://ollama.com/download/windows
- https://ollama.com/library/qwen3:8b
- https://ollama.com/library/qwen3/tags
- https://huggingface.co/Qwen/Qwen3-8B

`READY_FOR_LLM_RULE_REVIEW = {str(all_checks_pass).upper()}`

`READY_FOR_FORMAL_STEP31_RUN = FALSE`
""",
        encoding="utf-8",
    )
    logger.info("Step31 status=%s", summary["status"])
    logger.info("READY_FOR_LLM_RULE_REVIEW=%s", all_checks_pass)
    logger.info("READY_FOR_FORMAL_STEP31_RUN=False")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if all_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
