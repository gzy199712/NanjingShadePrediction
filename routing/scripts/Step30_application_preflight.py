"""Phase G9 / Step30 user application and deployment preflight."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "routing" / "configs" / "phase_g9_application_draft.yaml"
DATA_DIR = ROOT / "routing" / "data" / "application"
REPORT_PATH = ROOT / "routing" / "reports" / "STEP30_APPLICATION_PREFLIGHT_REPORT.md"
LOG_PATH = ROOT / "routing" / "logs" / "step30_application_preflight.log"


def now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("step30")


def check(mode: str) -> int:
    logger = setup_logging()
    started = now_iso()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    if mode == "run":
        logger.error(
            "Step30 formal application run is not approved. Complete check and user review first."
        )
        return 2

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen_files = [
        ROOT / config["frozen_backend"]["interface_module"],
        ROOT / config["frozen_backend"]["schema_module"],
        ROOT / config["frozen_backend"]["cli_entrypoint"],
        ROOT / config["frozen_backend"]["formal_summary"],
        ROOT / config["frozen_backend"]["independent_qc"],
        ROOT / "routing/data/graph/step27_route_graph.npz",
        ROOT / "routing/data/graph/step27_hourly_costs.npz",
        ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv",
    ]
    hashes: list[dict[str, Any]] = []
    for path in tqdm(
        frozen_files,
        desc="Step30 frozen backend",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            exists = path.is_file()
            checks.append(
                {
                    "category": "frozen_backend",
                    "check": str(path.relative_to(ROOT)),
                    "status": "PASS" if exists else "FAIL",
                    "detail": "present" if exists else "missing",
                }
            )
            if exists:
                hashes.append(
                    {
                        "filepath": str(path),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
            else:
                failures.append(
                    {
                        "item": str(path),
                        "error_message": "Required frozen backend file is missing",
                    }
                )
        except Exception as exc:
            failures.append({"item": str(path), "error_message": str(exc)})

    formal_summary_path = ROOT / config["frozen_backend"]["formal_summary"]
    independent_qc_path = ROOT / config["frozen_backend"]["independent_qc"]
    if formal_summary_path.is_file() and independent_qc_path.is_file():
        formal = json.loads(formal_summary_path.read_text(encoding="utf-8"))
        qc = json.loads(independent_qc_path.read_text(encoding="utf-8"))
        status_checks = [
            ("step29_formal_status", formal.get("status") == "PASS", formal.get("status")),
            (
                "step29_independent_qc",
                qc.get("status") == "PASS",
                f"{qc.get('passed_check_count')}/{qc.get('check_count')}",
            ),
            (
                "no_external_request",
                formal.get("external_request_made") is False,
                formal.get("external_request_made"),
            ),
            (
                "no_exact_location_persistence",
                formal.get("exact_location_persisted") is False,
                formal.get("exact_location_persisted"),
            ),
        ]
        for name, passed, detail in tqdm(
            status_checks,
            desc="Step30 formal evidence",
            unit="check",
            dynamic_ncols=True,
        ):
            checks.append(
                {
                    "category": "formal_evidence",
                    "check": name,
                    "status": "PASS" if passed else "FAIL",
                    "detail": detail,
                }
            )

    dependencies = [
        ("numpy", True),
        ("pandas", True),
        ("jsonschema", True),
        ("yaml", True),
        ("osgeo", True),
        ("pydantic", False),
        ("fastapi", False),
        ("uvicorn", False),
        ("jinja2", False),
    ]
    dependency_rows: list[dict[str, Any]] = []
    for module, required_for_check in tqdm(
        dependencies,
        desc="Step30 dependencies",
        unit="module",
        dynamic_ncols=True,
    ):
        available = importlib.util.find_spec(module) is not None
        status = "PASS" if available or not required_for_check else "FAIL"
        dependency_rows.append(
            {
                "module": module,
                "available": available,
                "required_for_check": required_for_check,
                "status": status,
            }
        )
        checks.append(
            {
                "category": "dependency",
                "check": module,
                "status": status,
                "detail": (
                    "available"
                    if available
                    else "not installed; formal app dependency decision pending"
                ),
            }
        )

    privacy_checks = {
        "localhost_only": config["recommended_delivery"]["bind_host"] == "127.0.0.1",
        "external_geocoder_disabled": not config["recommended_delivery"][
            "external_geocoder_allowed"
        ],
        "external_tiles_disabled": not config["recommended_delivery"][
            "external_map_tiles_allowed"
        ],
        "exact_location_persistence_disabled": not config["recommended_delivery"][
            "exact_location_persistence_allowed"
        ],
        "request_body_logging_disabled": not config["recommended_delivery"][
            "request_body_logging_allowed"
        ],
    }
    for name, passed in tqdm(
        privacy_checks.items(),
        total=len(privacy_checks),
        desc="Step30 privacy design",
        unit="check",
        dynamic_ncols=True,
    ):
        checks.append(
            {
                "category": "privacy_design",
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": str(passed),
            }
        )

    required_failures = sum(row["status"] == "FAIL" for row in checks)
    formal_runtime_ready = (
        required_failures == 0
        and importlib.util.find_spec("fastapi") is not None
        and importlib.util.find_spec("uvicorn") is not None
        and not config["pending_user_decisions"]
    )
    ready_for_rule_review = required_failures == 0

    write_csv(
        DATA_DIR / "step30_preflight_qc.csv",
        checks,
        ["category", "check", "status", "detail"],
    )
    write_csv(
        DATA_DIR / "step30_dependency_qc.csv",
        dependency_rows,
        ["module", "available", "required_for_check", "status"],
    )
    write_csv(
        DATA_DIR / "step30_frozen_input_hashes.csv",
        hashes,
        ["filepath", "size_bytes", "sha256"],
    )
    write_csv(
        DATA_DIR / "step30_failed_records.csv",
        failures,
        ["item", "error_message"],
    )
    summary = {
        "phase": "G9",
        "step": 30,
        "mode": "check",
        "started_at": started,
        "ended_at": now_iso(),
        "status": "PASS" if ready_for_rule_review else "FAIL",
        "check_count": len(checks),
        "passed_check_count": sum(row["status"] == "PASS" for row in checks),
        "failed_check_count": required_failures,
        "failed_record_count": len(failures),
        "ready_for_phase_g9_rule_review": ready_for_rule_review,
        "ready_for_formal_step30_run": formal_runtime_ready,
        "formal_service_started": False,
        "dependencies_installed": False,
        "frozen_inputs_modified": False,
        "pending_user_decisions": config["pending_user_decisions"],
    }
    (DATA_DIR / "step30_check_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# Phase G9 / Step30 用户应用与部署预检

生成时间：{summary['ended_at']}

## 结论

- `STATUS = {summary['status']}`
- 检查通过：{summary['passed_check_count']}/{summary['check_count']}
- 失败记录：{summary['failed_record_count']}
- `READY_FOR_PHASE_G9_RULE_REVIEW = {str(ready_for_rule_review).upper()}`
- `READY_FOR_FORMAL_STEP30_RUN = {str(formal_runtime_ready).upper()}`

本次将尚未定义的Phase G9限定为用户端应用与部署层的检查阶段。没有启动HTTP服务、没有安装依赖、没有访问外部服务，也没有修改Step27–Step29冻结结果。

## 推荐交付形态

建议先交付仅绑定`127.0.0.1`的本地浏览器应用。底层继续调用Step29六个确定性工具；系统即使不接大模型也必须支持完整的地名选择、路线比较、指标解释和地图导出。大模型只能作为可选自然语言参数解析层。

为满足位置隐私，正式版本默认禁用外部地理编码、外部地图瓦片、请求正文日志和精确位置持久化。地图背景应使用本地数据。

## 环境发现

- ArcPy环境中的确定性后端依赖可用。
- FastAPI和Uvicorn当前未安装，因此检查阶段没有启动Web服务。
- 不建议在未审核的情况下直接修改`arcpy35`环境；建议正式运行前决定是否建立独立应用环境或采用隔离的后端进程方案。

## 待人工确定

1. 是否确认首个正式版本为本机浏览器应用，只监听`127.0.0.1`。
2. 离线地图背景采用简化本地道路底图，还是另行准备本地地图瓦片。
3. 首版是否接入大模型自然语言；若接入，需要确定服务商、模型和密钥保存方式。
4. 是否批准创建独立应用运行环境，避免改变`arcpy35`。

完成以上规则审核前，不得运行Step30正式应用、安装服务依赖或开放网络端口。
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    logger.info("Step30 check status=%s", summary["status"])
    logger.info("READY_FOR_PHASE_G9_RULE_REVIEW=%s", ready_for_rule_review)
    logger.info("READY_FOR_FORMAL_STEP30_RUN=%s", formal_runtime_ready)
    return 0 if ready_for_rule_review else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(check(parse_args().mode))

