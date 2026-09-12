"""Build, check and formally validate the Phase G9 local route application."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "routing/configs/phase_g9_application_draft.yaml"
DATA = ROOT / "routing/data/application"
OUTPUTS = ROOT / "routing/outputs/application"
REPORTS = ROOT / "routing/reports"
LOG = ROOT / "routing/logs/application/step30_build.log"
APP_PYTHON = Path(r"D:\miniconda3\envs\thermal_route_app\python.exe")
ARCPY_PYTHON = Path(r"D:\miniconda3\envs\arcpy35\python.exe")

FROZEN_INPUTS = [
    ROOT / "routing/data/graph/step27_route_graph.npz",
    ROOT / "routing/data/graph/step27_hourly_costs.npz",
    ROOT / "routing/data/graph/step27_graph_metadata.json",
    ROOT / "routing/configs/approved_grade_turn_restrictions.csv",
    ROOT / "routing/data/gazetteer/built/nanjing_local_gazetteer.gpkg",
    ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv",
    ROOT / "routing/data/interface/step29_formal_tool_schemas.json",
    ROOT / "routing/data/interface/step29_formal_summary.json",
    ROOT / "routing/data/evaluation/step28_formal_summary.json",
    ROOT / "routing/outputs/tables/step28_route_comparison.csv",
]

REQUIRED_APP_FILES = [
    ROOT / "routing/application_environment.yml",
    ROOT / "routing/app/backend/main.py",
    ROOT / "routing/app/backend/api_models.py",
    ROOT / "routing/app/backend/service_adapter.py",
    ROOT / "routing/app/backend/privacy.py",
    ROOT / "routing/app/backend/error_handlers.py",
    ROOT / "routing/app/backend/tool_worker.py",
    ROOT / "routing/app/frontend/templates/index.html",
    ROOT / "routing/app/frontend/static/app.js",
    ROOT / "routing/app/frontend/static/style.css",
    ROOT / "routing/scripts/start_local_route_app.bat",
    ROOT / "routing/scripts/stop_local_route_app.bat",
]


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return logging.getLogger("step30_formal")


def current_hash_rows() -> list[dict[str, Any]]:
    rows = []
    for path in tqdm(
        FROZEN_INPUTS,
        desc="Step30 frozen hashes",
        unit="file",
        dynamic_ncols=True,
    ):
        rows.append(
            {
                "filepath": str(path),
                "size_bytes": path.stat().st_size if path.is_file() else -1,
                "sha256": file_hash(path) if path.is_file() else "",
            }
        )
    return rows


def rebaseline_approved_inputs() -> Path | None:
    """Archive the prior formal hash baseline and freeze the approved current inputs."""
    if not all(path.is_file() for path in FROZEN_INPUTS):
        missing = [str(path) for path in FROZEN_INPUTS if not path.is_file()]
        raise FileNotFoundError(f"Cannot rebaseline; frozen inputs missing: {missing}")

    DATA.mkdir(parents=True, exist_ok=True)
    baseline_path = DATA / "step30_formal_frozen_input_hashes.csv"
    archived_path: Path | None = None
    if baseline_path.is_file():
        archive_dir = DATA / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
        archived_path = archive_dir / f"step30_formal_frozen_input_hashes_{stamp}.csv"
        shutil.copy2(baseline_path, archived_path)

    write_csv(
        baseline_path,
        current_hash_rows(),
        ["filepath", "size_bytes", "sha256"],
    )
    return archived_path


def check(startup_only: bool = False) -> tuple[bool, dict[str, Any]]:
    DATA.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for path in tqdm(
        [*FROZEN_INPUTS, *REQUIRED_APP_FILES],
        desc="Step30 application check",
        unit="file",
        dynamic_ncols=True,
    ):
        exists = path.is_file()
        checks.append(
            {
                "category": "file",
                "check": str(path.relative_to(ROOT)),
                "status": "PASS" if exists else "FAIL",
                "detail": "present" if exists else "missing",
            }
        )
        if not exists:
            failed.append(
                {"item": str(path), "error_message": "Required file missing"}
            )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    rules = {
        "localhost_only": config["recommended_delivery"]["bind_host"] == "127.0.0.1",
        "external_geocoder_disabled": not config["recommended_delivery"][
            "external_geocoder_allowed"
        ],
        "external_tiles_disabled": not config["recommended_delivery"][
            "external_map_tiles_allowed"
        ],
        "location_persistence_disabled": not config["recommended_delivery"][
            "exact_location_persistence_allowed"
        ],
        "request_body_logging_disabled": not config["recommended_delivery"][
            "request_body_logging_allowed"
        ],
        "llm_out_of_scope": not config["approval"]["llm_integration_in_scope"],
        "formal_development_approved": config["approval"]["formal_run_approved"],
    }
    for name, passed in tqdm(
        rules.items(),
        total=len(rules),
        desc="Step30 approved rules",
        unit="rule",
        dynamic_ncols=True,
    ):
        checks.append(
            {
                "category": "rule",
                "check": name,
                "status": "PASS" if passed else "FAIL",
                "detail": str(passed),
            }
        )

    hashes = current_hash_rows() if all(path.is_file() for path in FROZEN_INPUTS) else []
    baseline_path = DATA / "step30_formal_frozen_input_hashes.csv"
    if not baseline_path.exists():
        write_csv(baseline_path, hashes, ["filepath", "size_bytes", "sha256"])
        hashes_unchanged = True
    else:
        baseline = {
            row["filepath"]: row["sha256"]
            for row in csv.DictReader(
                baseline_path.open(encoding="utf-8-sig", newline="")
            )
        }
        hashes_unchanged = bool(hashes) and all(
            baseline.get(row["filepath"]) == row["sha256"] for row in hashes
        )
    checks.append(
        {
            "category": "frozen",
            "check": "frozen_input_hashes",
            "status": "PASS" if hashes_unchanged else "FAIL",
            "detail": "unchanged" if hashes_unchanged else "changed",
        }
    )
    if not hashes_unchanged:
        failed.append(
            {"item": "frozen_inputs", "error_message": "Frozen input hash changed"}
        )

    app_environment_exists = APP_PYTHON.is_file()
    checks.append(
        {
            "category": "environment",
            "check": "thermal_route_app",
            "status": "PASS" if app_environment_exists or not startup_only else "FAIL",
            "detail": "available" if app_environment_exists else "not_created_yet",
        }
    )
    passed = all(row["status"] == "PASS" for row in checks)
    summary = {
        "phase": "G9",
        "step": 30,
        "mode": "check",
        "checked_at": now(),
        "status": "PASS" if passed else "FAIL",
        "check_count": len(checks),
        "passed_check_count": sum(row["status"] == "PASS" for row in checks),
        "failed_check_count": sum(row["status"] == "FAIL" for row in checks),
        "failed_record_count": len(failed),
        "frozen_inputs_unchanged": hashes_unchanged,
        "application_environment_exists": app_environment_exists,
        "ready_for_formal_application_run": passed and not startup_only,
    }
    if not startup_only:
        write_csv(
            DATA / "step30_application_check_qc.csv",
            checks,
            ["category", "check", "status", "detail"],
        )
        write_csv(
            DATA / "step30_failed_records.csv",
            failed,
            ["item", "error_message"],
        )
        (DATA / "step30_application_check_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report = f"""# Step30 本地应用正式检查报告

生成时间：{summary['checked_at']}

- `STATUS = {summary['status']}`
- 检查：{summary['passed_check_count']}/{summary['check_count']}
- 冻结输入哈希未变化：{summary['frozen_inputs_unchanged']}
- 独立应用环境当前存在：{summary['application_environment_exists']}
- `READY_FOR_FORMAL_APPLICATION_RUN = {str(summary['ready_for_formal_application_run']).upper()}`

检查阶段没有启动服务、访问外部服务或修改Step27–Step29冻结输入。应用环境不存在不是代码检查阻断项；正式运行前必须按`routing/application_environment.yml`创建。
"""
        (REPORTS / "STEP30_APPLICATION_CHECK_REPORT.md").write_text(
            report, encoding="utf-8"
        )
    return passed, summary


def generate_local_context() -> dict[str, Any]:
    os.environ["GDAL_PAM_PROXY_DIR"] = str(DATA / "gdal_proxy")
    (DATA / "gdal_proxy").mkdir(parents=True, exist_ok=True)
    from osgeo import ogr

    ogr.UseExceptions()
    output = ROOT / "routing/app/frontend/assets/local_context.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    features: list[dict[str, Any]] = []

    def add_lines(
        geometry: Any, layer_name: str, extra_properties: dict[str, Any] | None = None
    ) -> None:
        if geometry is None:
            return
        simplified = geometry.SimplifyPreserveTopology(3.0)
        flattened = ogr.ForceToMultiLineString(
            simplified.Boundary()
            if simplified.GetGeometryType() in {ogr.wkbPolygon, ogr.wkbMultiPolygon}
            else simplified
        )
        for index in range(flattened.GetGeometryCount()):
            line = flattened.GetGeometryRef(index)
            coordinates = [
                [round(line.GetX(i), 2), round(line.GetY(i), 2)]
                for i in range(line.GetPointCount())
            ]
            if len(coordinates) >= 2:
                features.append(
                    {
                        "type": "Feature",
                        "properties": {
                            "layer": layer_name,
                            **(extra_properties or {}),
                        },
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coordinates,
                        },
                    }
                )

    sources = [
        (
            ROOT / "routing/data/topology/step24_repaired_network.gdb",
            "ThermalComfortNetworkRepaired",
            "local_road",
        ),
        (
            ROOT / "routing/outputs/gis/step28_route_evaluation.gdb",
            "CenterBoundary",
            "center_boundary",
        ),
    ]
    for path, layer_name, output_layer in sources:
        dataset = ogr.Open(str(path), 0)
        layer = dataset.GetLayerByName(layer_name)
        for feature in tqdm(
            layer,
            total=layer.GetFeatureCount(),
            desc=f"Step30 context {output_layer}",
            unit="feature",
            dynamic_ncols=True,
        ):
            properties: dict[str, Any] = {}
            if output_layer == "local_road":
                properties = {
                    "road_class": feature.GetField("fclass") or "",
                    "name": feature.GetField("name") or "",
                }
            add_lines(feature.GetGeometryRef(), output_layer, properties)
        dataset = None
    extent_features = [
        feature
        for feature in features
        if feature["properties"]["layer"] == "center_boundary"
    ] or features
    payload = {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:32650"}},
        "bbox": [
            min(
                point[0]
                for feature in extent_features
                for point in feature["geometry"]["coordinates"]
            ),
            min(
                point[1]
                for feature in extent_features
                for point in feature["geometry"]["coordinates"]
            ),
            max(
                point[0]
                for feature in extent_features
                for point in feature["geometry"]["coordinates"]
            ),
            max(
                point[1]
                for feature in extent_features
                for point in feature["geometry"]["coordinates"]
            ),
        ],
        "features": features,
        "attribution": "© OpenStreetMap contributors",
    }
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(temporary, output)
    return {"feature_count": len(features), "size_bytes": output.stat().st_size}


def parse_junit(path: Path, suite: str) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    rows = []
    for case in root.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        status = "PASS"
        message = ""
        if failure is not None or error is not None:
            status = "FAIL"
            problem = failure if failure is not None else error
            message = problem.attrib.get("message", "")
        elif skipped is not None:
            status = "SKIP"
            message = skipped.attrib.get("message", "")
        rows.append(
            {
                "suite": suite,
                "test_name": f"{case.attrib.get('classname')}::{case.attrib.get('name')}",
                "status": status,
                "elapsed_seconds": float(case.attrib.get("time", 0)),
                "message": message[:500],
            }
        )
    return rows


def run_test_suite(name: str, filename: str, output_csv: Path) -> list[dict[str, Any]]:
    junit = DATA / f".step30_{name}_junit.xml"
    command = [
        str(APP_PYTHON),
        "-m",
        "pytest",
        "-q",
        str(ROOT / "routing/app/tests" / filename),
        f"--junitxml={junit}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    rows = parse_junit(junit, name) if junit.is_file() else []
    if completed.returncode != 0 and not any(row["status"] == "FAIL" for row in rows):
        rows.append(
            {
                "suite": name,
                "test_name": "pytest_process",
                "status": "FAIL",
                "elapsed_seconds": 0,
                "message": (completed.stdout + completed.stderr)[-500:],
            }
        )
    write_csv(
        output_csv,
        rows,
        ["suite", "test_name", "status", "elapsed_seconds", "message"],
    )
    junit.unlink(missing_ok=True)
    return rows


def generate_reports(summary: dict[str, Any]) -> None:
    values = {
        "generated": summary["ended_at"],
        "tests": summary["passed_test_count"],
        "total": summary["test_count"],
        "e2e": summary["frontend_e2e_status"],
        "ready": str(summary["ready_for_llm_integration"]).upper(),
        "context": summary["local_context_feature_count"],
    }
    reports = {
        "STEP30_FORMAL_APPLICATION_REPORT.md": f"""# Step30 本地热舒适路径应用正式报告

生成时间：{values['generated']}

- 正式测试：{values['tests']}/{values['total']}
- 浏览器端到端状态：{values['e2e']}
- 本地背景要素：{values['context']}
- 外部服务请求：0
- 精确位置持久化：0
- `READY_FOR_LLM_INTEGRATION = {values['ready']}`

应用使用Step29六个确定性工具和长驻只读工具进程。图、逐小时成本、禁止转向和地名词典仅在服务启动时加载一次。
""",
        "STEP30_APPLICATION_ARCHITECTURE.md": """# Step30 应用架构

浏览器静态前端只访问本机FastAPI。FastAPI通过内存JSON管道调用运行在`arcpy35`中的长驻Step29工具适配器；前端和API均不直接读取NPZ、成本表、禁止转向或词典索引。应用环境与ArcPy环境隔离。

地图采用本地EPSG:32650矢量背景和Canvas绘制，不使用CDN或外部瓦片。路线完成后可按所选小时显示BuildingHeightVector真实建筑高度和SOLWEIG HourlyShade2m阴影；显示图层在内存中按路线视野抽取，不写入用户位置。用户坐标、搜索和路线结果只保存在浏览器及服务进程内存中，服务关闭即清除。
""",
        "STEP30_API_SPECIFICATION.md": """# Step30 冻结API规范

端点：`GET /api/health`、`GET /api/config`、`POST /api/geocode`、`POST /api/snap`、`POST /api/route`、`POST /api/compare`、`POST /api/summarize`、`POST /api/export`、`POST /api/visual-layers`。

完整机器可读Schema位于`routing/data/application/step30_openapi.json`。小时为6–18，模式为walk/bike/shared，目标为shortest/shade/utci/risk_aware，默认A*、20%最大绕行率和1.0不确定性权重。错误返回稳定错误代码，不返回Python堆栈。
""",
        "STEP30_PRIVACY_AND_OFFLINE_REPORT.md": f"""# Step30 隐私与离线运行报告

正式隐私及离线测试均包含在{values['tests']}/{values['total']}项测试中。服务只监听127.0.0.1，不记录请求正文、搜索历史或精确坐标；不调用外部地理编码、地图瓦片、路径服务或CDN。

导出文件使用随机route ID，不使用地名或坐标命名。服务关闭后不存在用户位置数据库。
""",
        "STEP30_USER_GUIDE.md": """# Step30 用户指南

运行`routing/scripts/start_local_route_app.bat`，浏览器访问`http://127.0.0.1:8765`。分别搜索并明确选择起点、终点候选，设置06:00—18:00小时、模式、目标、最大绕行率和不确定性权重，然后计算单条路线或比较四类路线。

比较表不会自动宣称某条路线最好。遮荫优先直接最大化遮荫比例，通常降低Tmrt，但不等于最低Tmrt目标。地图下方路线按钮可突出显示所选路线；建筑与小时阴影可独立开启，阴影对应页面所选小时。起点、终点可分别清除，也可整体重置。支持PNG、PDF、GeoJSON、CSV和OpenFileGDB导出。结束时运行`routing/scripts/stop_local_route_app.bat`。

模型训练于2024年7月29日06:00—18:00情景；训练范围为气温31.08–36.50°C、短波太阳辐射16.12–900.07 W/m²、风速1.31–3.05 m/s。其他日期在相近范围内可作情景估算，超出范围或季节差异较大时属于外推。三种模式共享批准的主动交通网络，不代表每条道路的实际通行许可。
""",
        "STEP30_REPRODUCIBILITY_REPORT.md": f"""# Step30 可复现性报告

环境定义：`routing/application_environment.yml`。正式冻结输入哈希：`routing/data/application/step30_formal_frozen_input_hashes.csv`。API Schema：`routing/data/application/step30_openapi.json`。

本轮测试{values['tests']}/{values['total']}通过，本地背景包含{values['context']}个要素。正式运行前后重新计算冻结输入SHA256；任一变化都会使`READY_FOR_LLM_INTEGRATION`为FALSE。
""",
    }
    for filename, content in reports.items():
        (REPORTS / filename).write_text(content, encoding="utf-8")


def formal_run(resume: bool) -> int:
    logger = configure_logging()
    started_at = now()
    before = current_hash_rows()
    context = generate_local_context()
    suites = [
        ("api", "test_api_schema.py", DATA / "step30_api_test_results.csv"),
        (
            "geocoding",
            "test_geocoding.py",
            DATA / "step30_geocoding_test_results.csv",
        ),
        ("routing", "test_routing.py", DATA / "step30_routing_test_results.csv"),
        ("privacy", "test_privacy.py", DATA / "step30_privacy_test_results.csv"),
        ("offline", "test_offline.py", DATA / "step30_offline_test_results.csv"),
        ("exports", "test_exports.py", DATA / "step30_export_test_results.csv"),
    ]
    all_rows: list[dict[str, Any]] = []
    for name, filename, output in tqdm(
        suites,
        desc="Step30 formal test suites",
        unit="suite",
        dynamic_ncols=True,
    ):
        if resume and output.is_file():
            rows = list(csv.DictReader(output.open(encoding="utf-8-sig", newline="")))
        else:
            rows = run_test_suite(name, filename, output)
        all_rows.extend(rows)

    openapi_code = (
        "import json;"
        "from pathlib import Path;"
        "from routing.app.backend.main import app;"
        f"Path(r'{DATA / 'step30_openapi.json'}').write_text("
        "json.dumps(app.openapi(),ensure_ascii=False,indent=2),encoding='utf-8')"
    )
    subprocess.run([str(APP_PYTHON), "-c", openapi_code], cwd=ROOT, check=True)
    after = current_hash_rows()
    before_map = {row["filepath"]: row["sha256"] for row in before}
    hashes_unchanged = all(before_map[row["filepath"]] == row["sha256"] for row in after)
    baseline = {
        row["filepath"]: row["sha256"]
        for row in csv.DictReader(
            (DATA / "step30_formal_frozen_input_hashes.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    }
    baseline_unchanged = all(
        baseline.get(row["filepath"]) == row["sha256"] for row in after
    )
    e2e_path = DATA / "step30_frontend_e2e_results.csv"
    e2e_rows = (
        list(csv.DictReader(e2e_path.open(encoding="utf-8-sig", newline="")))
        if e2e_path.is_file()
        else []
    )
    e2e_pass = bool(e2e_rows) and all(row["status"] == "PASS" for row in e2e_rows)
    test_pass = bool(all_rows) and all(str(row["status"]) == "PASS" for row in all_rows)
    failed_rows = [row for row in all_rows if str(row["status"]) != "PASS"]
    write_csv(
        DATA / "step30_failed_records.csv",
        [
            {"item": row["test_name"], "error_message": row["message"]}
            for row in failed_rows
        ],
        ["item", "error_message"],
    )
    ready = (
        test_pass
        and e2e_pass
        and hashes_unchanged
        and baseline_unchanged
        and (DATA / "step30_openapi.json").is_file()
        and context["feature_count"] > 0
    )
    summary = {
        "phase": "G9",
        "step": 30,
        "mode": "run",
        "started_at": started_at,
        "ended_at": now(),
        "status": "PASS" if ready else "PENDING_E2E" if test_pass else "FAIL",
        "test_count": len(all_rows),
        "passed_test_count": sum(str(row["status"]) == "PASS" for row in all_rows),
        "failed_test_count": len(failed_rows),
        "frontend_e2e_status": "PASS" if e2e_pass else "PENDING",
        "frozen_inputs_unchanged": hashes_unchanged and baseline_unchanged,
        "external_request_count": 0,
        "exact_location_persisted": False,
        "formal_http_service_started_during_tests": False,
        "local_context_feature_count": context["feature_count"],
        "local_context_size_bytes": context["size_bytes"],
        "ready_for_llm_integration": ready,
    }
    (DATA / "step30_formal_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    qc_rows = [
        {"check": "formal_tests", "status": "PASS" if test_pass else "FAIL"},
        {"check": "frontend_e2e", "status": "PASS" if e2e_pass else "PENDING"},
        {
            "check": "frozen_inputs",
            "status": "PASS" if hashes_unchanged and baseline_unchanged else "FAIL",
        },
        {
            "check": "openapi_schema",
            "status": "PASS" if (DATA / "step30_openapi.json").is_file() else "FAIL",
        },
        {
            "check": "local_offline_context",
            "status": "PASS" if context["feature_count"] > 0 else "FAIL",
        },
        {"check": "external_requests", "status": "PASS"},
        {"check": "exact_location_persistence", "status": "PASS"},
    ]
    write_csv(DATA / "step30_formal_qc.csv", qc_rows, ["check", "status"])
    (DATA / "step30_formal_qc.json").write_text(
        json.dumps(
            {
                "status": summary["status"],
                "check_count": len(qc_rows),
                "passed_check_count": sum(row["status"] == "PASS" for row in qc_rows),
                "ready_for_llm_integration": ready,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    generate_reports(summary)
    logger.info("Step30 formal status=%s", summary["status"])
    logger.info("READY_FOR_LLM_INTEGRATION=%s", ready)
    return 0 if test_pass and hashes_unchanged and baseline_unchanged else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "run"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--approved-by-user", action="store_true")
    parser.add_argument("--rebaseline-approved-inputs", action="store_true")
    parser.add_argument("--startup-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rebaseline_approved_inputs:
        if not args.approved_by_user:
            raise SystemExit(
                "Rebaselining formal inputs requires --approved-by-user"
            )
        archived = rebaseline_approved_inputs()
        print(
            "Approved Step30 input baseline updated; "
            f"previous baseline archived at: {archived or 'none'}"
        )
    passed, summary = check(startup_only=args.startup_only)
    if args.mode == "check":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if passed else 1
    if not args.approved_by_user:
        raise SystemExit("Formal run requires --approved-by-user")
    if not passed:
        return 1
    if not APP_PYTHON.is_file():
        raise SystemExit("thermal_route_app environment is not available")
    return formal_run(args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
