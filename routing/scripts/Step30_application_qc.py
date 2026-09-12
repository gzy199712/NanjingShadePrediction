"""Independent QC for the formal Phase G9 local application."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "routing/data/application"

from routing.app.backend.main import app


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    checks: list[dict[str, Any]] = []
    performance: list[dict[str, Any]] = []
    summary = json.loads(
        (DATA / "step30_formal_summary.json").read_text(encoding="utf-8")
    )
    checks.append(
        {
            "check": "formal_summary_pass",
            "status": "PASS" if summary["status"] == "PASS" else "FAIL",
            "detail": summary["status"],
        }
    )

    test_files = [
        DATA / "step30_api_test_results.csv",
        DATA / "step30_geocoding_test_results.csv",
        DATA / "step30_routing_test_results.csv",
        DATA / "step30_privacy_test_results.csv",
        DATA / "step30_offline_test_results.csv",
        DATA / "step30_export_test_results.csv",
        DATA / "step30_frontend_e2e_results.csv",
    ]
    test_rows = []
    for path in tqdm(
        test_files,
        desc="Step30 independent test evidence",
        unit="file",
        dynamic_ncols=True,
    ):
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
        test_rows.extend(rows)
        passed = bool(rows) and all(row["status"] == "PASS" for row in rows)
        checks.append(
            {
                "check": f"tests_{path.stem}",
                "status": "PASS" if passed else "FAIL",
                "detail": f"{sum(row['status'] == 'PASS' for row in rows)}/{len(rows)}",
            }
        )

    baseline = list(
        csv.DictReader(
            (DATA / "step30_formal_frozen_input_hashes.csv").open(
                encoding="utf-8-sig", newline=""
            )
        )
    )
    for row in tqdm(
        baseline,
        desc="Step30 independent frozen hashes",
        unit="file",
        dynamic_ncols=True,
    ):
        path = Path(row["filepath"])
        passed = path.is_file() and digest(path) == row["sha256"]
        checks.append(
            {
                "check": f"hash_{path.name}",
                "status": "PASS" if passed else "FAIL",
                "detail": "unchanged" if passed else "changed_or_missing",
            }
        )

    schema = json.loads((DATA / "step30_openapi.json").read_text(encoding="utf-8"))
    required_endpoints = {
        "/api/health",
        "/api/config",
        "/api/geocode",
        "/api/snap",
        "/api/route",
        "/api/compare",
        "/api/summarize",
        "/api/export",
        "/api/visual-layers",
    }
    schema_pass = required_endpoints.issubset(schema["paths"])
    checks.append(
        {
            "check": "openapi_required_endpoints",
            "status": "PASS" if schema_pass else "FAIL",
            "detail": f"{len(required_endpoints & set(schema['paths']))}/{len(required_endpoints)}",
        }
    )

    context = json.loads(
        (ROOT / "routing/app/frontend/assets/local_context.json").read_text(
            encoding="utf-8"
        )
    )
    checks.append(
        {
            "check": "local_vector_context",
            "status": "PASS" if len(context["features"]) == 35534 else "FAIL",
            "detail": str(len(context["features"])),
        }
    )

    source = "\n".join(
        path.read_text(encoding="utf-8")
        for base in [ROOT / "routing/app/backend", ROOT / "routing/app/frontend"]
        for path in base.rglob("*")
        if path.is_file() and path.suffix in {".py", ".js", ".html", ".css"}
    ).lower()
    remote_markers = ["tile.openstreetmap", "nominatim", "googleapis", "https://"]
    checks.append(
        {
            "check": "no_remote_runtime_source",
            "status": "PASS"
            if not any(marker in source for marker in remote_markers)
            else "FAIL",
            "detail": "no remote URL or provider marker",
        }
    )

    logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (ROOT / "routing/logs/application").glob("*.log")
    )
    precise_coordinate = re.search(r"\b(?:118|119)\.\d{5,}\b|\b3[12]\.\d{5,}\b", logs)
    checks.append(
        {
            "check": "logs_contain_no_precise_coordinates",
            "status": "PASS" if precise_coordinate is None else "FAIL",
            "detail": "none" if precise_coordinate is None else "coordinate_like_value_found",
        }
    )

    package_rows = [{"package": "python", "version": platform.python_version()}]
    for package in tqdm(
        [
            "fastapi",
            "uvicorn",
            "pydantic",
            "numpy",
            "pandas",
            "pyproj",
            "shapely",
            "jsonschema",
            "jinja2",
            "python-multipart",
            "tqdm",
        ],
        desc="Step30 environment versions",
        unit="package",
        dynamic_ncols=True,
    ):
        package_rows.append({"package": package, "version": version(package)})
    write_csv(
        DATA / "step30_environment_versions.csv",
        package_rows,
        ["package", "version"],
    )

    with TestClient(app) as client:
        for operation, method, path, payload in [
            ("health", "get", "/api/health", None),
            ("geocode_origin", "post", "/api/geocode", {"query": "新街口", "limit": 5}),
            (
                "geocode_destination",
                "post",
                "/api/geocode",
                {"query": "鼓楼", "limit": 5},
            ),
        ]:
            started = time.perf_counter()
            response = getattr(client, method)(path, json=payload) if payload else client.get(path)
            performance.append(
                {
                    "operation": operation,
                    "elapsed_seconds": time.perf_counter() - started,
                    "status_code": response.status_code,
                    "pass": response.status_code == 200,
                }
            )
        origin_item = client.post(
            "/api/geocode", json={"query": "新街口", "limit": 5}
        ).json()["candidates"][0]
        destination_item = client.post(
            "/api/geocode", json={"query": "鼓楼", "limit": 5}
        ).json()["candidates"][0]
        coordinate = lambda item: {
            "crs": "EPSG:4326",
            "longitude": item["longitude"],
            "latitude": item["latitude"],
        }
        request = {
            "origin": coordinate(origin_item),
            "destination": coordinate(destination_item),
            "hour": 14,
            "mode": "walk",
            "objective": "risk_aware",
            "algorithm": "astar",
            "max_detour_ratio": 0.20,
            "uncertainty_weight": 1.0,
        }
        started = time.perf_counter()
        comparison = client.post(
            "/api/compare",
            json={
                **request,
                "objectives": ["shortest", "shade", "utci", "risk_aware"],
            },
        )
        performance.append(
            {
                "operation": "compare_four_routes",
                "elapsed_seconds": time.perf_counter() - started,
                "status_code": comparison.status_code,
                "pass": comparison.status_code == 200,
            }
        )
        compare_pass = (
            comparison.status_code == 200
            and comparison.json()["route_count"] == 4
            and comparison.json()["all_found"]
        )
        checks.append(
            {
                "check": "independent_four_route_call",
                "status": "PASS" if compare_pass else "FAIL",
                "detail": "4/4" if compare_pass else "failed",
            }
        )

    write_csv(
        DATA / "step30_performance_results.csv",
        performance,
        ["operation", "elapsed_seconds", "status_code", "pass"],
    )
    performance_pass = all(row["pass"] for row in performance)
    checks.append(
        {
            "check": "performance_calls_completed",
            "status": "PASS" if performance_pass else "FAIL",
            "detail": f"{sum(row['pass'] for row in performance)}/{len(performance)}",
        }
    )

    all_pass = all(row["status"] == "PASS" for row in checks)
    write_csv(
        DATA / "step30_independent_qc.csv",
        checks,
        ["check", "status", "detail"],
    )
    result = {
        "step": "Step30_application_independent_qc",
        "status": "PASS" if all_pass else "FAIL",
        "check_count": len(checks),
        "passed_check_count": sum(row["status"] == "PASS" for row in checks),
        "formal_test_evidence_count": len(test_rows),
        "ready_for_llm_integration": all_pass,
    }
    (DATA / "step30_independent_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
