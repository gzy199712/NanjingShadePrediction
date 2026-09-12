"""Independent QC for the formal Step29 route interface."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "routing" / "data" / "interface"
os.environ["GDAL_PAM_PROXY_DIR"] = str(DATA / "gdal_proxy")

from osgeo import ogr

ogr.UseExceptions()


def main() -> int:
    summary = json.loads(
        (DATA / "step29_formal_summary.json").read_text(encoding="utf-8")
    )
    schemas = json.loads(
        (DATA / "step29_formal_tool_schemas.json").read_text(encoding="utf-8")
    )
    tables = {
        "tools": pd.read_csv(DATA / "step29_formal_tool_qc.csv"),
        "geocoder": pd.read_csv(DATA / "step29_formal_geocoder_qc.csv"),
        "routes": pd.read_csv(DATA / "step29_formal_route_qc.csv"),
        "algorithms": pd.read_csv(DATA / "step29_formal_algorithm_qc.csv"),
        "exports": pd.read_csv(DATA / "step29_formal_export_qc.csv"),
        "privacy": pd.read_csv(DATA / "step29_formal_privacy_qc.csv"),
    }
    for table in tqdm(
        tables.values(),
        total=len(tables),
        desc="Checking Step29 QC tables",
        unit="table",
        dynamic_ncols=True,
    ):
        table["pass"] = table["pass"].astype(str).str.lower().eq("true")

    export_checks = []
    for row in tqdm(
        tables["exports"].itertuples(index=False),
        total=len(tables["exports"]),
        desc="Checking Step29 exports",
        unit="export",
        dynamic_ncols=True,
    ):
        path = Path(row.output_path)
        detail = ""
        passed = path.exists()
        if row.format == "file_geodatabase" and passed:
            dataset = ogr.Open(str(path), 0)
            layer = dataset.GetLayerByName("Route") if dataset else None
            count = layer.GetFeatureCount() if layer else -1
            epsg = (
                layer.GetSpatialRef().GetAuthorityCode(None)
                if layer is not None
                else None
            )
            detail = f"count={count};epsg={epsg}"
            passed = count == 1 and epsg == "32650"
            dataset = None
        elif passed:
            detail = f"bytes={path.stat().st_size}"
            passed = path.stat().st_size > 0
        export_checks.append(
            {"format": row.format, "detail": detail, "pass": passed}
        )

    cli_payload = DATA / "step29_cli_qc_request.json"
    cli_payload.write_text(
        json.dumps({"query": "新街口", "city": "南京市", "limit": 3}, ensure_ascii=False),
        encoding="utf-8",
    )
    cli = subprocess.run(
        [
            r"D:\miniconda3\envs\arcpy35\python.exe",
            str(ROOT / "routing" / "scripts" / "Step29_route_cli.py"),
            "geocode_place",
            "--json",
            str(cli_payload),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    cli_payload.unlink(missing_ok=True)
    cli_result = json.loads(cli.stdout) if cli.returncode == 0 else {}

    forbidden_columns = {
        "origin_x",
        "origin_y",
        "destination_x",
        "destination_y",
        "longitude",
        "latitude",
    }
    persisted_coordinate_columns = sorted(
        {
            column
            for table in tables.values()
            for column in table.columns
            if column.lower() in forbidden_columns
        }
    )
    checks = [
        ("formal_summary_pass", summary["status"], summary["status"] == "PASS"),
        ("schema_tool_count", len(schemas["tools"]), len(schemas["tools"]) == 6),
        ("tool_rows_pass", len(tables["tools"]), tables["tools"]["pass"].all()),
        ("geocoder_rows_pass", len(tables["geocoder"]), tables["geocoder"]["pass"].all()),
        ("route_rows_pass", len(tables["routes"]), tables["routes"]["pass"].all()),
        ("algorithm_rows_pass", len(tables["algorithms"]), tables["algorithms"]["pass"].all()),
        ("privacy_rows_pass", len(tables["privacy"]), tables["privacy"]["pass"].all()),
        ("export_rows_pass", len(export_checks), all(row["pass"] for row in export_checks)),
        ("failed_record_count", summary["failed_record_count"], summary["failed_record_count"] == 0),
        ("external_request_made", summary["external_request_made"], not summary["external_request_made"]),
        ("exact_location_persisted", summary["exact_location_persisted"], not summary["exact_location_persisted"]),
        ("frozen_inputs_modified", summary["frozen_inputs_modified"], not summary["frozen_inputs_modified"]),
        ("http_service_started", summary["formal_http_service_started"], not summary["formal_http_service_started"]),
        ("persisted_coordinate_columns", ",".join(persisted_coordinate_columns), not persisted_coordinate_columns),
        ("cli_return_code", cli.returncode, cli.returncode == 0),
        (
            "cli_local_geocoder_candidates",
            cli_result.get("candidate_count", 0),
            cli_result.get("candidate_count", 0) > 0
            and not cli_result.get("external_request_made", True),
        ),
    ]
    qc = pd.DataFrame(checks, columns=["check", "value", "pass"])
    qc.to_csv(
        DATA / "step29_formal_independent_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(export_checks).to_csv(
        DATA / "step29_formal_export_independent_qc.csv",
        index=False,
        encoding="utf-8-sig",
    )
    passed = bool(qc["pass"].all())
    result = {
        "step": "Step29_formal_interface_independent_qc",
        "checked_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": "PASS" if passed else "FAIL",
        "check_count": len(qc),
        "passed_check_count": int(qc["pass"].sum()),
        "export_check_count": len(export_checks),
        "ready_for_phase_g9": passed,
    }
    (DATA / "step29_formal_independent_qc.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
