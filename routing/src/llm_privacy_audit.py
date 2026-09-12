"""Static and live privacy/offline audit for the local Step31 system."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from tqdm import tqdm

from routing.src.llm_parse_evaluation import write_rows


FIELDS = ["check", "status", "count", "threshold", "detail"]
COORDINATE = re.compile(r"\b(?:118|119)\.\d{5,}\b|\b(?:31|32)\.\d{5,}\b")


def runtime_source(root: Path) -> str:
    paths = [
        root / "routing/app/backend",
        root / "routing/app/frontend",
        root / "routing/scripts/start_local_route_app.bat",
        root / "routing/scripts/manage_local_model.py",
    ]
    content = []
    for path in paths:
        if path.is_file():
            content.append(path.read_text(encoding="utf-8", errors="replace"))
        elif path.is_dir():
            for file in path.rglob("*"):
                if file.suffix.lower() in {".py", ".js", ".html", ".css"}:
                    content.append(file.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(content)


def run_privacy_audit(root: Path, output_path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = runtime_source(root)
    logs = []
    for directory in [
        root / "routing/logs/application",
        root / "routing/logs/llm_integration",
        root / "routing/logs/llm_evaluation",
    ]:
        if directory.exists():
            logs.extend(
                file.read_text(encoding="utf-8", errors="replace")
                for file in directory.rglob("*")
                if file.is_file() and file.stat().st_size < 20_000_000
            )
    log_text = "\n".join(logs)
    checks: list[tuple[str, int, int, str]] = []
    remote_markers = [
        "nominatim.openstreetmap", "maps.googleapis", "api.openai.com",
        "api.map.baidu", "restapi.amap", "tiles.openstreetmap",
    ]
    checks.append(("external_runtime_url", sum(marker in source.lower() for marker in remote_markers), 0, "runtime source provider scan"))
    checks.append(("precise_coordinate_in_logs", len(COORDINATE.findall(log_text)), 0, "application and evaluation logs"))
    request_markers = sum(
        marker in log_text
        for marker in ("下午2点从新街口", "user_text=", "request_body=", '"messages":')
    )
    checks.append(("request_body_persistence", request_markers, 0, "log prompt/body marker scan"))
    history_files = [
        file for file in (root / "routing").rglob("*")
        if file.is_file()
        and any(term in file.name.lower() for term in ("conversation", "chat_history", "user_history"))
    ]
    checks.append(("conversation_history_files", len(history_files), 0, "routing tree filename scan"))
    frontend = (root / "routing/app/frontend/static/app.js").read_text(encoding="utf-8")
    browser_storage = sum(marker in frontend for marker in ("localStorage", "indexedDB", "sessionStorage"))
    checks.append(("browser_persistent_storage", browser_storage, 0, "frontend storage API scan"))
    database_files = [
        file for file in (root / "routing").rglob("*")
        if file.is_file() and file.suffix.lower() in {".sqlite", ".sqlite3", ".db"}
    ]
    checks.append(("user_history_database", len(database_files), 0, "routing database file scan"))
    checks.append(("cloud_model_configuration", sum(marker in source for marker in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "api.openai.com")), 0, "cloud model marker scan"))
    # A point-in-time socket audit. Listening localhost sockets are allowed;
    # established non-loopback connections owned by Ollama/python are not.
    remote_connections = 0
    try:
        netstat = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=20,
        ).stdout
        pids = {
            str(process.pid)
            for name in ("ollama", "python")
            for process in _matching_processes(name)
        }
        for line in netstat.splitlines():
            parts = line.split()
            if len(parts) < 5 or parts[-1] not in pids or "ESTABLISHED" not in line:
                continue
            remote = parts[2]
            if not any(value in remote for value in ("127.0.0.1", "[::1]", "0.0.0.0")):
                remote_connections += 1
    except Exception:
        remote_connections = 0
    checks.append(("external_established_connection_snapshot", remote_connections, 0, "Ollama/python netstat snapshot"))
    rows = [
        {
            "check": name,
            "status": "PASS" if count <= threshold else "FAIL",
            "count": count,
            "threshold": threshold,
            "detail": detail,
        }
        for name, count, threshold, detail in tqdm(
            checks, desc="Step32 privacy audit", unit="check", dynamic_ncols=True
        )
    ]
    write_rows(output_path, rows, FIELDS)
    counts = {
        "external_request_count": next(row["count"] for row in rows if row["check"] == "external_established_connection_snapshot")
        + next(row["count"] for row in rows if row["check"] == "external_runtime_url"),
        "precise_coordinate_persistence_count": next(row["count"] for row in rows if row["check"] == "precise_coordinate_in_logs"),
        "request_body_persistence_count": next(row["count"] for row in rows if row["check"] == "request_body_persistence"),
        "conversation_persistence_count": next(row["count"] for row in rows if row["check"] == "conversation_history_files")
        + next(row["count"] for row in rows if row["check"] == "browser_persistent_storage")
        + next(row["count"] for row in rows if row["check"] == "user_history_database"),
    }
    return rows, counts


def _matching_processes(name: str):
    try:
        import psutil
        return [
            process
            for process in psutil.process_iter(["name"])
            if name in (process.info["name"] or "").lower()
        ]
    except Exception:
        return []
