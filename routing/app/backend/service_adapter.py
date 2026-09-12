"""Memory-only adapter from FastAPI to the deterministic route worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARCPY_PYTHON = Path(
    os.environ.get(
        "THERMAL_ROUTE_TOOL_PYTHON",
        sys.executable,
    )
)


class ServiceAdapter:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()
        self.routes: dict[str, dict[str, Any]] = {}

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not ARCPY_PYTHON.is_file():
            raise RuntimeError("Route worker Python environment is unavailable")
        worker_env = os.environ.copy()
        proxy = ROOT / "routing/data/application/gdal_proxy"
        proxy.mkdir(parents=True, exist_ok=True)
        worker_env["GDAL_PAM_PROXY_DIR"] = str(proxy)
        probe = subprocess.run(
            [str(ARCPY_PYTHON), "-c", "from osgeo import ogr"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
            env=worker_env,
        )
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr.strip() or "GDAL is unavailable in the route worker environment")
        self.process = subprocess.Popen(
            [
                str(ARCPY_PYTHON),
                str(ROOT / "routing/app/backend/tool_worker.py"),
            ],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            env=worker_env,
        )
        ready_line = self.process.stdout.readline()
        if not ready_line:
            raise RuntimeError("Route worker exited before initialization")
        ready = json.loads(ready_line)
        if not ready.get("ready"):
            raise RuntimeError("Route worker failed to initialize")

    def close(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                self._request("shutdown")
            finally:
                self.process.wait(timeout=15)
        self.routes.clear()
        self.process = None

    def _request(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        self.start()
        request_id = uuid.uuid4().hex
        message = {"id": request_id, "operation": operation, **kwargs}
        with self.lock:
            assert self.process and self.process.stdin and self.process.stdout
            self.process.stdin.write(
                json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
            response = json.loads(self.process.stdout.readline())
        if response.get("id") != request_id:
            raise RuntimeError("Tool worker response correlation failed")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error_message", "Tool worker error")))
        return response["result"]

    def tool(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        result = self._request("tool", tool=name, payload=payload)
        if name == "route" and result.get("route_id"):
            self.routes[str(result["route_id"])] = result
        elif name == "compare_routes":
            for route in result.get("routes", []):
                if route.get("route_id"):
                    self.routes[str(route["route_id"])] = route
        return result

    def health(self) -> dict[str, Any]:
        return self._request("health")

    def route_result(self, route_id: str) -> dict[str, Any]:
        if route_id not in self.routes:
            raise ValueError("Unknown or expired route_id")
        return self.routes[route_id]

    def geometry(self, route_result: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "derived_export",
            route_result=route_result,
            format="geometry",
            transient=True,
        )

    def export(self, route_result: dict[str, Any], output_format: str) -> dict[str, Any]:
        return self._request(
            "derived_export",
            route_result=route_result,
            format=output_format,
            transient=False,
        )

    def visual_layers(
        self, bbox: list[float], hour: int, max_buildings: int,
        raster_max_dimension: int = 1200, include_vectors: bool = True,
    ) -> dict[str, Any]:
        return self._request(
            "visual_layers",
            bbox=bbox,
            hour=hour,
            max_buildings=max_buildings,
            raster_max_dimension=raster_max_dimension,
            include_vectors=include_vectors,
        )
