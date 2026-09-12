"""Memory-only adapter for GPU streetscape shade assessment."""

from __future__ import annotations

import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
PYTHON = Path(r"D:\miniconda3\envs\gptthermalcomfort\python.exe")


class PlannerAnalysisAdapter:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self.process = subprocess.Popen(
            [str(PYTHON), str(ROOT / "routing/app/backend/planner_analysis_worker.py")],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert self.process.stdout
        ready = json.loads(self.process.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError("GPU streetscape analysis worker failed to initialize")

    def analyze(
        self, image_path: Path, planner_intent: str, capture_month: int | None = None,
        confidence: int = 65, road_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.start()
        request_id = uuid.uuid4().hex
        message = {
            "id": request_id,
            "operation": "analyze",
            "image_path": str(image_path),
            "planner_intent": planner_intent,
            "capture_month": capture_month,
            "confidence": max(0, min(100, int(confidence))),
            "road_context": road_context or {},
        }
        with self.lock:
            assert self.process and self.process.stdin and self.process.stdout
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = json.loads(self.process.stdout.readline())
        if response.get("id") != request_id:
            raise RuntimeError("Planner worker response correlation failed")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error_message", "GPU analysis failed")))
        return response["result"]

    def analyze_directions(
        self,
        directions: list[dict[str, Any]],
        panorama_path: Path,
        planner_intent: str,
        capture_month: int | None = None,
        confidence: int = 65,
        road_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.start()
        request_id = uuid.uuid4().hex
        message = {
            "id": request_id,
            "operation": "analyze_directions",
            "directions": [
                {"heading": int(item["heading"]), "image_path": str(Path(item["image_path"]).resolve())}
                for item in directions
            ],
            "panorama_path": str(panorama_path.resolve()),
            "planner_intent": planner_intent,
            "capture_month": capture_month,
            "confidence": max(0, min(100, int(confidence))),
            "road_context": road_context or {},
        }
        with self.lock:
            assert self.process and self.process.stdin and self.process.stdout
            self.process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
            response = json.loads(self.process.stdout.readline())
        if response.get("id") != request_id:
            raise RuntimeError("Planner worker response correlation failed")
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error_message", "GPU directional analysis failed")))
        return response["result"]

    def close(self) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                assert self.process.stdin
                self.process.stdin.write(json.dumps({"id": "close", "operation": "shutdown"}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=15)
            except Exception:
                self.process.terminate()
        self.process = None
