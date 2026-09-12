"""Start, stop, or inspect the project-owned local Ollama service."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OLLAMA = ROOT / ".local" / "ollama" / "ollama.exe"
MODELS = ROOT / ".local" / "models" / "ollama"
PID_FILE = ROOT / "routing" / "logs" / "llm_integration" / "ollama.pid"
API = "http://127.0.0.1:11434/api/version"


def health() -> dict:
    try:
        with urllib.request.urlopen(API, timeout=2) as response:
            return {"available": True, **json.loads(response.read().decode("utf-8"))}
    except Exception:
        return {"available": False}


def start() -> int:
    current = health()
    if current["available"]:
        print(f"Ollama already available: {current}")
        return 0
    if not OLLAMA.is_file():
        raise FileNotFoundError(f"Ollama executable not found: {OLLAMA}")
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["OLLAMA_MODELS"] = str(MODELS)
    process = subprocess.Popen(
        [str(OLLAMA), "serve"],
        cwd=str(ROOT),
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    PID_FILE.write_text(str(process.pid), encoding="ascii")
    for _ in range(30):
        time.sleep(0.5)
        current = health()
        if current["available"]:
            print(f"Ollama started: pid={process.pid}, version={current.get('version')}")
            return 0
    process.terminate()
    raise RuntimeError("Ollama did not become ready within 15 seconds")


def stop() -> int:
    if not PID_FILE.is_file():
        print("No project-owned Ollama PID file; external service left untouched.")
        return 0
    pid = int(PID_FILE.read_text(encoding="ascii").strip())
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        PID_FILE.unlink(missing_ok=True)
    print(f"Stopped project-owned Ollama process tree: pid={pid}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop", "status"])
    args = parser.parse_args()
    if args.action == "start":
        return start()
    if args.action == "stop":
        return stop()
    print(json.dumps(health(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
