"""Start the local-only server and maintain a PID file for graceful shutdown."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import uvicorn


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PID_PATH = ROOT / "routing" / "logs" / "application" / "server.pid"
STOP_PATH = ROOT / "routing" / "logs" / "application" / "stop.request"


def main() -> int:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    STOP_PATH.unlink(missing_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="ascii")
    config = uvicorn.Config(
        "routing.app.backend.main:app",
        host="127.0.0.1",
        port=8765,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def watch_stop_request() -> None:
        while not server.should_exit:
            if STOP_PATH.exists():
                server.should_exit = True
                return
            time.sleep(0.25)

    watcher = threading.Thread(target=watch_stop_request, daemon=True)
    watcher.start()
    try:
        server.run()
    finally:
        STOP_PATH.unlink(missing_ok=True)
        PID_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
