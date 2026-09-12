"""Validate, maintain or serve the current walking/cycling shade-routing system."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from workflows.runtime import PROJECT_ROOT, Stage, make_logger, record_failure, require_paths, require_writable, run_stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "repair-topology", "serve"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--arcpy-python", type=Path, default=Path(r"D:\miniconda3\envs\arcpy35\python.exe"))
    args = parser.parse_args()
    logger = make_logger("routing_navigation")
    try:
        require_paths(
            [
                PROJECT_ROOT / "routing/app/backend/main.py",
                PROJECT_ROOT / "routing/data/graph/step27_route_graph.npz",
                PROJECT_ROOT / "routing/data/graph/step27_hourly_costs.npz",
                PROJECT_ROOT / "routing/data/graph/step27_graph_metadata.json",
                PROJECT_ROOT / "routing/data/gazetteer/built/gazetteer_aliases.csv",
            ],
            kind="file",
        )
        require_writable(PROJECT_ROOT / "routing/logs")
        if args.action == "check":
            logger.info("routing preflight passed")
            return 0
        if args.action == "repair-topology":
            script = PROJECT_ROOT / "routing/scripts/Step24g_apply_endpoint_extension_repairs.py"
            require_paths([script, args.arcpy_python], kind="file")
            probe = subprocess.run([str(args.arcpy_python), "-c", "import arcpy; print(arcpy.GetInstallInfo()['Version'])"], text=True, capture_output=True, check=False)
            if probe.returncode != 0:
                raise RuntimeError(probe.stderr.strip() or "arcpy35 import failed")
            command = [str(args.arcpy_python), str(script), "--apply"]
            if args.overwrite:
                command.append("--overwrite")
            return run_stages("routing_navigation", [Stage("repair_topology", command)], logger)
        command = [sys.executable, "-m", "uvicorn", "routing.app.backend.main:app", "--host", "127.0.0.1", "--port", "8765"]
        return run_stages("routing_navigation", [Stage("serve", command)], logger)
    except Exception as error:
        logger.exception("workflow failed")
        record_failure("routing_navigation", "preflight", args.action, error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
