"""Build multi-weather training tables from prepared spatial features.

Checks are executed before any stage. Original data is read-only by contract.
Example: python -m workflows.data_pipeline --stages 112 113 114 115 --execute
"""

from __future__ import annotations

import argparse
import sys
from workflows.runtime import PROJECT_ROOT, Stage, make_logger, record_failure, require_paths, require_writable, run_stages


SCRIPT_DIR = PROJECT_ROOT / "data_pipeline" / "scripts"
SCRIPT_NAMES = {
    "112": "Step112_era5_summer_weather_selection.py",
    "113": "Step113_solweig_gpu_multiweather.py", "114": "Step114_build_multidate_training_tables.py",
    "115": "Step115_spatiotemporal_training_preflight.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", required=True, choices=tuple(SCRIPT_NAMES))
    parser.add_argument("--execute", action="store_true", help="Run after preflight; otherwise only validate")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable, help="Interpreter used by the selected stages")
    args = parser.parse_args()
    logger = make_logger("data_pipeline")
    try:
        scripts = [SCRIPT_DIR / SCRIPT_NAMES[index] for index in args.stages]
        require_paths(scripts, kind="file")
        require_writable(PROJECT_ROOT / "data" / "training_data")
        logger.info("preflight passed stages=%s", args.stages)
        if not args.execute:
            return 0
        stages = []
        for index, script in zip(args.stages, scripts):
            command = [args.python, str(script)]
            if index in {"112", "113", "114"}:
                command.extend(["--mode", "run", "--approved-by-user"])
                if args.overwrite and index in {"112", "114"}:
                    command.append("--overwrite")
            elif index == "115":
                command.extend(["--mode", "run"])
                if args.overwrite:
                    command.append("--overwrite")
            stages.append(Stage(f"step{index}", command))
        return run_stages("data_pipeline", stages, logger)
    except Exception as error:
        logger.exception("preflight failed")
        record_failure("data_pipeline", "preflight", ",".join(args.stages), error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
