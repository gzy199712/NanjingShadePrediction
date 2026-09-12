"""Validate or rebuild the CEUS submission figure suite."""

from __future__ import annotations

import argparse
import sys

from workflows.runtime import PROJECT_ROOT, Stage, make_logger, record_failure, require_paths, require_writable, run_stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = make_logger("publication_figures")
    try:
        script = PROJECT_ROOT / "streetscape/scripts/stage_36_ceus_figure_suite.py"
        config = PROJECT_ROOT / "streetscape/configs/stage_36_ceus_figure_suite.yaml"
        require_paths([script, config], kind="file")
        require_writable(PROJECT_ROOT / "author_workspace/publication/ceus")
        logger.info("publication figure preflight passed")
        if not args.execute:
            return 0
        # The figure builder is deterministic and overwrites its own output suite.
        command = [sys.executable, str(script)]
        return run_stages("publication_figures", [Stage("ceus_suite", command)], logger)
    except Exception as error:
        logger.exception("workflow failed")
        record_failure("publication_figures", "preflight", "CEUS", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
