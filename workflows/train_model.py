"""Train, evaluate and finalize the current multi-date thermal model on GPU."""

from __future__ import annotations

import argparse
import sys

from workflows.runtime import PROJECT_ROOT, Stage, make_logger, record_failure, require_cuda, require_paths, require_writable, run_stages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    logger = make_logger("model_training")
    try:
        manifest = PROJECT_ROOT / "data/training_data/multiweather/step115_spatiotemporal_split/training_manifest_spatiotemporal.csv"
        scripts = PROJECT_ROOT / "training/scripts"
        required = [
            scripts / "Step117_train_multidate_model.py",
            scripts / "Step118_evaluate_multidate_test.py",
            scripts / "Step119_multidate_evaluation_reporting.py",
            scripts / "Step124_finalize_multidate_model.py",
        ]
        require_paths(required, kind="file")
        require_paths([manifest], kind="file")
        require_cuda()
        require_writable(PROJECT_ROOT / "training" / "checkpoints")
        logger.info("preflight passed; CUDA-only execution is enforced")
        if not args.execute:
            return 0
        extra = ["--overwrite"] if args.overwrite else []
        stages = [
            Stage("train", [sys.executable, str(required[0]), "--mode", "run", "--approved-by-user", *extra]),
            Stage("evaluate", [sys.executable, str(required[1]), "--mode", "run", "--approved-by-user", *extra]),
            Stage("report", [sys.executable, str(required[2]), "--mode", "run", *extra]),
        ]
        if args.include_ablation:
            ablations = {
                121: ("train_ablations", scripts / "Step121_train_multidate_ablations.py"),
                122: ("evaluate_ablations", scripts / "Step122_evaluate_multidate_ablations.py"),
                123: ("report_ablations", scripts / "Step123_ablation_reporting.py"),
            }
            require_paths([path for _, path in ablations.values()], kind="file")
            for number, (name, path) in ablations.items():
                command = [sys.executable, str(path), "--mode", "run"]
                if number in (121, 122):
                    command.append("--approved-by-user")
                command.extend(extra)
                stages.append(Stage(name, command))
        stages.append(Stage("finalize", [sys.executable, str(required[3]), "--mode", "run", *extra]))
        return run_stages("model_training", stages, logger)
    except Exception as error:
        logger.exception("workflow failed")
        record_failure("model_training", "preflight", "multidate training", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
