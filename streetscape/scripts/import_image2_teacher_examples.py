"""Match SVIoptimization Image2 teachers to Nanjing points and expose model estimates."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "streetscape/data/generation/stage_69_planner_ready_generation_package"
PERFORMANCE_OUTPUT = ROOT / "outputs/planner_citywide_performance"
ASSET_FIELDS = (
    "source",
    "planner_guidance_overlay",
    "image2_reference",
    "compact_generation_mask",
    "planner_generation_card",
)
BENEFIT_FILES = (
    "point_benefit_summary.csv",
    "point_hour_benefits.csv.gz",
    "summary.json",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evenly_spaced(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("--examples-per-split must be positive")
    rows = sorted(rows, key=lambda row: int(row["point_id"]))
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[len(rows) // 2]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teachers", type=Path, required=True)
    parser.add_argument("--benefits", type=Path, required=True)
    parser.add_argument("--examples-per-split", type=int, default=4)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    benefit_paths = [args.benefits / name for name in BENEFIT_FILES]
    point_metadata = ROOT / "data/training_data/static/point_static_metadata_8975.csv"
    required = [args.teachers, *benefit_paths, point_metadata]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input(s): " + "; ".join(missing))

    teachers = read_csv(args.teachers)
    project_ids = {row["point_id"] for row in read_csv(point_metadata)}
    benefits = {row["point_id"]: row for row in read_csv(benefit_paths[0])}
    matched = [row for row in teachers if row["point_id"] in project_ids]
    duplicate_keys = len(matched) - len({(row["point_id"], row["heading"]) for row in matched})
    missing_assets = sum(
        not Path(row[field]).is_file()
        for row in matched
        for field in ASSET_FIELDS
    )
    evaluable = [
        row for row in matched
        if benefits.get(row["point_id"], {}).get("overall_report_status")
        == "estimated_standardized_intervention"
    ]
    check = {
        "status": (
            "PASS"
            if len(matched) == len(teachers) and not duplicate_keys and not missing_assets
            else "FAIL"
        ),
        "teacher_pairs": len(teachers),
        "matched_pairs": len(matched),
        "unique_points": len({row["point_id"] for row in matched}),
        "duplicate_point_heading_keys": duplicate_keys,
        "missing_assets": missing_assets,
        "thermally_evaluable_pairs": len(evaluable),
    }
    if args.check:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["status"] == "PASS" else 2
    if check["status"] != "PASS":
        raise RuntimeError(check)
    if OUTPUT.exists() and any(OUTPUT.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {OUTPUT}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    joined = [
        {
            **row,
            **{
                f"thermal_{key}": value
                for key, value in benefits[row["point_id"]].items()
                if key != "point_id"
            },
        }
        for row in matched
    ]
    write_csv(OUTPUT / "planner_ready_manifest.csv", matched)
    write_csv(OUTPUT / "teacher_model_pre_evaluation.csv", joined)

    examples = []
    for split in ("train", "val", "test"):
        split_rows = [row for row in evaluable if row["split"] == split]
        examples.extend(evenly_spaced(split_rows, args.examples_per_split))
    local_examples = []
    for row in examples:
        target = OUTPUT / "examples" / row["pair_id"]
        target.mkdir(parents=True, exist_ok=True)
        copied = dict(row)
        for field in ASSET_FIELDS:
            destination = target / Path(row[field]).name
            shutil.copy2(row[field], destination)
            copied[field] = str(destination.resolve())
        local_examples.append(copied)
    write_csv(OUTPUT / "example_manifest.csv", local_examples)

    PERFORMANCE_OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in BENEFIT_FILES:
        shutil.copy2(args.benefits / name, PERFORMANCE_OUTPUT / name)

    def aggregate(field: str) -> dict[str, float]:
        values = [
            float(benefits[row["point_id"]][field])
            for row in evaluable
            if benefits[row["point_id"]][field]
        ]
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
        }

    summary = {
        **check,
        "example_pairs": len(local_examples),
        "example_pairs_by_split": {
            split: sum(row["split"] == split for row in local_examples)
            for split in ("train", "val", "test")
        },
        "thermal_estimate": "five-seed SOLWEIG-calibrated standardized intervention response",
        "evaluable_performance": {
            "mean_effective_shade_gain": aggregate("mean_effective_shade_gain"),
            "mean_delta_tmrt_c": aggregate("mean_estimated_delta_tmrt_c"),
            "mean_delta_utci_c": aggregate("mean_estimated_delta_utci_c"),
            "hour_14_effective_shade_gain": aggregate("hour_14_effective_shade_gain"),
            "hour_14_delta_tmrt_c": aggregate("hour_14_estimated_delta_tmrt_c"),
            "hour_14_delta_utci_c": aggregate("hour_14_estimated_delta_utci_c"),
        },
        "claim_boundary": "Planning estimate; not measured or site-specific causal effect.",
        "external_teacher_assets_are_read_only_references": True,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
