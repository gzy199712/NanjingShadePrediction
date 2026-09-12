"""Image2 streetscape generation interface.

The generator is external. This script keeps the local contract boring:
prepare/check a case manifest, ask Image2 to generate from source.png +
planning_overlay.png + prompt.txt, then register the returned image.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_ROOT = ROOT / "streetscape/data/generation/image2_cases"


def read_rows(manifest: Path) -> list[dict[str, str]]:
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(manifest: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise RuntimeError("manifest has no rows")
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def image_path(case_root: Path, pair_id: str) -> Path:
    return case_root / pair_id / "gpt-image-2.png"


def status(case_root: Path) -> dict[str, object]:
    if not (case_root / "pair_manifest.csv").is_file():
        return {
            "case_root": str(case_root.resolve()),
            "status": "NOT_PREPARED",
            "message": "Prepare pair_manifest.csv and its case folders first.",
            "total": 0,
            "complete": 0,
            "first_missing": None,
        }
    rows = read_rows(case_root / "pair_manifest.csv")
    complete = sum(1 for row in rows if image_path(case_root, row["pair_id"]).is_file())
    first_missing = next((row["pair_id"] for row in rows if not image_path(case_root, row["pair_id"]).is_file()), None)
    return {"case_root": str(case_root.resolve()), "status": "READY", "total": len(rows), "complete": complete, "first_missing": first_missing}


def next_case(case_root: Path) -> dict[str, str] | None:
    for row in read_rows(case_root / "pair_manifest.csv"):
        case = case_root / row["pair_id"]
        if not (case / "gpt-image-2.png").is_file():
            return {
                "pair_id": row["pair_id"],
                "source": str((case / "source.png").resolve()),
                "planning_overlay": str((case / "planning_overlay.png").resolve()),
                "prompt": (case / "prompt.txt").read_text(encoding="utf-8"),
                "expected_output": str((case / "gpt-image-2.png").resolve()),
            }
    return None


def register(case_root: Path, pair_id: str, generated_image: Path) -> dict[str, str]:
    rows = read_rows(case_root / "pair_manifest.csv")
    pair_ids = {row["pair_id"] for row in rows}
    if pair_id not in pair_ids:
        raise KeyError(pair_id)
    if not generated_image.is_file():
        raise FileNotFoundError(generated_image)
    target = image_path(case_root, pair_id)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(generated_image, target)
    for row in rows:
        if row["pair_id"] == pair_id:
            row["status"] = "COMPLETE"
            row["teacher_output_path"] = str(target.resolve())
            break
    write_rows(case_root / "pair_manifest.csv", rows)
    return {"pair_id": pair_id, "status": "COMPLETE", "teacher_output_path": str(target.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=DEFAULT_CASE_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    sub.add_parser("next")
    done = sub.add_parser("register")
    done.add_argument("--pair-id", required=True)
    done.add_argument("--generated-image", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(status(args.case_root), ensure_ascii=False, indent=2))
    elif args.command == "next":
        print(json.dumps(next_case(args.case_root), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(register(args.case_root, args.pair_id, args.generated_image), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
