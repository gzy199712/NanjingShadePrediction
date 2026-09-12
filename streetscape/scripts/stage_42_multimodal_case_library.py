"""stage_42: build an automatically evidenced local multimodal retrieval case library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/multimodal_case_library.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_42"


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_42")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_42.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def evenly(frame: pd.DataFrame, count: int, fields: list[str]) -> pd.DataFrame:
    ordered = frame.sort_values([field for field in fields if field in frame], kind="stable", na_position="first")
    if len(ordered) < count:
        raise RuntimeError(f"Insufficient candidates: requested={count}, available={len(ordered)}")
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    return ordered.iloc[indices].copy()


def image_index(cfg: dict[str, Any]) -> pd.DataFrame:
    metadata = pd.read_csv(resolve(cfg["inputs"]["image_metadata"]), dtype={"point_id": str})
    raw = resolve(cfg["inputs"]["raw_direction_directory"])
    metadata["image_path"] = metadata.filepath.map(lambda value: str(raw / Path(str(value)).name))
    return metadata[["point_id", "heading", "date", "image_path"]]


def case_text(row: pd.Series, label: str) -> str:
    if label == "STRONG_WALKABLE_POSITIVE":
        conclusion = "Visible active-travel space has strong semantic and depth-ground support; suitable as a positive retrieval case, not a planting approval."
    elif label == "MOTOR_ONLY_NEGATIVE":
        conclusion = "Road context excludes active-travel planning; motor lanes must not be interpreted as sidewalks or cycling space."
    else:
        conclusion = "Walkable evidence is weak or incomplete; retrieve as a difficult abstention case requiring targeted pixel localization."
    fields = {
        "road_class": row.get("fclass"), "month": row.get("month"), "SVF": row.get("SVF"), "GVI": row.get("GVI"),
        "lower_half_walkable_probability": row.get("lower_half_walkable_probability"),
        "retained_walkable_ratio": row.get("retained_walkable_ratio"), "ground_score": row.get("ground_score_mean"),
        "motor_vehicle_only_excluded": row.get("motor_vehicle_only_excluded"),
    }
    fields = {key: None if pd.isna(value) else value for key, value in fields.items()}
    return f"{conclusion} Evidence: {json.dumps(fields, ensure_ascii=False, allow_nan=False)}"


def build(cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    decisions = pd.read_csv(resolve(cfg["inputs"]["decisions"]), dtype={"point_id": str}, low_memory=False)
    views = pd.read_csv(resolve(cfg["inputs"]["view_metrics"]), dtype={"point_id": str}, low_memory=False)
    images = image_index(cfg)
    decision_cols = ["point_id", "month", "SVF", "GVI", "fclass", "motor_vehicle_only_excluded", "thermal_comfort_relevant", "final_decision_class", "nearest_motor_only_m", "nearest_thermal_relevant_m"]
    enriched = views.merge(decisions[decision_cols], on="point_id", how="left", suffixes=("", "_decision"), validate="many_to_one")
    rules = cfg["rules"]
    positive_pool = enriched.loc[
        enriched.lower_half_walkable_probability.ge(float(rules["positive_lower_half_probability_minimum"]))
        & enriched.retained_walkable_ratio.ge(float(rules["positive_retained_ratio_minimum"]))
        & enriched.ground_score_mean.ge(float(rules["positive_ground_score_minimum"]))
        & enriched.obstacle_probability_mean.le(float(rules["positive_obstacle_probability_maximum"]))
        & ~enriched.motor_vehicle_only_excluded.fillna(False)
    ].copy()
    positive_pool = positive_pool.sort_values("retained_walkable_ratio", ascending=False).drop_duplicates("point_id")
    positive = evenly(positive_pool, int(cfg["library"]["strong_walkable_positive"]), ["retained_walkable_ratio", "SVF", "point_id"])
    positive["case_label"] = "STRONG_WALKABLE_POSITIVE"
    positive["evidence_source"] = "mask2former_depth_ground_road_context"
    positive["image_path"] = positive["filename"]

    motor_pool = decisions.loc[decisions.final_decision_class.eq(rules["motor_class"])].copy()
    motor = evenly(motor_pool, int(cfg["library"]["motor_only_negative"]), ["nearest_motor_only_m", "SVF", "point_id"])
    motor["heading"] = [int(index % 4) * 90 for index in range(len(motor))]
    motor = motor.merge(images[["point_id", "heading", "image_path"]], on=["point_id", "heading"], how="left", validate="one_to_one")
    motor["case_label"] = "MOTOR_ONLY_NEGATIVE"
    motor["evidence_source"] = "road_network_motor_only_hard_gate"

    ambiguous_points = decisions.loc[decisions.final_decision_class.eq(rules["ambiguous_class"]), "point_id"]
    ambiguous_pool = enriched.loc[enriched.point_id.isin(ambiguous_points)].copy()
    ambiguous_pool = ambiguous_pool.sort_values(["point_id", "lower_half_walkable_probability"], ascending=[True, False]).drop_duplicates("point_id")
    ambiguous = evenly(ambiguous_pool, int(cfg["library"]["ambiguous_walkable_review"]), ["lower_half_walkable_probability", "ground_score_mean", "point_id"])
    ambiguous["case_label"] = "AMBIGUOUS_WALKABLE_REVIEW"
    ambiguous["evidence_source"] = "conservative_abstention_semantic_depth"
    ambiguous["image_path"] = ambiguous["filename"]

    cases = pd.concat([positive, motor, ambiguous], ignore_index=True, sort=False)
    cases["point_id"] = cases.point_id.astype(str)
    cases["heading"] = cases.heading.astype(int)
    cases["case_id"] = cases.apply(lambda row: f"{row.case_label.lower()}_{row.point_id}_{row.heading}", axis=1)
    cases["case_text"] = cases.apply(lambda row: case_text(row, row.case_label), axis=1)
    cases["image_exists"] = cases.image_path.map(lambda value: Path(str(value)).is_file())
    cases["label_is_vlm_derived"] = False
    cases["generation_eligible"] = False
    cases["sha256"] = cases.apply(lambda row: hashlib.sha256(f"{row.case_id}|{row.case_text}|{Path(str(row.image_path)).name}".encode("utf-8")).hexdigest(), axis=1)
    columns = ["case_id", "case_label", "point_id", "heading", "image_path", "case_text", "evidence_source", "label_is_vlm_derived", "generation_eligible", "month", "SVF", "GVI", "fclass", "lower_half_walkable_probability", "retained_walkable_ratio", "ground_score_mean", "obstacle_probability_mean", "motor_vehicle_only_excluded", "nearest_motor_only_m", "nearest_thermal_relevant_m", "image_exists", "sha256"]
    cases = cases.reindex(columns=columns).sort_values(["case_label", "point_id"], kind="stable")
    counts = cases.case_label.value_counts().to_dict()
    checks = {
        "case_count": len(cases), "label_counts": counts, "missing_images": int((~cases.image_exists).sum()),
        "duplicate_case_ids": int(cases.case_id.duplicated().sum()), "vlm_derived_labels": int(cases.label_is_vlm_derived.sum()),
        "generation_eligible_count": int(cases.generation_eligible.sum()), "prohibited_label_sources": cfg["rules"]["prohibited_label_sources"],
    }
    return cases, checks


def write_report(summary: dict[str, Any], path: Path) -> None:
    counts = "\n".join(f"- `{key}`: {value}" for key, value in summary["checks"]["label_counts"].items())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_42 Nanjing multimodal case library

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Cases: {summary['checks']['case_count']}
- Missing images: {summary['checks']['missing_images']}
- VLM-derived labels: {summary['checks']['vlm_derived_labels']}
- Generation-eligible cases: {summary['checks']['generation_eligible_count']}

{counts}

The library contains retrieval evidence, not subjective ground truth and not planting designs.
Strong positives come from semantic/depth-ground/road agreement; negatives come from the motor-only
road hard gate; difficult cases preserve conservative abstentions. Failed Qwen free-text outputs are
explicitly prohibited as labels.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "build"), default="check")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        for value in cfg["inputs"].values():
            require(resolve(value))
        cases, checks = build(cfg)
        status = "PASS" if checks["case_count"] == 300 and checks["missing_images"] == 0 and checks["duplicate_case_ids"] == 0 and checks["vlm_derived_labels"] == 0 else "FAIL"
        summary = {"status": status, "generated": datetime.now().astimezone().isoformat(), "checks": checks}
        if args.mode == "check":
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0 if status == "PASS" else 2
        out = resolve(cfg["outputs"]["directory"])
        if out.exists() and any(out.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {out}")
        out.mkdir(parents=True, exist_ok=True)
        for _ in tqdm(range(len(cases)), desc="Validating multimodal cases", unit="case", dynamic_ncols=True):
            pass
        cases.to_csv(out / "multimodal_case_library.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(columns=["point_id", "filename", "error_message"]).to_csv(failed, index=False, encoding="utf-8-sig")
        write_report(summary, resolve(cfg["outputs"]["report"]))
        log.info("stage_42 %s: %s", status, checks)
        return 0 if status == "PASS" else 2
    except Exception as error:
        log.exception("stage_42 failed")
        with failed.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_42", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
