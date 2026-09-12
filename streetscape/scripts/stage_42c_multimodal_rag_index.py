"""stage_42c: build and verify the 300-case full-CUDA multimodal RAG index."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streetscape.scripts.stage_42b_qwen3vl_embedding_preflight import (  # noqa: E402
    OfficialQwen3VLEmbedder,
    assert_cuda,
)


CONFIG = ROOT / "streetscape/configs/multimodal_rag_index.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_42c"


def resolve(value: str) -> Path:
    return ROOT / value


def make_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_42c")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(LOG_DIR / "stage_42c.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        log.addHandler(handler)
    return log


def load_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def retrieval_metrics(embeddings: np.ndarray, labels: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    order = np.argsort(-similarity, axis=1)[:, :5]
    rows = []
    top1_matches, top5_precisions = [], []
    for index, neighbors in enumerate(order):
        matches = labels[neighbors] == labels[index]
        top1_matches.append(bool(matches[0]))
        top5_precisions.append(float(matches.mean()))
        rows.append(
            {
                "case_id": index,
                "query_label": labels[index],
                "top1_label": labels[neighbors[0]],
                "top1_similarity": float(similarity[index, neighbors[0]]),
                "top1_match": bool(matches[0]),
                "top5_label_precision": float(matches.mean()),
                "neighbor_indices": "|".join(str(int(value)) for value in neighbors),
            }
        )
    metrics = {
        "top1_label_accuracy": float(np.mean(top1_matches)),
        "top5_label_precision": float(np.mean(top5_precisions)),
    }
    return metrics, pd.DataFrame(rows)


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# stage_42c multimodal RAG index

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Indexed cases: {summary['case_count']}
- Classes: {summary['class_counts']}
- Embedding dimension: {summary['embedding_dimension']}
- Checkpoint integrity: {summary['weight_integrity_verified']}
- Full GPU: {summary['full_gpu_verified']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB
- Total / mean encoding time: {summary['total_encode_seconds']:.2f} s / {summary['mean_encode_seconds']:.3f} s
- Leave-one-out top-1 label accuracy: {summary['top1_label_accuracy']:.1%}
- Leave-one-out top-5 label precision: {summary['top5_label_precision']:.1%}

This index is a local retrieval evidence layer. Labels are deterministic semantic/depth/road
evidence states, not VLM-generated truth. Retrieval can support an agent decision but cannot by
itself unlock image generation or override motor-only and geometric safety gates.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "run"), default="check")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = make_logger()
    failed_path = LOG_DIR / "failed_files.csv"
    try:
        cfg = load_config()
        library_path = resolve(cfg["inputs"]["case_library"])
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        frame = pd.read_csv(library_path, dtype={"point_id": str})
        expected = int(cfg["quality_gates"]["expected_cases"])
        images_ok = frame.image_path.map(lambda value: Path(value).is_file())
        checks = {
            "case_count": len(frame),
            "unique_case_ids": int(frame.case_id.nunique()),
            "images_present": int(images_ok.sum()),
            "cuda_available": torch.cuda.is_available(),
        }
        checks["status"] = "PASS" if (
            len(frame) == expected
            and frame.case_id.nunique() == expected
            and images_ok.all()
            and torch.cuda.is_available()
        ) else "FAIL"
        if args.mode == "check":
            print(json.dumps(checks, ensure_ascii=False, indent=2))
            return 0 if checks["status"] == "PASS" else 2
        if checks["status"] != "PASS":
            raise RuntimeError(f"Preflight failed: {checks}")
        if args.offline:
            import os

            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        output = resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        embedder = OfficialQwen3VLEmbedder(
            cfg["model"]["model_id"], cfg["model"]["instruction"], args.offline
        )
        loading = embedder.loading_info
        key_counts = {
            name: len(loading.get(name, []))
            for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        }
        if any(key_counts.values()):
            raise RuntimeError(f"Checkpoint integrity failure: {key_counts}")
        placement = assert_cuda(embedder.model)
        started = time.perf_counter()
        vectors, failures = [], []
        for row in tqdm(frame.to_dict("records"), desc="Indexing multimodal cases", unit="case", dynamic_ncols=True):
            try:
                vectors.append(
                    embedder.encode(
                        row["case_text"], row["image_path"], int(cfg["model"]["output_dimension"])
                    )
                )
            except Exception as error:
                failures.append(
                    {"point_id": row["point_id"], "filename": row["image_path"], "error_message": f"{type(error).__name__}: {error}"}
                )
                vectors.append(np.full(int(cfg["model"]["output_dimension"]), np.nan, dtype=np.float32))
        elapsed = time.perf_counter() - started
        if failures:
            raise RuntimeError(f"{len(failures)} case embeddings failed")
        embeddings = np.stack(vectors).astype(np.float32)
        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embedding matrix contains non-finite values")
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError("Embedding normalization gate failed")
        metrics, evaluation = retrieval_metrics(embeddings, frame.case_label.to_numpy())
        evaluation["case_id"] = frame.case_id.to_numpy()
        peak = torch.cuda.max_memory_reserved() / 1024**3
        gates_pass = (
            metrics["top1_label_accuracy"] >= float(cfg["quality_gates"]["minimum_top1_label_accuracy"])
            and metrics["top5_label_precision"] >= float(cfg["quality_gates"]["minimum_top5_label_precision"])
            and peak <= float(cfg["runtime"]["maximum_peak_vram_gib"])
        )
        status = "INDEX_PASS" if gates_pass else "RETRIEVAL_SIGNAL_FAIL"
        np.save(output / "case_embeddings.npy", embeddings)
        metadata_columns = [
            "case_id", "case_label", "point_id", "heading", "image_path", "case_text",
            "evidence_source", "generation_eligible", "sha256",
        ]
        frame[metadata_columns].to_csv(output / "case_index.csv.gz", index=False, compression="gzip", encoding="utf-8")
        evaluation.to_csv(output / "leave_one_out_retrieval.csv.gz", index=False, compression="gzip", encoding="utf-8")
        summary = {
            "status": status,
            "generated": datetime.now().astimezone().isoformat(),
            "model_id": cfg["model"]["model_id"],
            "case_count": len(frame),
            "class_counts": {str(k): int(v) for k, v in frame.case_label.value_counts().sort_index().items()},
            "embedding_dimension": int(embeddings.shape[1]),
            "weight_integrity_verified": True,
            "checkpoint_key_counts": key_counts,
            "full_gpu_verified": True,
            "placement": placement,
            "peak_vram_gib": round(peak, 3),
            "total_encode_seconds": elapsed,
            "mean_encode_seconds": elapsed / len(frame),
            **metrics,
        }
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(columns=["point_id", "filename", "error_message"]).to_csv(failed_path, index=False, encoding="utf-8-sig")
        write_report(summary, resolve(cfg["outputs"]["report"]))
        log.info("stage_42c %s: %s", status, summary)
        return 0 if status == "INDEX_PASS" else 2
    except Exception as error:
        log.exception("stage_42c failed")
        rows = locals().get("failures", []) or [{"point_id": "", "filename": "stage_42c", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"}]
        with failed_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            writer.writeheader()
            writer.writerows(rows)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
