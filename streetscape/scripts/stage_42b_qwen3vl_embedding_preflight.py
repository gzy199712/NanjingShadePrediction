"""stage_42b: full-CUDA multimodal embedding preflight on six local cases."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from qwen_vl_utils.vision_process import process_vision_info
from tqdm import tqdm
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLModel,
    Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "streetscape/configs/qwen3vl_embedding_preflight.yaml"
LOG_DIR = ROOT / "artifacts/logs/streetscape_planning/stage_42b"


@dataclass
class Qwen3VLForEmbeddingOutput(ModelOutput):
    """Output contract used by the official Qwen3-VL embedding implementation."""

    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Official Qwen wrapper shape; the `model.` level matches the checkpoint keys."""

    _checkpoint_conversion_mapping = {}
    accepts_loss_kwargs = False
    config_class = Qwen3VLConfig
    config: Qwen3VLConfig

    def __init__(self, config: Qwen3VLConfig):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Qwen3VLForEmbeddingOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            cache_position=cache_position,
            **kwargs,
        )
        return Qwen3VLForEmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


class OfficialQwen3VLEmbedder:
    """Minimal image+text path adapted from QwenLM/Qwen3-VL-Embedding."""

    def __init__(self, model_id: str, instruction: str, offline: bool):
        self.model, self.loading_info = Qwen3VLForEmbedding.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=offline,
            output_loading_info=True,
        )
        self.model = self.model.to("cuda").eval()
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_id,
            padding_side="right",
            local_files_only=offline,
        )
        self.instruction = instruction

    @staticmethod
    def _punctuate(text: str) -> str:
        text = text.strip()
        if text and not unicodedata.category(text[-1]).startswith("P"):
            text += "."
        return text

    @torch.inference_mode()
    def encode(self, text: str, image_path: str, output_dimension: int) -> np.ndarray:
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self._punctuate(self.instruction)}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image, "min_pixels": 4096, "max_pixels": 1843200},
                        {"type": "text", "text": text},
                    ],
                },
            ]
            rendered = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            images, video_inputs, video_kwargs = process_vision_info(
                conversation,
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
            if video_inputs is not None:
                videos, video_metadata = zip(*video_inputs)
                videos, video_metadata = list(videos), list(video_metadata)
            else:
                videos, video_metadata = None, None
            inputs = self.processor(
                text=[rendered],
                images=images,
                videos=videos,
                video_metadata=video_metadata,
                truncation=True,
                max_length=8192,
                padding=True,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
        inputs = {name: value.to("cuda") for name, value in inputs.items()}
        outputs = self.model(**inputs)
        mask = inputs["attention_mask"]
        final_positions = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1
        pooled = outputs.last_hidden_state[
            torch.arange(outputs.last_hidden_state.shape[0], device="cuda"), final_positions
        ]
        pooled = F.normalize(pooled[:, :output_dimension].float(), p=2, dim=-1)
        return pooled[0].cpu().numpy()


def resolve(value: str) -> Path:
    return ROOT / value


def logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("stage_42b")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (logging.FileHandler(LOG_DIR / "stage_42b.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def pilot_cases(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    rows = []
    for _, group in frame.groupby("case_label", sort=True):
        ordered = group.sort_values(["point_id", "heading"], kind="stable")
        indices = [0, len(ordered) - 1] if count == 2 else list(range(min(count, len(ordered))))
        rows.append(ordered.iloc[indices])
    return pd.concat(rows, ignore_index=True).drop_duplicates("case_id")


def assert_cuda(model) -> dict[str, Any]:
    devices: dict[str, int] = {}
    for parameter in tqdm(model.parameters(), desc="Verifying embedding placement", unit="tensor", dynamic_ncols=True):
        devices[parameter.device.type] = devices.get(parameter.device.type, 0) + 1
    if any(device != "cuda" for device in devices):
        raise RuntimeError(f"Non-CUDA embedding parameters detected: {devices}")
    return {"parameter_devices": devices}


def nearest_neighbor_accuracy(embeddings: np.ndarray, labels: list[str]) -> tuple[float, list[dict[str, Any]], np.ndarray]:
    similarity = embeddings @ embeddings.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = similarity.argmax(axis=1)
    rows = []
    for index, neighbor in enumerate(neighbors):
        rows.append({"query_index": index, "neighbor_index": int(neighbor), "query_label": labels[index], "neighbor_label": labels[int(neighbor)], "similarity": float(similarity[index, neighbor]), "label_match": labels[index] == labels[int(neighbor)]})
    return float(np.mean([row["label_match"] for row in rows])), rows, similarity


def write_report(summary: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# stage_42b Qwen3-VL multimodal embedding preflight

Status: **{summary['status']}**  
Generated: {summary['generated']}

- Model: `{summary['model_id']}`
- Cases: {summary['case_count']}
- Embedding dimension: {summary['embedding_dimension']}
- Checkpoint weight integrity: {summary['weight_integrity_verified']}
- Missing / unexpected checkpoint keys: {summary['missing_key_count']} / {summary['unexpected_key_count']}
- Full GPU: {summary['full_gpu_verified']}
- Peak reserved VRAM: {summary['peak_vram_gib']:.3f} GiB
- Mean encode time: {summary['mean_encode_seconds']:.3f} s/case
- Leave-one-out nearest-neighbor label accuracy: {summary['nearest_neighbor_label_accuracy']:.1%}

The diagnostic contains only two examples per class and is a hardware/signal preflight, not the
formal retrieval evaluation. Passing permits indexing the 300-case library; it does not validate
planning recommendations or unlock generation.
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "pilot", "run"), default="check")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    log = logger()
    failed = LOG_DIR / "failed_files.csv"
    try:
        cfg = config()
        library_path = resolve(cfg["inputs"]["case_library"])
        if not library_path.is_file():
            raise FileNotFoundError(library_path)
        library = pd.read_csv(library_path, dtype={"point_id": str})
        pilot = pilot_cases(library, int(cfg["pilot"]["cases_per_label"]))
        check = {"status": "PASS" if torch.cuda.is_available() and len(library) == 300 and len(pilot) == 6 and pilot.image_exists.all() else "FAIL", "library_cases": len(library), "pilot_cases": len(pilot), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "sentence_transformers": None}
        import sentence_transformers
        check["sentence_transformers"] = sentence_transformers.__version__
        if args.mode == "check":
            print(json.dumps(check, ensure_ascii=False, indent=2))
            return 0 if check["status"] == "PASS" else 2
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        output = resolve(cfg["outputs"]["directory"])
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {output}")
        output.mkdir(parents=True, exist_ok=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        log.info("Loading %s", cfg["model"]["model_id"])
        embedder = OfficialQwen3VLEmbedder(
            cfg["model"]["model_id"],
            cfg["model"]["instruction"],
            args.offline,
        )
        missing_keys = embedder.loading_info.get("missing_keys", [])
        unexpected_keys = embedder.loading_info.get("unexpected_keys", [])
        mismatched_keys = embedder.loading_info.get("mismatched_keys", [])
        error_messages = embedder.loading_info.get("error_msgs", [])
        if missing_keys or unexpected_keys or mismatched_keys or error_messages:
            raise RuntimeError(
                "Checkpoint integrity failure: "
                f"missing={len(missing_keys)}, unexpected={len(unexpected_keys)}, "
                f"mismatched={len(mismatched_keys)}, errors={len(error_messages)}"
            )
        placement = assert_cuda(embedder.model)
        vectors, times = [], []
        for case in tqdm(pilot.to_dict("records"), desc="Embedding pilot cases", unit="case", dynamic_ncols=True):
            started = time.perf_counter()
            vector = embedder.encode(
                case["case_text"], case["image_path"], int(cfg["model"]["output_dimension"])
            )
            times.append(time.perf_counter() - started)
            vectors.append(vector.astype(np.float32))
        embeddings = np.stack(vectors)
        labels = pilot.case_label.tolist()
        accuracy, neighbor_rows, similarity = nearest_neighbor_accuracy(embeddings, labels)
        peak = torch.cuda.max_memory_reserved() / 1024**3
        if peak > float(cfg["runtime"]["maximum_peak_vram_gib"]):
            raise RuntimeError(f"Peak VRAM {peak:.3f} GiB exceeds gate")
        status = "PILOT_PASS" if accuracy >= float(cfg["pilot"]["minimum_nearest_neighbor_label_accuracy"]) else "PILOT_SIGNAL_FAIL"
        pilot[["case_id", "case_label", "point_id", "heading", "image_path", "evidence_source"]].to_csv(output / "pilot_manifest.csv", index=False, encoding="utf-8-sig")
        np.save(output / "pilot_embeddings.npy", embeddings)
        np.save(output / "pilot_similarity.npy", similarity)
        pd.DataFrame(neighbor_rows).to_csv(output / "nearest_neighbors.csv", index=False, encoding="utf-8-sig")
        summary = {"status": status, "generated": datetime.now().astimezone().isoformat(), "model_id": cfg["model"]["model_id"], "case_count": len(pilot), "embedding_dimension": int(embeddings.shape[1]), "weight_integrity_verified": True, "missing_key_count": 0, "unexpected_key_count": 0, "mismatched_key_count": 0, "full_gpu_verified": True, "peak_vram_gib": round(peak, 3), "mean_encode_seconds": float(np.mean(times)), "nearest_neighbor_label_accuracy": accuracy, "placement": placement}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(columns=["point_id", "filename", "error_message"]).to_csv(failed, index=False, encoding="utf-8-sig")
        write_report(summary, resolve(cfg["outputs"]["report"]))
        log.info("stage_42b %s: %s", status, summary)
        return 0 if status == "PILOT_PASS" else 2
    except Exception as error:
        log.exception("stage_42b failed")
        with failed.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["point_id", "filename", "error_message"])
            if handle.tell() == 0:
                writer.writeheader()
            writer.writerow({"point_id": "", "filename": "stage_42b", "error_message": f"{type(error).__name__}: {error}\n{traceback.format_exc()}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
