"""Explicit, schema-driven loading for the formal training package."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TrainingPaths:
    """Resolved paths for every formal model input."""

    project_root: Path
    training_data: Path
    manifest: Path
    semantic_features: Path
    static_feature_schema: Path
    dino_mean_features: Path
    dino_window_features: Path
    dino_window_index: Path
    dynamic_conditions: Path
    labels: Path
    spatial_split: Path
    model_input_spec: Path
    model_target_spec: Path
    training_data_schema: Path
    overlap_pairs_qc: Path
    output_dir: Path

    @classmethod
    def from_config(cls, config_path: Path, config: dict[str, Any]) -> "TrainingPaths":
        project_value = Path(str(config["project"]["root"]))
        project_root = (
            project_value
            if project_value.is_absolute()
            else (config_path.parent / project_value).resolve()
        )
        training_value = Path(str(config["project"]["training_data"]))
        training_data = (
            training_value
            if training_value.is_absolute()
            else project_root / training_value
        ).resolve()
        inputs = config["inputs"]

        def training_path(key: str) -> Path:
            value = Path(str(inputs[key]))
            return (value if value.is_absolute() else training_data / value).resolve()

        overlap_value = Path(str(inputs["overlap_pairs_qc"]))
        overlap_path = (
            overlap_value
            if overlap_value.is_absolute()
            else project_root / overlap_value
        ).resolve()
        output_value = Path(str(config["preflight"]["output_dir"]))
        output_dir = (
            output_value
            if output_value.is_absolute()
            else project_root / output_value
        ).resolve()
        return cls(
            project_root=project_root,
            training_data=training_data,
            manifest=training_path("manifest"),
            semantic_features=training_path("semantic_features"),
            static_feature_schema=training_path("static_feature_schema"),
            dino_mean_features=training_path("dino_mean_features"),
            dino_window_features=training_path("dino_window_features"),
            dino_window_index=training_path("dino_window_index"),
            dynamic_conditions=training_path("dynamic_conditions"),
            labels=training_path("labels"),
            spatial_split=training_path("spatial_split"),
            model_input_spec=training_path("model_input_spec"),
            model_target_spec=training_path("model_target_spec"),
            training_data_schema=training_path("training_data_schema"),
            overlap_pairs_qc=overlap_path,
            output_dir=output_dir,
        )

    def formal_inputs(self) -> dict[str, Path]:
        """Return the required immutable input inventory."""
        return {
            "training_manifest": self.manifest,
            "semantic_features": self.semantic_features,
            "static_feature_schema": self.static_feature_schema,
            "dinov2_mean_features": self.dino_mean_features,
            "dinov2_window_features": self.dino_window_features,
            "dinov2_window_index": self.dino_window_index,
            "dynamic_conditions": self.dynamic_conditions,
            "point_hour_labels": self.labels,
            "spatial_split_assignments": self.spatial_split,
            "model_input_spec": self.model_input_spec,
            "model_target_spec": self.model_target_spec,
            "training_data_schema": self.training_data_schema,
        }


@dataclass
class FormalSpecs:
    """Machine-readable field and target definitions."""

    model_input: dict[str, Any]
    model_target: dict[str, Any]
    training_schema: dict[str, Any]
    static_schema: dict[str, Any]
    semantic_fields: list[str]
    dino_mean_fields: list[str]
    dynamic_fields: list[str]
    shade_target: str
    tmrt_target: str
    utci_target: str
    window_azimuth: list[float]


@dataclass
class FormalData:
    """Loaded tabular inputs plus a memory-mapped directional tensor."""

    manifest: pd.DataFrame
    semantic: pd.DataFrame
    dino_mean: pd.DataFrame
    dino_window: np.memmap
    dino_index: pd.DataFrame
    dynamic: pd.DataFrame
    labels: pd.DataFrame
    split: pd.DataFrame


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def load_specs(paths: TrainingPaths) -> FormalSpecs:
    model_input = load_yaml(paths.model_input_spec)
    model_target = load_yaml(paths.model_target_spec)
    training_schema = load_json(paths.training_data_schema)
    static_schema = load_json(paths.static_feature_schema)
    semantic_fields = list(
        static_schema["semantic_features"]["model_input_fields"]
    )
    dino_mean_fields = list(
        static_schema["dinov2_mean_features"]["feature_fields"]
    )
    dynamic_fields = list(model_input["dynamic_query_fields"])
    training_targets = model_target["training_targets"]
    validation_target = model_target["validation_only_target"]
    return FormalSpecs(
        model_input=model_input,
        model_target=model_target,
        training_schema=training_schema,
        static_schema=static_schema,
        semantic_fields=semantic_fields,
        dino_mean_fields=dino_mean_fields,
        dynamic_fields=dynamic_fields,
        shade_target=next(
            name for name in training_targets if name == "shade_rate"
        ),
        tmrt_target=next(name for name in training_targets if name == "tmrt_mean"),
        utci_target=next(name for name in validation_target if name == "utci_mean"),
        window_azimuth=[
            float(value)
            for value in model_input["directional_dino_tokens"]["azimuth_degrees"]
        ],
    )


def read_formal_csv(path: Path) -> pd.DataFrame:
    """Read a formal CSV while preserving identifiers and validating headers later."""
    return pd.read_csv(path, low_memory=False)


def load_formal_data(paths: TrainingPaths) -> FormalData:
    """Load formal data without copying the large directional DINO tensor."""
    return FormalData(
        manifest=read_formal_csv(paths.manifest),
        semantic=read_formal_csv(paths.semantic_features),
        dino_mean=read_formal_csv(paths.dino_mean_features),
        dino_window=np.load(paths.dino_window_features, mmap_mode="r"),
        dino_index=read_formal_csv(paths.dino_window_index),
        dynamic=read_formal_csv(paths.dynamic_conditions),
        labels=read_formal_csv(paths.labels),
        split=read_formal_csv(paths.spatial_split),
    )


class ManifestDataset(Dataset[dict[str, Any]]):
    """Dataset that uses only explicit row references from the formal manifest."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        semantic: pd.DataFrame,
        dino_mean: pd.DataFrame,
        dino_window: np.memmap,
        dynamic: pd.DataFrame,
        labels: pd.DataFrame,
        specs: FormalSpecs,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.semantic = semantic[specs.semantic_fields].to_numpy(
            dtype=np.float32, copy=True
        )
        self.dino_mean = dino_mean[specs.dino_mean_fields].to_numpy(
            dtype=np.float32, copy=True
        )
        self.dino_window = dino_window
        self.dynamic = dynamic[specs.dynamic_fields].to_numpy(
            dtype=np.float32, copy=True
        )
        self.shade = labels[specs.shade_target].to_numpy(
            dtype=np.float32, copy=True
        )
        self.tmrt = labels[specs.tmrt_target].to_numpy(
            dtype=np.float32, copy=True
        )
        self.utci = labels[specs.utci_target].to_numpy(
            dtype=np.float32, copy=True
        )
        self.window_azimuth = np.asarray(specs.window_azimuth, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.manifest.iloc[index]
        static_row = int(row["static_feature_row"])
        mean_row = int(row["dino_mean_row"])
        window_row = int(row["dino_window_row"])
        dynamic_row = int(row["dynamic_condition_row"])
        label_row = int(row["label_row"])
        return {
            "semantic_features": torch.from_numpy(self.semantic[static_row]),
            "dino_mean": torch.from_numpy(self.dino_mean[mean_row]),
            "dino_windows": torch.from_numpy(
                np.array(self.dino_window[window_row], dtype=np.float32, copy=True)
            ),
            "window_azimuth": torch.from_numpy(self.window_azimuth.copy()),
            "dynamic_features": torch.from_numpy(self.dynamic[dynamic_row]),
            "shade_target": torch.tensor(
                self.shade[label_row], dtype=torch.float32
            ),
            "tmrt_target": torch.tensor(
                self.tmrt[label_row], dtype=torch.float32
            ),
            "utci_target": torch.tensor(
                self.utci[label_row], dtype=torch.float32
            ),
            "sample_id": str(row["sample_id"]),
            "point_id": str(row["point_id"]),
            "hour": int(row["hour"]),
            "dataset_split": str(row["dataset_split"]),
            "manifest_row": int(row.name),
        }


def choose_manifest_rows(
    manifest: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    """Return a deterministic audit subset without changing split definitions."""
    if count >= len(manifest):
        return manifest.copy()
    generator = np.random.default_rng(seed)
    positions = np.sort(generator.choice(len(manifest), size=count, replace=False))
    return manifest.iloc[positions].copy()


def require_columns(
    frame: pd.DataFrame, required: Sequence[str], source_name: str
) -> list[str]:
    return [field for field in required if field not in frame.columns]

