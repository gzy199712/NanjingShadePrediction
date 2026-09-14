"""Minimal strong-baseline models sharing the formal four-input contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class _Heads(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers.extend((nn.Linear(current, width), nn.GELU(), nn.Dropout(dropout)))
            current = width
        self.backbone = nn.Sequential(*layers)
        self.shade = nn.Linear(current, 1)
        self.tmrt = nn.Linear(current, 1)

    def forward(self, features: torch.Tensor, directions: int) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        shade_logit = self.shade(hidden).squeeze(1)
        return {
            "shade_logit": shade_logit,
            "shade_prediction": torch.sigmoid(shade_logit),
            "tmrt_standardized": self.tmrt(hidden).squeeze(1),
            "attention_weights": torch.full(
                (features.shape[0], directions),
                1.0 / directions,
                device=features.device,
                dtype=features.dtype,
            ),
        }


class MultimodalMLP(nn.Module):
    """Plain concatenation of semantic, global, pooled directional and weather inputs."""

    def __init__(self, semantic_dim: int, dino_dim: int, dynamic_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.heads = _Heads(
            semantic_dim + dino_dim * 3 + dynamic_dim,
            [int(value) for value in config["hidden_dims"]],
            float(config["dropout"]),
        )

    def forward(self, semantic: torch.Tensor, global_dino: torch.Tensor, directional: torch.Tensor, dynamic: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = torch.cat((directional.mean(1), directional.std(1, unbiased=False)), dim=1)
        return self.heads(torch.cat((semantic, global_dino, pooled, dynamic), dim=1), directional.shape[1])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class DirectionalMLP(nn.Module):
    """Shared direction encoder plus order-free pooling; no azimuth or dynamic query."""

    def __init__(self, semantic_dim: int, dino_dim: int, dynamic_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        width = int(config["directional_hidden_dim"])
        self.direction_encoder = nn.Sequential(nn.Linear(dino_dim, width), nn.GELU())
        self.heads = _Heads(
            semantic_dim + dino_dim + width + dynamic_dim,
            [int(value) for value in config["hidden_dims"]],
            float(config["dropout"]),
        )

    def forward(self, semantic: torch.Tensor, global_dino: torch.Tensor, directional: torch.Tensor, dynamic: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = self.direction_encoder(directional).mean(1)
        return self.heads(torch.cat((semantic, global_dino, pooled, dynamic), dim=1), directional.shape[1])

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def make_baseline_model(model_id: str, semantic_dim: int, dino_dim: int, dynamic_dim: int, config: dict[str, Any]) -> nn.Module:
    classes = {"multimodal_mlp": MultimodalMLP, "directional_mlp": DirectionalMLP}
    if model_id not in classes:
        raise ValueError(f"Unknown neural baseline: {model_id}")
    return classes[model_id](semantic_dim, dino_dim, dynamic_dim, config)
