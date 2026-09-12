"""Solar-conditioned, direction-aware multitask neural network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ModelDimensions:
    semantic_dim: int
    dino_dim: int
    dynamic_dim: int
    window_count: int
    directional_hidden: int
    semantic_hidden: int
    global_dino_hidden: int
    dynamic_hidden: int
    fusion_hidden_1: int
    fusion_hidden_2: int
    attention_heads: int


class SolarDirectionalMultitaskModel(nn.Module):
    """Predict shade rate and standardized Tmrt from formal model inputs."""

    def __init__(
        self,
        semantic_dim: int,
        dino_dim: int,
        dynamic_dim: int,
        window_azimuth_degrees: list[float],
        model_config: dict[str, Any],
        dropout: float,
    ) -> None:
        super().__init__()
        directional_hidden = int(model_config["directional_hidden_dim"])
        semantic_hidden = int(model_config["semantic_hidden_dim"])
        global_dino_hidden = int(model_config["global_dino_hidden_dim"])
        dynamic_hidden = int(model_config["dynamic_hidden_dim"])
        fusion_hidden = [int(value) for value in model_config["fusion_hidden_dims"]]
        attention_heads = int(model_config["attention_heads"])
        if directional_hidden != dynamic_hidden:
            raise ValueError(
                "dynamic_hidden_dim must equal directional_hidden_dim for attention"
            )
        if directional_hidden % attention_heads != 0:
            raise ValueError("directional hidden dimension must divide attention heads")

        self.dimensions = ModelDimensions(
            semantic_dim=semantic_dim,
            dino_dim=dino_dim,
            dynamic_dim=dynamic_dim,
            window_count=len(window_azimuth_degrees),
            directional_hidden=directional_hidden,
            semantic_hidden=semantic_hidden,
            global_dino_hidden=global_dino_hidden,
            dynamic_hidden=dynamic_hidden,
            fusion_hidden_1=fusion_hidden[0],
            fusion_hidden_2=fusion_hidden[1],
            attention_heads=attention_heads,
        )
        self.directional_projection = nn.Sequential(
            nn.Linear(dino_dim, directional_hidden),
            nn.LayerNorm(directional_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.direction_encoding = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.Linear(128, directional_hidden),
        )
        azimuth_radians = torch.deg2rad(
            torch.tensor(window_azimuth_degrees, dtype=torch.float32)
        )
        fixed_encoding = torch.stack(
            [torch.sin(azimuth_radians), torch.cos(azimuth_radians)], dim=1
        )
        self.register_buffer("window_direction_encoding", fixed_encoding)
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(dynamic_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, dynamic_hidden),
            nn.LayerNorm(dynamic_hidden),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=directional_hidden,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, semantic_hidden),
            nn.LayerNorm(semantic_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(semantic_hidden, semantic_hidden),
            nn.LayerNorm(semantic_hidden),
            nn.GELU(),
        )
        self.global_dino_encoder = nn.Sequential(
            nn.LayerNorm(dino_dim),
            nn.Linear(dino_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, global_dino_hidden),
            nn.LayerNorm(global_dino_hidden),
            nn.GELU(),
        )
        fusion_dim = (
            directional_hidden
            + semantic_hidden
            + global_dino_hidden
            + dynamic_hidden
        )
        expected_fusion = (
            int(model_config["directional_hidden_dim"])
            + int(model_config["semantic_hidden_dim"])
            + int(model_config["global_dino_hidden_dim"])
            + int(model_config["dynamic_hidden_dim"])
        )
        if fusion_dim != expected_fusion:
            raise ValueError("fusion dimension mismatch")
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden[0]),
            nn.LayerNorm(fusion_hidden[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden[0], fusion_hidden[1]),
            nn.LayerNorm(fusion_hidden[1]),
            nn.GELU(),
        )
        self.shade_hidden = nn.Sequential(
            nn.Linear(fusion_hidden[1], 128),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.shade_output = nn.Linear(128, 1)
        tmrt_input_dim = fusion_hidden[1] + 128 + 1 + dynamic_hidden
        self.tmrt_head = nn.Sequential(
            nn.Linear(tmrt_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        dino_mean: torch.Tensor,
        dino_windows: torch.Tensor,
        dynamic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        directional = self.directional_projection(dino_windows)
        directional = directional + self.direction_encoding(
            self.window_direction_encoding
        ).unsqueeze(0)
        dynamic_embedding = self.dynamic_encoder(dynamic_features)
        direction_context, attention_weights = self.cross_attention(
            query=dynamic_embedding.unsqueeze(1),
            key=directional,
            value=directional,
            need_weights=True,
            average_attn_weights=True,
        )
        direction_context = direction_context.squeeze(1)
        semantic_global = self.semantic_encoder(semantic_features)
        dino_global = self.global_dino_encoder(dino_mean)
        fused = self.fusion(
            torch.cat(
                [
                    direction_context,
                    semantic_global,
                    dino_global,
                    dynamic_embedding,
                ],
                dim=1,
            )
        )
        shade_hidden = self.shade_hidden(fused)
        shade_logit = self.shade_output(shade_hidden).squeeze(1)
        shade_prediction = torch.sigmoid(shade_logit)
        tmrt_standardized = self.tmrt_head(
            torch.cat(
                [
                    fused,
                    shade_hidden,
                    shade_prediction.unsqueeze(1),
                    dynamic_embedding,
                ],
                dim=1,
            )
        ).squeeze(1)
        return {
            "shade_logit": shade_logit,
            "shade_prediction": shade_prediction,
            "tmrt_standardized": tmrt_standardized,
            "attention_weights": attention_weights.squeeze(1),
        }

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

