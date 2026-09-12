"""Ablation-aware form of the frozen solar-directional multitask model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from training.src.model import SolarDirectionalMultitaskModel


@dataclass(frozen=True)
class AblationSpec:
    variant_id: str
    zero_dynamic_fields: tuple[str, ...] = ()
    use_global_dinov2: bool = True
    use_directional_dinov2: bool = True
    use_semantic_structure: bool = True
    use_cross_attention: bool = True
    use_azimuth_encoding: bool = True
    connect_shade_to_tmrt: bool = True


class AblationSolarDirectionalMultitaskModel(SolarDirectionalMultitaskModel):
    """Keep tensor dimensions fixed while removing specified information paths."""

    def __init__(
        self,
        semantic_dim: int,
        dino_dim: int,
        dynamic_fields: list[str],
        window_azimuth_degrees: list[float],
        model_config: dict[str, Any],
        dropout: float,
        ablation: AblationSpec,
    ) -> None:
        super().__init__(
            semantic_dim,
            dino_dim,
            len(dynamic_fields),
            window_azimuth_degrees,
            model_config,
            dropout,
        )
        self.ablation = ablation
        missing = sorted(set(ablation.zero_dynamic_fields) - set(dynamic_fields))
        if missing:
            raise ValueError(f"Unknown dynamic ablation fields: {missing}")
        self.zero_dynamic_indices = tuple(
            dynamic_fields.index(field) for field in ablation.zero_dynamic_fields
        )

    def forward(
        self,
        semantic_features: torch.Tensor,
        dino_mean: torch.Tensor,
        dino_windows: torch.Tensor,
        dynamic_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.zero_dynamic_indices:
            dynamic_features = dynamic_features.clone()
            dynamic_features[:, list(self.zero_dynamic_indices)] = 0.0
        dynamic_embedding = self.dynamic_encoder(dynamic_features)

        if self.ablation.use_directional_dinov2:
            directional = self.directional_projection(dino_windows)
            if self.ablation.use_azimuth_encoding:
                directional = directional + self.direction_encoding(
                    self.window_direction_encoding
                ).unsqueeze(0)
            if self.ablation.use_cross_attention:
                direction_context, attention_weights = self.cross_attention(
                    query=dynamic_embedding.unsqueeze(1),
                    key=directional,
                    value=directional,
                    need_weights=True,
                    average_attn_weights=True,
                )
                direction_context = direction_context.squeeze(1)
            else:
                direction_context = directional.mean(dim=1)
                attention_weights = torch.full(
                    (directional.shape[0], directional.shape[1]),
                    1.0 / directional.shape[1],
                    dtype=directional.dtype,
                    device=directional.device,
                )
        else:
            direction_context = torch.zeros(
                (dynamic_embedding.shape[0], self.dimensions.directional_hidden),
                dtype=dynamic_embedding.dtype,
                device=dynamic_embedding.device,
            )
            attention_weights = torch.full(
                (dynamic_embedding.shape[0], self.dimensions.window_count),
                1.0 / self.dimensions.window_count,
                dtype=dynamic_embedding.dtype,
                device=dynamic_embedding.device,
            )

        if self.ablation.use_semantic_structure:
            semantic_global = self.semantic_encoder(semantic_features)
        else:
            semantic_global = torch.zeros(
                (dynamic_embedding.shape[0], self.dimensions.semantic_hidden),
                dtype=dynamic_embedding.dtype,
                device=dynamic_embedding.device,
            )
        if self.ablation.use_global_dinov2:
            dino_global = self.global_dino_encoder(dino_mean)
        else:
            dino_global = torch.zeros(
                (dynamic_embedding.shape[0], self.dimensions.global_dino_hidden),
                dtype=dynamic_embedding.dtype,
                device=dynamic_embedding.device,
            )

        fused = self.fusion(
            torch.cat(
                [direction_context, semantic_global, dino_global, dynamic_embedding],
                dim=1,
            )
        )
        shade_hidden = self.shade_hidden(fused)
        shade_logit = self.shade_output(shade_hidden).squeeze(1)
        shade_prediction = torch.sigmoid(shade_logit)
        if self.ablation.connect_shade_to_tmrt:
            tmrt_shade_hidden = shade_hidden
            tmrt_shade_prediction = shade_prediction.unsqueeze(1)
        else:
            tmrt_shade_hidden = torch.zeros_like(shade_hidden)
            tmrt_shade_prediction = torch.zeros_like(shade_prediction.unsqueeze(1))
        tmrt_standardized = self.tmrt_head(
            torch.cat(
                [fused, tmrt_shade_hidden, tmrt_shade_prediction, dynamic_embedding],
                dim=1,
            )
        ).squeeze(1)
        return {
            "shade_logit": shade_logit,
            "shade_prediction": shade_prediction,
            "tmrt_standardized": tmrt_standardized,
            # Match the frozen base-model contract: [batch, directions].
            "attention_weights": (
                attention_weights.squeeze(1)
                if attention_weights.ndim == 3
                else attention_weights
            ),
        }
