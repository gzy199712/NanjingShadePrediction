"""Loss definitions for continuous shade rate and Tmrt regression."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class MultitaskLossOutput:
    total: torch.Tensor
    shade: torch.Tensor
    tmrt: torch.Tensor


class MultitaskSmoothL1Loss(nn.Module):
    def __init__(self, lambda_shade: float, lambda_tmrt: float) -> None:
        super().__init__()
        self.lambda_shade = float(lambda_shade)
        self.lambda_tmrt = float(lambda_tmrt)
        self.regression = nn.SmoothL1Loss()

    def forward(
        self,
        shade_prediction: torch.Tensor,
        shade_target: torch.Tensor,
        tmrt_prediction_standardized: torch.Tensor,
        tmrt_target_standardized: torch.Tensor,
    ) -> MultitaskLossOutput:
        shade = self.regression(shade_prediction, shade_target)
        tmrt = self.regression(
            tmrt_prediction_standardized, tmrt_target_standardized
        )
        total = self.lambda_shade * shade + self.lambda_tmrt * tmrt
        return MultitaskLossOutput(total=total, shade=shade, tmrt=tmrt)

