# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from typing import Sequence, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


def _ones_like(metric: ArrayLike) -> ArrayLike:
    if isinstance(metric, torch.Tensor):
        return torch.ones_like(metric)
    return np.ones_like(metric)


def apply_multiplicative_metric_weight(metric: ArrayLike, weight: float) -> ArrayLike:
    weight = float(weight)
    if weight < 0.0:
        raise ValueError(f"Multiplicative metric weights must be non-negative, got {weight}.")
    if weight == 0.0:
        return _ones_like(metric)
    if weight == 1.0:
        return metric
    return metric ** weight


def aggregate_multiplicative_weighted_score(
    multiplicative_metrics: Sequence[ArrayLike],
    weighted_metrics: Sequence[ArrayLike],
    weighted_metric_weights: Sequence[float],
) -> ArrayLike:
    if len(weighted_metrics) != len(weighted_metric_weights):
        raise ValueError(
            "Weighted metrics and weights must have the same length: "
            f"{len(weighted_metrics)} != {len(weighted_metric_weights)}."
        )
    if not weighted_metrics:
        raise ValueError("At least one weighted metric is required to aggregate a score.")

    weights = [float(weight) for weight in weighted_metric_weights]
    if any(weight < 0.0 for weight in weights):
        raise ValueError(f"Weighted metric weights must be non-negative, got {weights}.")

    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError(
            "At least one weighted metric weight must be positive when aggregating a score."
        )

    weighted_score = weighted_metrics[0] * weights[0]
    for metric, weight in zip(weighted_metrics[1:], weights[1:]):
        weighted_score = weighted_score + metric * weight
    weighted_score = weighted_score / total_weight

    final_score = weighted_score
    for metric in multiplicative_metrics:
        final_score = final_score * metric
    return final_score