"""
Evaluation Metrics for Few-Shot Action Recognition.

Paper Section 4.1:
    "We report the mean accuracy and 95% confidence interval over
     10,000 randomly sampled test episodes."

Standard FSAR evaluation protocol:
    1. Sample 10,000 test episodes
    2. Compute accuracy for each episode
    3. Report: mean ± 95% CI
"""

import numpy as np
import scipy.stats
import torch
from typing import List, Tuple


def compute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy for a single episode.

    Args:
        predictions: [Q] — predicted class indices
        targets:     [Q] — ground truth class indices

    Returns:
        Accuracy as a float in [0, 1]
    """
    assert predictions.shape == targets.shape, (
        f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
    )
    correct = (predictions == targets).sum().item()
    total = targets.shape[0]
    return correct / total


def confidence_interval(
    accuracies: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """
    Compute mean accuracy and confidence interval.

    Uses Student's t-distribution for the CI calculation, which is
    appropriate for the sample sizes used in FSAR evaluation (10,000 episodes).

    Args:
        accuracies: List of per-episode accuracy values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        tuple of (mean_accuracy, confidence_interval_half_width)
        
    Example:
        mean, ci = confidence_interval(accs)
        print(f"Accuracy: {mean*100:.2f}% ± {ci*100:.2f}%")
    """
    n = len(accuracies)
    if n == 0:
        return 0.0, 0.0

    arr = np.array(accuracies)
    mean = np.mean(arr)

    if n == 1:
        return float(mean), 0.0

    # Standard error of the mean
    se = scipy.stats.sem(arr)

    # t-value for the given confidence level and degrees of freedom
    t_value = scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)

    # Confidence interval half-width
    ci = se * t_value

    return float(mean), float(ci)


def compute_episode_metrics(
    result: dict,
    query_labels: torch.Tensor,
) -> dict:
    """
    Compute comprehensive metrics for a single episode.

    Args:
        result: Output dict from Manta.forward()
        query_labels: [Q] ground truth labels

    Returns:
        dict with accuracy and per-class metrics
    """
    pred_labels = result['pred_labels']
    distances = result['distances']

    # Overall accuracy
    accuracy = compute_accuracy(pred_labels, query_labels)

    # Per-class accuracy
    n_way = distances.shape[1]
    per_class_acc = {}
    for c in range(n_way):
        mask = (query_labels == c)
        if mask.sum() > 0:
            class_acc = (pred_labels[mask] == c).float().mean().item()
            per_class_acc[c] = class_acc

    return {
        'accuracy': accuracy,
        'per_class_accuracy': per_class_acc,
        'mean_distance': distances.mean().item(),
    }
