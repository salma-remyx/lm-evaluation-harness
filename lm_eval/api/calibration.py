"""Confidence-calibration metrics for multiple-choice evaluation.

Calibration measures whether a model's stated confidence matches its
empirical accuracy: a well-calibrated model that reports 70% confidence
should be correct roughly 70% of the time. This module computes the
standard binned calibration errors (Expected and Maximum Calibration
Error) from the per-item ``(gold, prob)`` tuples the multiple_choice
pipeline already produces for ``brier_score``.

Adapted from GRACE: A Granular Benchmark for Evaluating Model Calibration
against Human Calibration (https://arxiv.org/abs/2502.19684). GRACE's core
insight is that calibration is best assessed by binning predictions by
confidence and comparing per-bin confidence against per-bin correctness.
The paper's *human*-adjusted CalScore additionally compares model
calibration against a human buzzpoint dataset; that comparison is out of
scope here (it needs an external dataset the harness does not host), so
this module implements the model-side calibration error only, over the
max-softmax confidence and argmax-correctness signals that are already
available at the ``brier_score`` call site.
"""

import numpy as np


def _confidence_correctness(items):
    """Reduce ``(gold, prob)`` tuples to per-item (confidence, correct).

    ``prob`` is a probability distribution over the answer choices (the
    ``softmax`` of the per-choice loglikelihoods). Confidence is the
    max-softmax probability and the prediction is its argmax, matching the
    signal GRACE uses for calibration.
    """
    gold, predictions = zip(*items, strict=True)
    predictions = np.asarray(predictions, dtype=float)
    gold = np.asarray(gold)

    confidences = predictions.max(axis=1)
    pred_idx = predictions.argmax(axis=1)
    correct = (pred_idx == gold).astype(float)
    return confidences, correct


def _bin_stats(confidences, correct, num_bins):
    """Group predictions into equal-width confidence bins.

    Yields ``(weight, bin_confidence, bin_accuracy)`` for every non-empty
    bin, where ``weight`` is the fraction of samples that fell in the bin.
    """
    total = len(confidences)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    # np.digitize with the right edge keeps confidence==1.0 in the last bin.
    bin_ids = np.clip(np.digitize(confidences, bin_edges[1:-1]), 0, num_bins - 1)

    for b in range(num_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            continue
        yield count / total, confidences[mask].mean(), correct[mask].mean()


def calibration_error(items, num_bins: int = 10, norm: str = "l1") -> float:
    """Binned confidence-calibration error over ``(gold, prob)`` tuples.

    ``norm="l1"`` returns the Expected Calibration Error (ECE), the
    sample-weighted mean gap between bin confidence and bin accuracy.
    ``norm="max"`` returns the Maximum Calibration Error (MCE), the largest
    such gap over all bins. Lower is better for both.
    """
    confidences, correct = _confidence_correctness(items)
    gaps = [
        (weight, abs(conf - acc))
        for weight, conf, acc in _bin_stats(confidences, correct, num_bins)
    ]
    if not gaps:
        return 0.0

    if norm == "max":
        return float(max(gap for _, gap in gaps))
    if norm == "l1":
        return float(sum(weight * gap for weight, gap in gaps))
    raise ValueError(f"Unknown calibration norm '{norm}', expected 'l1' or 'max'.")


def expected_calibration_error(items) -> float:
    """Expected Calibration Error (ECE) — the ``brier_score``-style drop-in."""
    return calibration_error(items, norm="l1")
