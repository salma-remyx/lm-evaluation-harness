"""Tests for the confidence-calibration metric integration.

These exercise the wiring into the existing multiple_choice pipeline:
the aggregation registered in ``lm_eval.api.metrics`` and the item that
``lm_eval.api.task.ConfigurableTask.process_results`` emits for it.
"""

import numpy as np

from lm_eval.api.metrics import calibration_error
from lm_eval.api.registry import get_aggregation, get_metric
from lm_eval.api.task import ConfigurableTask
from lm_eval.config.task import TaskConfig


def test_calibration_error_registered():
    """The metric and aggregation are discoverable through the registry."""
    assert get_aggregation("calibration_error") is not None
    assert get_metric("calibration_error") is not None


def test_expected_calibration_error_value():
    """Two 0.9-confidence predictions, one right one wrong, land in the same
    bin -> accuracy 0.5, confidence 0.9, gap 0.4.
    """
    items = [
        (0, [0.9, 0.05, 0.05]),  # pred 0 == gold 0  (correct)
        (1, [0.9, 0.05, 0.05]),  # pred 0 != gold 1  (incorrect)
    ]
    assert calibration_error(items) == 0.4


def test_maximum_calibration_error_value():
    from lm_eval.api.calibration import calibration_error as raw_ce

    items = [
        (0, [0.9, 0.05, 0.05]),
        (1, [0.9, 0.05, 0.05]),
    ]
    # Single populated bin -> ECE and MCE coincide here.
    assert raw_ce(items, norm="max") == 0.4


def test_perfect_calibration_is_zero():
    """A confident, correct prediction contributes no calibration gap."""
    items = [(0, [1.0, 0.0, 0.0])]
    assert calibration_error(items) == 0.0


class _MockCalibrationTask(ConfigurableTask):
    """Minimal multiple_choice task requesting the calibration metric."""

    def __init__(self, metrics):
        config = {
            "task": "test_calibration",
            "output_type": "multiple_choice",
            "metric_list": [{"metric": m} for m in metrics],
            "doc_to_choice": ["A", "B", "C"],
            "doc_to_target": 1,
            "target_delimiter": " ",
        }
        self._config = TaskConfig(**config)
        self.OUTPUT_TYPE = "multiple_choice"
        self.multiple_input = 0
        self.multiple_target = 0
        self._metric_fn_list = {m: None for m in metrics}
        self._metric_fn_kwargs = {m: {} for m in metrics}
        self._aggregation_list = {}
        self._higher_is_better = {}

    def doc_to_choice(self, doc):
        return ["A", "B", "C"]

    def doc_to_target(self, doc):
        return 1

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def download(self, **kwargs):
        pass


def test_process_results_emits_calibration_item():
    """The task edit feeds the metric a (gold, prob) tuple, and that tuple
    aggregates through the registered calibration aggregation.
    """
    task = _MockCalibrationTask(["acc", "calibration_error"])
    # Choice B (index 1, the gold) has the highest loglikelihood.
    results = [(-2.0, False), (-1.0, True), (-3.0, False)]
    result_dict = task.process_results({}, results)

    assert "calibration_error" in result_dict
    gold, probs = result_dict["calibration_error"]
    assert gold == 1
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-6)

    # The emitted item is contract-compatible with the aggregation.
    agg = get_aggregation("calibration_error")
    assert 0.0 <= agg([result_dict["calibration_error"]]) <= 1.0


def test_process_results_omits_calibration_when_not_requested():
    task = _MockCalibrationTask(["acc"])
    results = [(-2.0, False), (-1.0, True), (-3.0, False)]
    result_dict = task.process_results({}, results)
    assert "calibration_error" not in result_dict
