"""Tests for the RACE-inspired multi-dimensional code-quality metrics and their
wiring into the existing HumanEval ``pass_at_k`` call site.
"""

import pytest

from lm_eval.tasks.humaneval import utils
from lm_eval.tasks.humaneval.code_quality import code_quality_metrics


CLEAN_CODE = '''\
def add(a, b):
    """Return the sum of two numbers."""
    return a + b
'''

MESSY_CODE = (
    "def F(X,Y,Z,W):\n"
    + "    if X:\n        if Y:\n            if Z:\n                if W:\n"
    + "                    return X and Y and Z and W or X or Y or Z\n"
    + "    XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX = 1\n"
    + "    return XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n"
)


def test_metrics_keys_and_bounds():
    out = code_quality_metrics([[CLEAN_CODE]])
    assert set(out) == {"readability", "complexity", "maintainability", "code_quality"}
    for value in out.values():
        assert 0.0 <= value <= 1.0


def test_clean_code_scores_higher_than_messy():
    clean = code_quality_metrics([[CLEAN_CODE]])
    messy = code_quality_metrics([[MESSY_CODE]])
    assert clean["readability"] > messy["readability"]
    assert clean["complexity"] > messy["complexity"]
    assert clean["code_quality"] > messy["code_quality"]


def test_invalid_code_does_not_raise_and_stays_neutral():
    out = code_quality_metrics([["def broken(:\n  return"]])
    # Unparsable code keeps neutral structural scores instead of raising.
    assert out["complexity"] == 0.5
    assert out["maintainability"] == 0.5


def test_accepts_flat_and_string_shapes():
    assert code_quality_metrics(CLEAN_CODE)["code_quality"] > 0.0
    assert code_quality_metrics([CLEAN_CODE])["code_quality"] > 0.0


class _FakeCodeEval:
    """Stand-in for the HF ``code_eval`` metric so the wiring can be exercised
    without code execution or the ``evaluate`` dependency.
    """

    def compute(self, references, predictions, k):
        return ({"pass@1": 1.0}, None)


def test_pass_at_k_merges_quality_into_results(monkeypatch):
    # Bypass the lazy ``code_eval`` load with a fake compute object.
    monkeypatch.setattr(utils, "compute_", _FakeCodeEval())

    result = utils.pass_at_k(
        references=["assert add(2, 3) == 5"],
        predictions=[[CLEAN_CODE]],
        k=[1],
    )

    # Existing correctness metric is still reported ...
    assert result["pass@1"] == 1.0
    # ... now augmented with the RACE quality dimensions at the same call site.
    for key in ("readability", "complexity", "maintainability", "code_quality"):
        assert key in result
        assert 0.0 <= result[key] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
