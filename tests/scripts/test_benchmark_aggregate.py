"""Tests for the DrBenchmark-style benchmark macro-aggregation.

Exercises the ``scripts.make_table_results.make_table`` wiring (a non-new
module) as well as the new ``scripts.benchmark_aggregate`` helpers.
"""

import pytest


pytest.importorskip("pytablewriter")

from scripts.benchmark_aggregate import (
    benchmark_macro_average,
    benchmark_score_row,
    primary_metric,
)
from scripts.make_table_results import make_table


def _result_dict():
    """A heterogeneous benchmark: accuracy, F1, and an unbounded ppl task."""
    return {
        "results": {
            "task_cls": {"acc": 0.80, "acc_stderr": 0.01},
            "task_ner": {"f1": 0.60, "f1_stderr": 0.02},
            "task_lm": {"ppl": 12.5},
        },
        "versions": {"task_cls": 1, "task_ner": 1, "task_lm": 1},
    }


def test_primary_metric_prefers_accuracy():
    assert primary_metric({"f1": 0.5, "acc": 0.9, "acc_stderr": 0.01}) == ("acc", 0.9)


def test_primary_metric_skips_stderr_and_alias():
    # Only a stderr column and an alias -> nothing comparable to report.
    assert primary_metric({"alias": "task", "acc_stderr": 0.01}) is None


def test_macro_average_excludes_unbounded_metrics():
    summary = benchmark_macro_average(_result_dict()["results"])
    # ppl task is dropped; mean of acc=0.80 and f1=0.60 -> 0.70.
    assert summary["num_tasks"] == 2
    assert summary["benchmark_score"] == pytest.approx(0.70)
    assert "task_lm" not in summary["per_task"]


def test_macro_average_returns_none_without_comparable_metrics():
    assert benchmark_macro_average({"task_lm": {"ppl": 5.0}}) is None


def test_benchmark_score_row_shape_matches_headers():
    row = benchmark_score_row(_result_dict())
    # [Task, Version, Metric, Value, "", Stderr]
    assert len(row) == 6
    assert row[3] == "70.00"
    assert row[5] == "n=2"


def test_make_table_appends_benchmark_row():
    table = make_table(_result_dict())
    # The wiring in make_table_results.py must surface the aggregate row.
    assert "benchmark (macro-avg)" in table
    # pytablewriter normalises the "70.00" cell to "70.0".
    assert "70.0" in table


def test_make_table_without_aggregatable_tasks():
    only_ppl = {"results": {"task_lm": {"ppl": 5.0}}, "versions": {"task_lm": 1}}
    table = make_table(only_ppl)
    assert "benchmark (macro-avg)" not in table


if __name__ == "__main__":
    test_primary_metric_prefers_accuracy()
    test_primary_metric_skips_stderr_and_alias()
    test_macro_average_excludes_unbounded_metrics()
    test_macro_average_returns_none_without_comparable_metrics()
    test_benchmark_score_row_shape_matches_headers()
    test_make_table_appends_benchmark_row()
    test_make_table_without_aggregatable_tasks()
    print("All tests passed.")
