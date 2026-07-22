import json

from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.loggers.unified_results_schema import (
    SCHEMA_VERSION,
    to_unified_schema,
)


def _sample_results() -> dict:
    """A minimal aggregated results dict in the shape save_results_aggregated
    receives (mirrors EvalAcc.dump + evaluator metadata).
    """
    return {
        "results": {
            "humaneval": {
                "name": "humaneval",
                "alias": "humaneval",
                "sample_len": 164,
                "pass@1,none": 0.32,
                "pass@1_stderr,none": "N/A",
                "acc,none": 0.5,
                "acc_stderr,none": 0.04,
            }
        },
        "versions": {"humaneval": 1.0},
        "n-shot": {"humaneval": 0},
        "n-samples": {"humaneval": 164},
        "higher_is_better": {"humaneval": {"pass@1": True, "acc": True}},
        "config": {
            "model": "hf",
            "model_args": "pretrained=EleutherAI/pythia-1b",
            "batch_size": 8,
            "device": "cpu",
        },
        "git_hash": "abc1234",
        "date": "2026-06-25T00:00:00",
    }


def test_to_unified_schema_structure():
    unified = to_unified_schema(_sample_results())

    assert unified["schema_version"] == SCHEMA_VERSION
    assert unified["source"]["framework"] == "lm-evaluation-harness"
    assert unified["source"]["git_hash"] == "abc1234"
    assert unified["model"]["name"] == "hf"
    assert unified["model"]["args"] == "pretrained=EleutherAI/pythia-1b"
    assert unified["model"]["config"]["batch_size"] == 8

    (bench,) = unified["benchmarks"]
    assert bench["benchmark"] == "humaneval"
    assert bench["version"] == 1.0
    assert bench["n_samples"] == 164

    metrics = {m["name"]: m for m in bench["metrics"]}
    assert metrics["pass@1"]["value"] == 0.32
    # "N/A" stderr is normalised to None.
    assert metrics["pass@1"]["stderr"] is None
    assert metrics["pass@1"]["higher_is_better"] is True
    assert metrics["acc"]["value"] == 0.5
    assert metrics["acc"]["stderr"] == 0.04


def test_save_results_aggregated_emits_unified_document(tmp_path):
    """Exercises the wiring edit in EvaluationTracker.save_results_aggregated:
    saving aggregated results must also produce the unified-schema companion.
    """
    tracker = EvaluationTracker(output_path=str(tmp_path))
    tracker.general_config_tracker.model_name = "pythia-1b"
    tracker.general_config_tracker.model_name_sanitized = "pythia-1b"

    tracker.save_results_aggregated(results=_sample_results(), samples=None)

    model_dir = tmp_path / "pythia-1b"
    unified_files = list(model_dir.glob("results_unified_*.json"))
    assert len(unified_files) == 1, "unified-schema companion file not written"

    unified = json.loads(unified_files[0].read_text(encoding="utf-8"))
    assert unified["schema_version"] == SCHEMA_VERSION
    assert unified["benchmarks"][0]["benchmark"] == "humaneval"

    # The original aggregated file is still written alongside it.
    assert list(model_dir.glob("results_2*.json"))
