"""Convert lm-evaluation-harness aggregated results into a source-agnostic,
unified evaluation-results schema.

The schema and approach are adapted from "Every Eval Ever: A Unifying Schema
and Community Repository for AI Evaluation Results"
(https://arxiv.org/abs/2606.14516). The paper observes that evaluation results
are scattered across incompatible formats with inconsistent metadata, which
hinders comparison, reuse, and cross-framework evaluation science. It proposes
a single, source-agnostic JSON document that standardises how a run's model,
evaluation, and per-benchmark metrics are represented.

This module implements the harness-side converter: given the results dict that
``EvaluationTracker.save_results_aggregated`` already serialises, it produces a
companion document in the unified shape so harness output can be compared and
shared alongside results from other frameworks. It deliberately does not
implement the community repository, instance-level schema, or the converters
for other frameworks described in the paper -- only the harness export.
"""

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "every-eval-ever/1.0"

# Keys inside a per-task results entry that are not "metric,filter" pairs.
_RESERVED_TASK_KEYS = frozenset({"name", "alias", "sample_len"})


def _split_metric_key(key: str) -> tuple[str, str]:
    """Split an lm-eval ``"metric,filter"`` key into ``(metric, filter)``.

    Keys without an explicit filter are reported against the ``"none"`` filter,
    matching how the harness aggregates metrics.
    """
    if "," in key:
        metric, filter_key = key.rsplit(",", 1)
        return metric, filter_key
    return key, "none"


def _collect_task_metrics(
    task_result: dict[str, Any],
    higher_is_better: dict[str, bool] | None,
) -> list[dict[str, Any]]:
    """Turn a flat ``{"acc,none": v, "acc_stderr,none": e, ...}`` entry into a
    list of structured metric records with value, stderr and direction.
    """
    higher_is_better = higher_is_better or {}
    # (metric, filter) -> record
    collected: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_key, value in task_result.items():
        if raw_key in _RESERVED_TASK_KEYS:
            continue
        metric_part, filter_key = _split_metric_key(raw_key)

        is_stderr = metric_part.endswith("_stderr")
        base_metric = metric_part[: -len("_stderr")] if is_stderr else metric_part

        record = collected.setdefault(
            (base_metric, filter_key),
            {
                "name": base_metric,
                "filter": filter_key,
                "value": None,
                "stderr": None,
                "higher_is_better": higher_is_better.get(base_metric),
            },
        )
        if is_stderr:
            # The harness stores missing stderr as the string "N/A".
            record["stderr"] = None if value == "N/A" else value
        else:
            record["value"] = value

    return [collected[k] for k in sorted(collected)]


def _model_section(results: dict[str, Any]) -> dict[str, Any]:
    """Assemble the standardised ``model`` block from harness metadata."""
    config = results.get("config", {}) or {}
    return {
        "name": results.get("model_name") or config.get("model"),
        "source": results.get("model_source"),
        "args": config.get("model_args"),
        "config": {
            k: v
            for k, v in config.items()
            if k not in {"model", "model_args"} and v is not None
        },
    }


def to_unified_schema(results: dict[str, Any]) -> dict[str, Any]:
    """Convert an lm-eval aggregated results dict to the unified schema.

    Args:
        results: The dict that ``save_results_aggregated`` serialises -- i.e.
            the output of ``EvalAcc.dump`` enriched with ``config``, model
            metadata, ``git_hash`` and ``date``.

    Returns:
        A single source-agnostic document describing the run, its model, and
        per-benchmark metrics.
    """
    per_task = results.get("results", {}) or {}
    versions = results.get("versions", {}) or {}
    n_shot = results.get("n-shot", {}) or {}
    n_samples = results.get("n-samples", {}) or {}
    higher_is_better = results.get("higher_is_better", {}) or {}

    benchmarks: list[dict[str, Any]] = []
    for task_name in sorted(per_task):
        task_result = per_task[task_name]
        benchmarks.append(
            {
                "benchmark": task_name,
                "alias": task_result.get("alias", task_name),
                "version": versions.get(task_name),
                "n_shot": n_shot.get(task_name),
                "n_samples": n_samples.get(task_name),
                "metrics": _collect_task_metrics(
                    task_result, higher_is_better.get(task_name)
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "framework": "lm-evaluation-harness",
            "git_hash": results.get("git_hash"),
        },
        "model": _model_section(results),
        "evaluation": {
            "date": results.get("date"),
            "total_evaluation_time_seconds": results.get(
                "total_evaluation_time_seconds"
            ),
            "system_instruction_sha": results.get("system_instruction_sha"),
            "chat_template_sha": results.get("chat_template_sha"),
            "fewshot_as_multiturn": results.get("fewshot_as_multiturn"),
        },
        "benchmarks": benchmarks,
    }


def unified_schema_path(aggregated_path: Path) -> Path:
    """Companion file path for the unified document next to the aggregated one.

    ``results_<date>.json`` -> ``results_unified_<date>.json``; any other stem
    gets a ``unified_`` prefix so the two files always sit side by side.
    """
    name = aggregated_path.name
    if name.startswith("results_"):
        new_name = "results_unified_" + name[len("results_") :]
    else:
        new_name = "unified_" + name
    return aggregated_path.with_name(new_name)


def write_unified_results(
    results: dict[str, Any],
    aggregated_path: Path,
    default: Any = None,
) -> Path:
    """Serialise the unified document beside an aggregated results file.

    Args:
        results: The harness aggregated results dict.
        aggregated_path: Path the aggregated JSON was written to.
        default: ``json.dumps`` fallback serialiser for non-serialisable
            objects (the harness passes ``handle_non_serializable``).

    Returns:
        The path the unified document was written to.
    """
    unified = to_unified_schema(results)
    out_path = unified_schema_path(aggregated_path)
    out_path.write_text(
        json.dumps(unified, indent=2, default=default, ensure_ascii=False),
        encoding="utf-8",
    )
    return out_path
