"""Benchmark-level macro-aggregation of heterogeneous DrBenchmark results.

Adapted from *DrBenchmark: A Large Language Understanding Evaluation Benchmark
for French Biomedical Domain* (Labrak et al., 2024, https://arxiv.org/abs/2402.13432).

DrBenchmark aggregates diverse downstream tasks (classification, NER, POS, NLI,
STS, QA, ...) into a single benchmark and reports one macro-averaged score so a
model can be ranked from many perspectives at once, even though the constituent
tasks use different primary metrics.

This module reproduces that *result* for the harness' own results JSON: given the
standard ``{"results": {...}, "versions": {...}}`` payload, it selects one primary
metric per task and macro-averages the comparable ([0, 1]-scaled) ones into a
single benchmark score. Unbounded metrics (perplexity, bits-per-byte, ...) are
excluded because averaging them with accuracy-like scores is not meaningful.

The macro-average across the ``drbenchmark`` subtasks is also expressed natively
in ``_drbenchmark.yaml`` via ``aggregate_metric_list`` (unweighted mean of
``acc``); the helpers here support post-hoc aggregation over an arbitrary,
heterogeneous results payload.
"""

# Metric-name suffix used by the harness to mark standard-error columns.
STDERR_SUFFIX = "_stderr"

# Keys that appear alongside metrics in a result entry but are not scores.
_NON_METRIC_KEYS = frozenset({"alias"})

# Metrics that are not bounded to [0, 1] and must not be macro-averaged with
# accuracy/F1-style scores.
UNBOUNDED_METRICS = frozenset(
    {
        "ppl",
        "perplexity",
        "word_perplexity",
        "byte_perplexity",
        "bits_per_byte",
    }
)

# Preferred primary metric per task, in descending priority. The first match in a
# task's result entry is used; otherwise the first comparable metric is taken.
_PRIMARY_METRIC_PREFERENCE = (
    "acc",
    "acc_norm",
    "exact_match",
    "f1",
    "mcc",
    "spearman",
    "pearson",
    "bleu",
    "rouge",
)


def _is_score(name, value):
    """Return True if ``name``/``value`` is a numeric, comparable metric."""
    if name in _NON_METRIC_KEYS or name.endswith(STDERR_SUFFIX):
        return False
    if name in UNBOUNDED_METRICS:
        return False
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def primary_metric(task_metrics):
    """Pick the primary (name, value) pair for a single task's result entry.

    Preference order is applied first (see ``_PRIMARY_METRIC_PREFERENCE``); if
    none of the preferred metrics are present the first comparable metric wins.
    Returns ``None`` when the task has no comparable metric.
    """
    comparable = {
        name: value for name, value in task_metrics.items() if _is_score(name, value)
    }
    if not comparable:
        return None
    for preferred in _PRIMARY_METRIC_PREFERENCE:
        if preferred in comparable:
            return preferred, comparable[preferred]
    name = next(iter(comparable))
    return name, comparable[name]


def benchmark_macro_average(results):
    """Macro-average the primary metric across every task in ``results``.

    ``results`` is the ``result_dict["results"]`` mapping of
    ``task_name -> {metric: value}``. Returns a summary dict::

        {"benchmark_score": float, "num_tasks": int, "per_task": {...}}

    where ``benchmark_score`` is the unweighted mean of the per-task primary
    scores and ``per_task`` maps each contributing task to its ``(metric, value)``.
    Returns ``None`` when no task contributes a comparable metric.
    """
    per_task = {}
    for task_name, task_metrics in results.items():
        if not isinstance(task_metrics, dict):
            continue
        selected = primary_metric(task_metrics)
        if selected is not None:
            per_task[task_name] = selected

    if not per_task:
        return None

    total = sum(value for _, value in per_task.values())
    return {
        "benchmark_score": total / len(per_task),
        "num_tasks": len(per_task),
        "per_task": per_task,
    }


def benchmark_score_row(result_dict):
    """Build a macro-average summary row for ``make_table``.

    The returned list matches the ``make_table`` header layout
    ``[Task, Version, Metric, Value, "", Stderr]`` and scales the score by 100 to
    match the default (non-percent) rows. Returns ``None`` when the results carry
    no comparable metric to aggregate.
    """
    summary = benchmark_macro_average(result_dict.get("results", {}))
    if summary is None:
        return None
    return [
        "**benchmark (macro-avg)**",
        "",
        "score",
        f"{summary['benchmark_score'] * 100:.2f}",
        "",
        f"n={summary['num_tasks']}",
    ]
