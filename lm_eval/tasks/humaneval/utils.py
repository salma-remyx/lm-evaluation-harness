from .code_quality import code_quality_metrics


compute_ = None


def _get_code_eval():
    """Lazily load and self-test the HF ``code_eval`` metric.

    Done on first use (rather than at import) so this module stays importable
    without the ``evaluate`` dependency / code-execution sandbox — e.g. for
    unit tests of the quality metrics below. The original self-test that checks
    code execution is enabled is preserved, just deferred to first call.
    """
    global compute_
    if compute_ is None:
        import evaluate as hf_evaluate

        compute_ = hf_evaluate.load("code_eval")
        test_cases = ["assert add(2, 3)==5"]
        candidates = [["def add(a,b): return a*b"]]
        compute_.compute(references=test_cases, predictions=candidates, k=[1])
    return compute_


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] = None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    res = _get_code_eval().compute(
        references=references,
        predictions=predictions,
        k=k,
    )[0]
    # Beyond correctness (RACE): report multi-dimensional static code quality
    # alongside pass@k. These extra keys are expanded into their own aggregated
    # metrics by the task harness.
    res.update(code_quality_metrics(predictions))
    return res


def build_predictions(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[doc["prompt"] + r for r in resp] for resp, doc in zip(resps, docs)]


def build_predictions_instruct(
    resps: list[list[str]], docs: list[dict]
) -> list[list[str]]:
    return [
        [
            doc["prompt"] + (r if r.find("```") == -1 else r[: r.find("```")])
            for r in resp
        ]
        for resp, doc in zip(resps, docs)
    ]
