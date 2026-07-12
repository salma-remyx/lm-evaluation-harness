import math

import pytest

# Import through the existing (non-new) filter package + registry to exercise the
# wiring edit: importing lm_eval.filters must register the new "arg_max" filter.
from lm_eval.api.instance import Instance
from lm_eval.api.registry import get_filter
from lm_eval.filters import build_filter_ensemble
from lm_eval.filters.scored_selection import (
    ScoredSelectionFilter,
    consensus_scores,
    expected_score,
)


def test_arg_max_registered_via_package_import():
    # The __init__ import of `scored_selection` must have registered the filter.
    assert get_filter("arg_max") is ScoredSelectionFilter


def test_expected_score_is_expectation_over_softmax():
    # Uniform logits over {0, 10} -> expectation 5.0.
    assert expected_score({0.0: 1.0, 10.0: 1.0}) == pytest.approx(5.0)
    # A single scoring token collapses to its value.
    assert expected_score({7.0: 0.0}) == pytest.approx(7.0)
    # Heavily favouring the high token pushes the expectation toward it.
    assert expected_score({0.0: 0.0, 1.0: 10.0}) == pytest.approx(1.0, abs=1e-3)
    with pytest.raises(ValueError):
        expected_score({})


def test_consensus_scores_reward_the_majority_candidate():
    candidates = ["the answer is 4", "the answer is 4", "totally unrelated text"]
    scores = consensus_scores(candidates)
    # The two agreeing candidates outscore the outlier.
    assert scores[0] == scores[1]
    assert scores[0] > scores[2]
    assert consensus_scores(["solo"]) == [1.0]


def test_filter_selects_argmax_with_custom_scorer():
    filt = ScoredSelectionFilter(scorer=lambda cand, doc: len(cand))
    resps = [["a", "bbbb", "cc"]]
    docs = [{"question": "q"}]
    assert filt.apply(resps, docs) == [["bbbb"]]


def test_filter_reduces_scoring_distribution_with_expected_score():
    # Scorer returns a {value: logit} distribution -> reduced via expected_score.
    def scorer(cand, doc):
        return {0.0: 0.0, 10.0: 0.0} if cand == "x" else {0.0: 0.0, 10.0: 10.0}

    filt = ScoredSelectionFilter(scorer=scorer)
    assert filt.apply([["x", "y"]], [{}]) == [["y"]]


def test_criteria_decomposition_averages_scorers():
    filt = ScoredSelectionFilter(
        criteria=[lambda c, d: 1.0 if "safe" in c else 0.0, lambda c, d: len(c)]
    )
    # "safe!" -> (1 + 5)/2 = 3.0 ; "longer word" -> (0 + 11)/2 = 5.5 -> longer wins.
    assert filt.apply([["safe!", "longer word"]], [{}]) == [["longer word"]]


def test_end_to_end_through_filter_ensemble():
    # Drive the real call site: build_filter_ensemble -> FilterEnsemble.apply.
    ensemble = build_filter_ensemble("verifier", [["arg_max", None]])
    instances = [
        Instance(
            request_type="generate_until",
            doc={"question": "2+2?"},
            arguments=("2+2?",),
            idx=0,
            resps=["the result is 4", "the result is 4", "banana pancakes"],
        )
    ]
    ensemble.apply(instances)
    # Consensus best-of-N keeps the agreeing majority answer.
    assert instances[0].filtered_resps["verifier"] == ["the result is 4"]


def test_repeats_must_be_positive():
    with pytest.raises(ValueError):
        ScoredSelectionFilter(repeats=0)


def test_repeated_evaluation_averages_stochastic_scorer():
    samples = iter([1.0, 3.0])  # two samples for one candidate -> mean 2.0
    filt = ScoredSelectionFilter(scorer=lambda c, d: next(samples), repeats=2)
    assert filt._score_one(filt.scorer, "x", {}) == math.fsum([1.0, 3.0]) / 2
