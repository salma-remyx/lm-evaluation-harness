"""Continuous-score best-of-N selection filters.

Adapted from "LLM-as-a-Verifier: A General-Purpose Verification Framework"
(https://arxiv.org/abs/2607.05391). The paper's core insight is that a verifier
should produce a *continuous* correctness score for each candidate solution --
computed as the expectation over the distribution of scoring-token logits --
rather than a discrete judge label, and that ranking candidates by these
continuous scores (best-of-N) gives better separation between correct and
incorrect solutions than plain majority voting.

This module implements that ranking mechanism as a drop-in selection filter,
resolving the long-standing ``arg_max`` TODO in ``selection.py``. Two pieces are
faithful to the paper:

* :func:`expected_score` -- the continuous score as ``E[value]`` under the
  softmax over scoring-token logits. A real verifier LLM (reachable through any
  harness backend, e.g. LiteLLM) can return a ``{value: logit}`` distribution
  over its scoring vocabulary and this collapses it to a calibrated scalar.
* :class:`ScoredSelectionFilter` -- ranks the N candidates per document by their
  continuous score and keeps the argmax, with optional *repeated evaluation*
  (averaging for variance reduction) and *criteria decomposition* (averaging
  independent per-criterion scorers for complexity reduction), the paper's two
  additional verification-scaling axes.

Scoped out (belongs in downstream work, not this filter): the learned verifier
LLM itself, the Claude Code progress extension, RL dense-reward feedback, and
the paper's benchmark suites. To keep the filter usable inside the harness today
without wiring a verifier backend, the *default* scorer substitutes the learned
verifier with a parameter-free self-consistency proxy (mean token-Jaccard
agreement of a candidate with the rest of the pool) -- correct candidates tend
to agree with one another, so consensus is a cheap continuous verification
signal. Inject a real verifier by passing a ``scorer`` callable.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

from lm_eval.api.filter import Filter
from lm_eval.api.registry import register_filter


# A scorer maps (candidate, doc) -> either a scalar reward or a {value: logit}
# distribution over scoring tokens (which is reduced with ``expected_score``).
Scorer = Callable[[str, dict], float | dict[float, float]]

_TOKEN_RE = re.compile(r"\w+")


def expected_score(score_logits: dict[float, float]) -> float:
    """Continuous verification score: ``E[value]`` under ``softmax(logits)``.

    ``score_logits`` maps each scoring token's numeric value (e.g. a 0..10
    granularity scale) to its logit (or log-probability). This is the paper's
    probabilistic scoring formulation -- taking the expectation over the scoring
    distribution rather than the argmax token yields a smooth, calibrated score.
    """
    if not score_logits:
        raise ValueError("score_logits must contain at least one scoring token")
    values = list(score_logits.keys())
    logits = list(score_logits.values())
    # Softmax in a shift-stable form.
    hi = max(logits)
    weights = [math.exp(logit - hi) for logit in logits]
    total = math.fsum(weights)
    return math.fsum(v * w for v, w in zip(values, weights, strict=True)) / total


def _token_set(text: object) -> frozenset:
    return frozenset(_TOKEN_RE.findall(str(text).lower()))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def consensus_scores(candidates: Sequence[str]) -> list[float]:
    """Parameter-free self-consistency proxy for the learned verifier.

    Scores each candidate by its mean token-Jaccard similarity to the other
    candidates in the pool, yielding a continuous value in ``[0, 1]``. With a
    single candidate the score is ``1.0``.
    """
    n = len(candidates)
    if n == 1:
        return [1.0]
    token_sets = [_token_set(c) for c in candidates]
    scores = []
    for i in range(n):
        agreement = math.fsum(
            _jaccard(token_sets[i], token_sets[j]) for j in range(n) if j != i
        )
        scores.append(agreement / (n - 1))
    return scores


def _reduce(raw: float | dict[float, float]) -> float:
    """Collapse a raw scorer output to a scalar, applying the paper's
    expectation-over-logits reduction when given a scoring distribution.
    """
    if isinstance(raw, dict):
        return expected_score(raw)
    return float(raw)


@register_filter("arg_max")
class ScoredSelectionFilter(Filter):
    """Select the best of N candidate responses by continuous verifier score.

    Args:
        scorer: callable ``(candidate, doc) -> float | {value: logit}``. When
            omitted, the parameter-free consensus proxy is used.
        repeats: number of times to evaluate each candidate; scores are averaged
            (repeated-evaluation scaling, for variance reduction). Meaningful for
            stochastic / sampled verifier scorers.
        criteria: optional list of independent scorers whose scores are averaged
            per candidate (criteria-decomposition scaling, for complexity
            reduction). May be combined with ``scorer``.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        repeats: int = 1,
        criteria: Sequence[Scorer] | None = None,
        **kwargs,
    ) -> None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        self.scorer = scorer
        self.repeats = int(repeats)
        self.criteria = list(criteria) if criteria else None
        super().__init__(**kwargs)

    def _score_one(self, scorer: Scorer, candidate: str, doc: dict) -> float:
        # Repeated evaluation: average the (possibly stochastic) scorer.
        samples = [_reduce(scorer(candidate, doc)) for _ in range(self.repeats)]
        return math.fsum(samples) / len(samples)

    def _candidate_scores(self, candidates: Sequence[str], doc: dict) -> list[float]:
        # Default path: no explicit verifier -> parameter-free consensus proxy.
        if self.scorer is None and self.criteria is None:
            return consensus_scores(candidates)

        scorers: list[Scorer] = []
        if self.scorer is not None:
            scorers.append(self.scorer)
        if self.criteria is not None:
            scorers.extend(self.criteria)

        scores = []
        for cand in candidates:
            # Criteria decomposition: average the per-criterion continuous scores.
            per_criterion = [self._score_one(s, cand, doc) for s in scorers]
            scores.append(math.fsum(per_criterion) / len(per_criterion))
        return scores

    def apply(self, resps, docs):
        """For each document, rank its candidate responses by continuous verifier
        score and keep the argmax (returned as a single-element list, mirroring
        ``MajorityVoteFilter``).
        """
        docs = list(docs)
        selected = []
        for candidates, doc in zip(resps, docs, strict=True):
            candidates = list(candidates)
            if not candidates:
                selected.append([])
                continue
            scores = self._candidate_scores(candidates, doc)
            best = max(range(len(candidates)), key=lambda i: scores[i])
            selected.append([candidates[best]])
        return selected
