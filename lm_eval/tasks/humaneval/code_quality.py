"""Multi-dimensional code-quality scoring for generated Python solutions.

Adapted from RACE — "Beyond Correctness: Benchmarking Multi-dimensional Code
Generation for Large Language Models" (Zheng et al., 2024,
https://arxiv.org/abs/2407.11470). RACE argues that judging code-generation
models by functional correctness alone is both incomplete (it ignores qualities
that matter in real development) and contamination-prone. It scores generations
along additional dimensions such as readability, maintainability and complexity.

This module implements a lightweight, dependency-free, static approximation of
those dimensions so the HumanEval task can report them alongside ``pass@k``
without executing the candidate code. Every score is in ``[0.0, 1.0]`` and is
oriented so that higher is better, matching the ``higher_is_better`` contract of
the ``pass@k`` metric it is reported next to.

Out of scope (by design): RACE's customised, requirement-conditioned prompts and
its LLM-judge dimensions. Those need bespoke datasets / a judge model and do not
fit the existing ``generate_until`` + execution-metric contract. The value here
is the "beyond correctness" signal computed cheaply over the same generations.
"""

import ast
import re
from collections.abc import Iterable


# Identifiers Python defines for us / that snake_case linting should ignore.
_DUNDER = re.compile(r"^__.*__$")
_SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")
_MAX_LINE_LENGTH = 79  # PEP 8 soft limit; lines beyond this hurt readability.


def _coerce_to_code_list(predictions) -> list[str]:
    """Flatten the various shapes ``pass_at_k`` can receive into code strings.

    ``pass_at_k`` is called with ``predictions`` shaped as ``list[list[str]]``
    (one inner list of candidate solutions per problem), but this helper also
    accepts a single ``str`` or a flat ``list[str]`` so it is reusable and easy
    to test.
    """
    if isinstance(predictions, str):
        return [predictions]
    out: list[str] = []
    for item in predictions:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Iterable):
            out.extend(s for s in item if isinstance(s, str))
    return out


def _readability_score(code: str) -> float:
    """Lexical readability: line length, naming and documentation.

    Purely token/regex based so it still yields a useful signal for partial or
    syntactically invalid generations (a common case for weaker models).
    """
    lines = [ln for ln in code.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    within_limit = sum(1 for ln in lines if len(ln) <= _MAX_LINE_LENGTH)
    line_len_score = within_limit / len(lines)

    identifiers = re.findall(r"\bdef\s+(\w+)|\b(\w+)\s*=", code)
    names = [n for pair in identifiers for n in pair if n]
    if names:
        well_named = sum(
            1
            for n in names
            if _DUNDER.match(n) or (_SNAKE_CASE.match(n) and len(n) > 1)
        )
        naming_score = well_named / len(names)
    else:
        naming_score = 1.0

    documented = (
        ('"""' in code)
        or ("'''" in code)
        or any(ln.lstrip().startswith("#") for ln in lines)
    )
    doc_score = 1.0 if documented else 0.0

    # Readability is dominated by line discipline and naming; documentation is a
    # smaller bonus.
    return 0.45 * line_len_score + 0.45 * naming_score + 0.10 * doc_score


# AST node types that introduce a branch / decision point (McCabe complexity).
_BRANCH_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.IfExp,
    ast.comprehension,
    ast.Assert,
)


def _cyclomatic_complexity(tree: ast.AST) -> int:
    count = 1
    for node in ast.walk(tree):
        if isinstance(node, _BRANCH_NODES):
            count += 1
        elif isinstance(node, ast.BoolOp):
            # `a and b and c` adds two decision points.
            count += len(node.values) - 1
    return count


def _complexity_score(tree: ast.AST) -> float:
    """Map cyclomatic complexity to a ``[0, 1]`` simplicity score.

    Complexity up to ~5 is treated as fully acceptable; beyond that the score
    decays smoothly so deeply branched solutions are penalised.
    """
    complexity = _cyclomatic_complexity(tree)
    if complexity <= 5:
        return 1.0
    return max(0.0, 5.0 / complexity)


def _max_nesting_depth(tree: ast.AST, depth: int = 0) -> int:
    nesting_types = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.With,
        ast.AsyncWith,
        ast.Try,
    )
    max_depth = depth
    for child in ast.iter_child_nodes(tree):
        child_depth = depth + 1 if isinstance(child, nesting_types) else depth
        max_depth = max(max_depth, _max_nesting_depth(child, child_depth))
    return max_depth


def _maintainability_score(tree: ast.AST, code: str) -> float:
    """Maintainability from nesting depth and function length."""
    depth = _max_nesting_depth(tree)
    # Depth up to 2 is fine; each extra level costs 0.2.
    depth_score = max(0.0, 1.0 - 0.2 * max(0, depth - 2))

    body_lines = [ln for ln in code.splitlines() if ln.strip()]
    n_lines = len(body_lines)
    # A focused HumanEval-style function is short; 30+ lines starts to hurt.
    length_score = 1.0 if n_lines <= 30 else max(0.0, 1.0 - (n_lines - 30) / 60.0)

    return 0.6 * depth_score + 0.4 * length_score


def code_quality_metrics(predictions) -> dict:
    """Return RACE-style quality dimensions averaged over ``predictions``.

    Accepts the ``list[list[str]]`` shape used by ``humaneval.utils.pass_at_k``
    (as well as a flat list or single string) and returns a dict with keys
    ``readability``, ``complexity``, ``maintainability`` and an aggregate
    ``code_quality``. Each value is a float in ``[0.0, 1.0]``, higher is better.

    The function never raises: invalid / unparsable code falls back to the
    lexical readability signal with neutral structural scores, so it is safe to
    call inside a metric over arbitrary model output.
    """
    codes = _coerce_to_code_list(predictions)
    if not codes:
        return {
            "readability": 0.0,
            "complexity": 0.0,
            "maintainability": 0.0,
            "code_quality": 0.0,
        }

    readability, complexity, maintainability = [], [], []
    for code in codes:
        readability.append(_readability_score(code))
        try:
            tree = ast.parse(code)
            complexity.append(_complexity_score(tree))
            maintainability.append(_maintainability_score(tree, code))
        except (SyntaxError, ValueError, RecursionError):
            # Unparsable generation: stay neutral on structural dimensions
            # rather than rewarding or unfairly punishing it.
            complexity.append(0.5)
            maintainability.append(0.5)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    r, c, m = _mean(readability), _mean(complexity), _mean(maintainability)
    return {
        "readability": r,
        "complexity": c,
        "maintainability": m,
        "code_quality": (r + c + m) / 3.0,
    }
