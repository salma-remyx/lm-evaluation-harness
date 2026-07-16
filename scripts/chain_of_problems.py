"""Compose atomic math word problems into chains-of-problems.

Adapted from *Scheherazade: Evaluating Chain-of-Thought Math Reasoning in
LLMs with Chain-of-Problems* (https://arxiv.org/abs/2410.00151). Scheherazade
turns saturated single-step benchmarks such as GSM8K into a harder,
multi-hop benchmark by *chaining* independent problems: the numeric answer of
one problem is threaded into the next, so a model must carry an intermediate
result across several reasoning steps to reach the final answer.

This module implements that core chaining mechanism in a self-contained,
parameter-free way. Each atomic problem is a plain dict:

    {
        "template": "Maria has {x} apples and gives away 4. How many remain?",
        "expr": "x - 4",          # answer as a function of the linked input x
        "default_input": 10,       # value used for the chain's base problem
        "name": "apples"           # optional label
    }

``expr`` is evaluated with a restricted arithmetic evaluator (no ``eval``), so
seeds stay pure JSON and no learned model is needed to recompute answers after
substitution. The resulting records carry a GSM8K-style ``#### <answer>``
target, so they plug directly into the harness's existing ``exact_match`` +
regex-extraction scoring contract (see ``lm_eval/tasks/gsm8k``).

Scoped to the two chaining *directions* the paper defines (forward / backward)
over permissive GSM8K-style seed data. The paper's LLM-based problem rewriting
and its separate manual-benchmark corpus are intentionally out of scope: this
composes structured seeds you provide.
"""

import ast
import operator
from typing import Any


Number = int | float

# Reference seed illustrating the expected shape. Pass a JSON file of this form
# to ``build_benchmark.py --chain_of_problems --seed_path <file>``. Forward
# chaining here computes 50 - 18 = 32, 32 / 4 = 8, 8 * 3 = 24.
EXAMPLE_SEED: dict[str, Any] = {
    "chains": [
        [
            {
                "template": "A bakery starts with {x} muffins and sells 18. How many remain?",
                "expr": "x - 18",
                "default_input": 50,
                "name": "muffins",
            },
            {
                "template": "The {x} muffins are split evenly onto 4 trays. How many per tray?",
                "expr": "x / 4",
                "default_input": 32,
                "name": "trays",
            },
            {
                "template": "Each of the {x} muffins on a tray is cut into 3 pieces. How many pieces?",
                "expr": "x * 3",
                "default_input": 8,
                "name": "pieces",
            },
        ]
    ]
}

# Whitelisted binary / unary operators for the restricted arithmetic evaluator.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_arith(expr: str, x: Number) -> Number:
    """Evaluate an arithmetic ``expr`` in the single variable ``x``.

    Only numeric literals, the variable ``x`` and the whitelisted operators in
    ``_BIN_OPS`` / ``_UNARY_OPS`` are permitted. Anything else (function calls,
    attribute access, other names) raises ``ValueError``; non-numeric literals
    raise ``TypeError``. This keeps problem seeds JSON-serializable without
    exposing ``eval``.
    """

    def _eval(node: ast.AST) -> Number:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise TypeError(f"non-numeric constant: {node.value!r}")
            return node.value
        if isinstance(node, ast.Name):
            if node.id != "x":
                raise ValueError(f"unknown name: {node.id!r}")
            return x
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"unsupported expression element: {ast.dump(node)}")

    return _eval(ast.parse(expr, mode="eval"))


def _as_int(value: Number) -> Number:
    """Render whole-valued floats as ints so targets match GSM8K formatting."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def solve_problem(problem: dict[str, Any], x: Number) -> Number:
    """Return the answer of ``problem`` when its linked input equals ``x``."""
    return safe_arith(problem["expr"], x)


def chain_problems(
    problems: list[dict[str, Any]], direction: str = "forward"
) -> dict[str, Any]:
    """Compose ``problems`` (in computation order) into a single chained doc.

    The first entry is the independent base problem (solved with its own
    ``default_input``); every later entry's linked input is the previous
    problem's answer. ``direction`` controls only *presentation*:

    * ``"forward"`` presents problems in computation order — each references
      the answer of the problem just above it (a backward text reference).
    * ``"backward"`` presents them reversed, so the asked-for problem appears
      first and references answers that appear *later*, forcing the solver to
      resolve the chain bottom-up.

    Returns a record with the composed ``question``, a GSM8K-style ``answer``
    solution string ending in ``#### <final>``, the numeric ``final_answer``,
    the per-step ``intermediate`` answers, and the ``direction``.
    """
    if not problems:
        raise ValueError("chain_problems requires at least one problem")
    if direction not in ("forward", "backward"):
        raise ValueError(f"unknown direction: {direction!r}")

    # Propagate answers along the dependency chain (always computation order).
    intermediate: list[Number] = []
    prev: Number = 0
    for idx, problem in enumerate(problems):
        x = problem["default_input"] if idx == 0 else prev
        prev = solve_problem(problem, x)
        intermediate.append(prev)
    final = _as_int(prev)

    # Present in computation order (forward) or reversed (backward).
    order = list(range(len(problems)))
    if direction == "backward":
        order.reverse()
    pos_of = {comp_idx: pos for pos, comp_idx in enumerate(order)}

    lines: list[str] = []
    for pos, comp_idx in enumerate(order):
        problem = problems[comp_idx]
        label = f"Problem {pos + 1}"
        if comp_idx == 0:
            # Base problem: uses its own concrete input value.
            body = problem["template"].format(x=_as_int(problem["default_input"]))
            lines.append(f"{label}: {body}")
        else:
            # Linked problem: input is the answer of the previous problem in
            # computation order, which lives at presentation position pos_of.
            ref_pos = pos_of[comp_idx - 1] + 1
            ref = f"A{ref_pos}"
            body = problem["template"].format(x=ref)
            lines.append(
                f"{label}: Let {ref} be the answer to Problem {ref_pos}. {body}"
            )

    asked = pos_of[len(problems) - 1] + 1
    lines.append(f"Give the final numerical answer to Problem {asked}.")
    question = "\n".join(lines)

    steps = "\n".join(
        f"A{pos + 1} = {_as_int(intermediate[comp_idx])}"
        for pos, comp_idx in enumerate(order)
    )
    answer = f"{steps}\n#### {final}"

    return {
        "question": question,
        "answer": answer,
        "final_answer": final,
        "intermediate": [_as_int(v) for v in intermediate],
        "direction": direction,
    }


def build_chain_benchmark(
    seed: list[Any] | dict[str, Any],
    direction: str = "forward",
    chain_length: int = 0,
) -> list[dict[str, Any]]:
    """Build chained-problem records from a ``seed``.

    ``seed`` is either a list of chains (each chain a list of problem dicts) or
    a dict with a ``"chains"`` key of the same shape. ``chain_length``, when
    positive, truncates each chain to that many problems. One record is
    produced per chain.
    """
    if isinstance(seed, dict):
        chains = seed["chains"]
    else:
        chains = seed

    records: list[dict[str, Any]] = []
    for chain in chains:
        problems = list(chain)
        if chain_length and chain_length > 0:
            problems = problems[:chain_length]
        records.append(chain_problems(problems, direction=direction))
    return records
