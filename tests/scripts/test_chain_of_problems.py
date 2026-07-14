import json

# Imports the existing call-site module (build_benchmark) plus the new
# capability module, and exercises the chain-of-problems wiring end to end.
from scripts.build_benchmark import build_chained_benchmark
from scripts.chain_of_problems import (
    EXAMPLE_SEED,
    build_chain_benchmark,
    chain_problems,
    safe_arith,
)


# Two atomic GSM8K-style problems. Chained forward:
#   P1: 10 apples - 4  = 6
#   P2: 6 * 3 + 1      = 19   (input x <- previous answer)
SEED_CHAIN = [
    {
        "template": "Maria has {x} apples and gives away 4. How many remain?",
        "expr": "x - 4",
        "default_input": 10,
        "name": "apples",
    },
    {
        "template": "Each of {x} baskets holds 3 pears plus 1 spare. How many pears total?",
        "expr": "x * 3 + 1",
        "default_input": 5,
        "name": "pears",
    },
]


def test_safe_arith_rejects_non_arithmetic():
    assert safe_arith("x * 3 + 1", 6) == 19
    for bad in ("__import__('os')", "open('/etc/passwd')", "y + 1"):
        try:
            safe_arith(bad, 1)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad!r}")


def test_forward_chain_propagates_answer():
    doc = chain_problems(SEED_CHAIN, direction="forward")
    assert doc["intermediate"] == [6, 19]
    assert doc["final_answer"] == 19
    assert doc["answer"].strip().endswith("#### 19")
    # Base problem shows its concrete value; linked problem references A1.
    assert "10 apples" in doc["question"]
    assert "Let A1 be the answer to Problem 1" in doc["question"]
    assert "Give the final numerical answer to Problem 2." in doc["question"]


def test_backward_chain_reverses_presentation():
    doc = chain_problems(SEED_CHAIN, direction="backward")
    # Same computation, so the final answer is unchanged...
    assert doc["final_answer"] == 19
    # ...but the asked-for problem (the dependent one) is presented first and
    # references an answer that appears later in the text.
    assert "Give the final numerical answer to Problem 1." in doc["question"]
    assert "Let A2 be the answer to Problem 2" in doc["question"]


def test_build_chain_benchmark_multiple_chains_and_truncation():
    records = build_chain_benchmark({"chains": [SEED_CHAIN]}, direction="forward")
    assert len(records) == 1 and records[0]["final_answer"] == 19
    # chain_length truncates to the base problem only.
    truncated = build_chain_benchmark([SEED_CHAIN], chain_length=1)
    assert truncated[0]["final_answer"] == 6


def test_example_seed_builds():
    records = build_chain_benchmark(EXAMPLE_SEED, direction="forward")
    assert records[0]["intermediate"] == [32, 8, 24]
    assert records[0]["final_answer"] == 24


def test_build_chained_benchmark_writes_gsm8k_style_jsonl(tmp_path):
    seed_path = tmp_path / "seed.json"
    out_path = tmp_path / "chain.jsonl"
    seed_path.write_text(json.dumps({"chains": [SEED_CHAIN]}))

    records = build_chained_benchmark(
        str(seed_path), str(out_path), direction="forward", chain_length=0
    )

    assert len(records) == 1
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    written = json.loads(lines[0])
    assert written["final_answer"] == 19
    assert written["answer"].strip().endswith("#### 19")
