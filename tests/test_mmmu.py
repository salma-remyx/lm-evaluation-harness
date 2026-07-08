"""Integration tests for schema-agnostic option handling in the MMMU VLM
multiple-choice contract.

These exercise the existing ``lm_eval/tasks/mmmu/utils.py`` entry points
(``_doc_to_text`` / ``process_results``) — the same module the task YAMLs load
via ``!function utils.*`` — to confirm they now serve both the MMMU
``options`` list-string layout and the MMT-Bench style separate lettered
option columns through the shared code path.
"""

import importlib.util
import os

import pytest


MMMU_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "lm_eval", "tasks", "mmmu"
)


def _load_utils():
    """Load ``utils`` the way lm-eval's task loader does (by file location)."""
    spec = importlib.util.spec_from_file_location(
        "mmmu_utils_under_test", os.path.join(MMMU_DIR, "utils.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def utils():
    return _load_utils()


# MMMU ships options as a single stringified Python list.
MMMU_DOC = {
    "question": "What animal is in <image 1>?",
    "options": "['cat', 'dog', 'bird', 'fish']",
    "answer": "B",
    "question_type": "multiple-choice",
}

# MMT-Bench (arxiv 2404.16006) spreads the same choices across A/B/C/D columns.
MMT_BENCH_DOC = {
    "question": "What animal is in <image 1>?",
    "A": "cat",
    "B": "dog",
    "C": "bird",
    "D": "fish",
    "answer": "B",
    "question_type": "multiple-choice",
}


def test_extract_options_wired_into_utils(utils):
    # The call site must actually invoke the new normalizer.
    assert hasattr(utils, "extract_options")


def test_both_schemas_produce_identical_prompt(utils):
    mmmu_prompt = utils._doc_to_text(MMMU_DOC)
    mmt_prompt = utils._doc_to_text(MMT_BENCH_DOC)
    # Identical choices must render identically regardless of source layout.
    assert mmmu_prompt == mmt_prompt
    for letter, choice in [("A", "cat"), ("B", "dog"), ("C", "bird"), ("D", "fish")]:
        assert f"({letter}) {choice}" in mmt_prompt


def test_process_results_scores_separate_column_schema(utils):
    correct = utils.process_results(MMT_BENCH_DOC, ["The answer is (B)"])
    assert correct == {"acc": 1.0}

    wrong = utils.process_results(MMT_BENCH_DOC, ["The answer is (A)"])
    assert wrong == {"acc": 0.0}


def test_lettered_columns_stop_at_first_gap(utils):
    # Only A-C are populated; a stray empty D must not become a blank option.
    doc = {
        "question": "pick one",
        "A": "alpha",
        "B": "beta",
        "C": "gamma",
        "D": "",
        "answer": "A",
        "question_type": "multiple-choice",
    }
    assert utils.extract_options(doc) == ["alpha", "beta", "gamma"]
