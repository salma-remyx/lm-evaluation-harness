"""Schema-agnostic multiple-choice option extraction for VLM benchmarks.

The MMMU multiple-choice contract stores answer options as a single
stringified Python list under an ``options`` column (e.g. ``"['a', 'b']"``).
Other comprehensive VLM benchmarks that reuse the same image + question +
A/B/C/D + answer schema instead spread the choices across separate lettered
columns. MMT-Bench (Ying et al., 2024, https://arxiv.org/abs/2404.16006) is a
representative example of the latter layout.

This helper normalizes either layout to a plain ``list[str]`` of option texts
so the shared ``utils`` prompt/scoring code can serve both without branching on
the dataset. Keeping the normalization in one place lets new separate-column
VLM multiple-choice benchmarks drop onto the existing MMMU contract with only a
task YAML, rather than a fork of the parsing logic.
"""

import ast
from string import ascii_uppercase


def _coerce_option_list(raw):
    """Return ``raw`` as a list of option strings.

    ``raw`` may already be a list/tuple (HF datasets column) or a stringified
    Python list as shipped by MMMU. Anything else is treated as a single
    option.
    """
    if isinstance(raw, (list, tuple)):
        return [str(opt) for opt in raw]
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return [str(opt) for opt in parsed]
        return [raw]
    return [str(raw)]


def _lettered_options(doc):
    """Collect consecutive ``A``, ``B``, ``C`` ... columns as an option list.

    Stops at the first missing/empty letter so trailing distractor columns that
    a given example does not use are not emitted as blank choices.
    """
    options = []
    for letter in ascii_uppercase:
        if letter not in doc:
            break
        value = doc[letter]
        if value is None:
            break
        text = str(value).strip()
        if text == "":
            break
        options.append(str(value))
    return options


def extract_options(doc):
    """Extract the multiple-choice option texts from a VLM benchmark ``doc``.

    Supports both the MMMU ``options`` list-string layout and the MMT-Bench
    style separate ``A``/``B``/``C``/``D`` columns. Raises ``KeyError`` if
    neither layout is present so misconfigured tasks fail loudly instead of
    silently scoring against an empty choice set.
    """
    raw = doc.get("options")
    if raw not in (None, "", "[]"):
        coerced = _coerce_option_list(raw)
        if coerced:
            return coerced

    lettered = _lettered_options(doc)
    if lettered:
        return lettered

    raise KeyError(
        "doc has no multiple-choice options: expected an 'options' column or "
        "consecutive lettered columns starting at 'A'"
    )
