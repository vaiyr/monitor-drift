"""CodeContests problem loading (deepmind/code_contests).

Loads CodeContests problems as CodingProblem objects with visible/hidden test splits. CodeContests has
a much richer test structure (30-100+ hidden tests per problem), which is the
reason we're validating the pipeline on it — a shortcut that passes visible
tests but fails ~100 hidden tests has to generalize badly across many concrete
inputs, which forces the probe features to be less problem-idiosyncratic.

Schema (deepmind/code_contests):
  name:            str
  description:     str (problem statement)
  public_tests:    {"input": list[str], "output": list[str]}
  private_tests:   {"input": list[str], "output": list[str]}
  generated_tests: {"input": list[str], "output": list[str]}
  source:          int (2=CODEFORCES, 3=DESCRIPTION_ONLY, 4=CODECHEF, 5=CODEJAM, 6=ATCODER, 7=AIZU, ...)
  difficulty:      int
  cf_rating:       int (Codeforces rating; 800-3500 range)
  cf_tags:         list[str]
  is_description_translated: bool

We filter to:
  - problems with ≥3 public tests and ≥10 hidden tests (public+private+generated)
  - non-translated English descriptions
  - cf_rating in a medium range (default: 800-1800)
  - descriptions with no obvious floating-point / special-checker signal
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.types import CodingProblem, TestCase


# Heuristic: skip problems whose output is likely to require custom-checker
# semantics (floating-point tolerance, multiple valid outputs, etc.). These
# aren't robustly parseable and would contaminate the shortcut/general labels.
_BAD_OUTPUT_SIGNALS = re.compile(
    r"\b(relative error|absolute error|up to \d+ decimal|precision|epsilon|"
    r"any of the valid|any valid answer|multiple (correct|valid) (answers|solutions)|"
    r"any such|print any|output any|you may print any)\b",
    re.IGNORECASE,
)
# Also skip interactive problems — they don't fit the stdin/stdout eval loop.
_INTERACTIVE_SIGNAL = re.compile(
    r"\binteractive (problem|task|judge)\b|this is an interactive problem",
    re.IGNORECASE,
)


def _collect_tests(record: dict, key: str) -> list[TestCase]:
    block = record.get(key) or {}
    inputs = block.get("input") or []
    outputs = block.get("output") or []
    n = min(len(inputs), len(outputs))
    return [TestCase(input=inputs[i], expected_output=outputs[i]) for i in range(n)]


def load_code_contests(
    splits: tuple[str, ...] = ("train", "valid", "test"),
    min_public_tests: int = 3,
    min_hidden_tests: int = 10,
    max_hidden_tests: int = 60,
    cf_rating_range: tuple[int, int] = (800, 1800),
    require_english: bool = True,
    skip_bad_checkers: bool = True,
) -> list[CodingProblem]:
    """Load CodeContests as CodingProblem records with visible/hidden splits.

    The visible split is `public_tests`; the hidden split is `private_tests +
    generated_tests`, capped at `max_hidden_tests` to bound per-problem eval
    cost.
    """
    from datasets import load_dataset

    problems: list[CodingProblem] = []
    seen_names: set[str] = set()

    for split in splits:
        print(f"[codecontests] loading split={split}")
        ds = load_dataset("deepmind/code_contests", split=split)
        for row in ds:
            name = row.get("name") or ""
            if not name or name in seen_names:
                continue

            description = row.get("description") or ""
            if not description or len(description) < 50:
                continue

            if require_english and row.get("is_description_translated"):
                continue

            if skip_bad_checkers and (
                _BAD_OUTPUT_SIGNALS.search(description)
                or _INTERACTIVE_SIGNAL.search(description)
            ):
                continue

            cf_rating = row.get("cf_rating") or 0
            if cf_rating_range is not None:
                lo, hi = cf_rating_range
                if cf_rating < lo or cf_rating > hi:
                    continue

            public_tests = _collect_tests(row, "public_tests")
            private_tests = _collect_tests(row, "private_tests")
            generated_tests = _collect_tests(row, "generated_tests")

            if len(public_tests) < min_public_tests:
                continue

            hidden_tests = private_tests + generated_tests
            if len(hidden_tests) < min_hidden_tests:
                continue
            hidden_tests = hidden_tests[:max_hidden_tests]

            # Skip problems whose first expected output contains a float —
            # second-pass defense against lenient checkers. Integer/string
            # outputs are the safe target for strict exact-match eval.
            first_out = hidden_tests[0].expected_output.strip()
            if first_out and any(
                "." in line and re.search(r"\d\.\d", line)
                for line in first_out.splitlines()
            ):
                continue

            problems.append(CodingProblem(
                id=f"cc_{name}".replace(" ", "_")[:120],
                prompt=description,
                starter_code="",
                visible_tests=public_tests,
                hidden_tests=hidden_tests,
                difficulty=f"cf_rating_{cf_rating}",
            ))
            seen_names.add(name)

        print(f"[codecontests] {len(problems)} problems accumulated after split={split}")

    return problems


def freeze_code_contests_set(
    problems: list[CodingProblem],
    n_problems: int = 1000,
    output_dir: str = "/results/data_cc",
) -> Path:
    """Freeze a CodeContests subset as a JSONL file."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = problems[:n_problems]
    jsonl_path = out / "problems.jsonl"
    with jsonl_path.open("w") as f:
        for p in selected:
            f.write(json.dumps(p.to_dict()) + "\n")

    sha = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    (out / "problems.sha256").write_text(sha + "\n")
    print(f"[codecontests] froze {len(selected)} problems → {jsonl_path} "
          f"(sha256: {sha[:16]}...)")
    return jsonl_path


def load_frozen_code_contests(frozen_dir: str | Path = "/results/data_cc") -> list[CodingProblem]:
    """Load a frozen CodeContests problem set from JSONL."""
    p = Path(frozen_dir) / "problems.jsonl"
    problems: list[CodingProblem] = []
    with p.open() as f:
        for line in f:
            if line.strip():
                problems.append(CodingProblem.from_dict(json.loads(line)))
    return problems
