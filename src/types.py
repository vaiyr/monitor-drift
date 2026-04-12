"""Types for the RL feature dynamics experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

SolutionLabel = Literal["shortcut", "general", "failing"]
ProblemCategory = Literal["shortcut_necessary", "shortcut_optional", "unclear"]


@dataclass
class TestCase:
    input: str
    expected_output: str


@dataclass
class CodingProblem:
    id: str
    prompt: str
    starter_code: str
    visible_tests: list[TestCase]
    hidden_tests: list[TestCase]
    difficulty: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "starter_code": self.starter_code,
            "visible_tests": [{"input": t.input, "expected_output": t.expected_output} for t in self.visible_tests],
            "hidden_tests": [{"input": t.input, "expected_output": t.expected_output} for t in self.hidden_tests],
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CodingProblem:
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            starter_code=d["starter_code"],
            visible_tests=[TestCase(**t) for t in d["visible_tests"]],
            hidden_tests=[TestCase(**t) for t in d["hidden_tests"]],
            difficulty=d["difficulty"],
        )


@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


@dataclass
class EvalResult:
    visible_passed: int
    visible_total: int
    hidden_passed: int
    hidden_total: int

    @property
    def all_visible_passed(self) -> bool:
        return self.visible_passed == self.visible_total

    @property
    def all_hidden_passed(self) -> bool:
        return self.hidden_passed == self.hidden_total

    @property
    def is_shortcut(self) -> bool:
        return self.all_visible_passed and not self.all_hidden_passed

    @property
    def is_general(self) -> bool:
        return self.all_visible_passed and self.all_hidden_passed


class Generation(BaseModel):
    """A single code generation for a problem."""
    problem_id: str
    code: str
    label: SolutionLabel
    visible_passed: int
    visible_total: int
    hidden_passed: int
    hidden_total: int
    checkpoint_step: int = 0  # 0 = base model


class ProblemStats(BaseModel):
    """Pass rate statistics for a single problem."""
    problem_id: str
    visible_pass_rate: float
    hidden_pass_rate: float
    n_samples: int
    category: ProblemCategory = "unclear"


class FeasibilityResult(BaseModel):
    """Result of the feasibility gate."""
    probe_auroc: float
    best_layer: int
    confidence_alignment: float
    n_shortcut_necessary: int
    n_shortcut_optional: int
    gate_passed: bool
