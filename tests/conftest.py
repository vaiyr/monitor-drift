"""Synthetic fixtures for testing."""

from __future__ import annotations

import numpy as np
import pytest

from src.types import CodingProblem, Generation, TestCase


@pytest.fixture
def sample_problem():
    return CodingProblem(
        id="test-001",
        prompt="Given two integers a and b, print their sum.",
        starter_code="",
        visible_tests=[
            TestCase(input="1 2\n", expected_output="3\n"),
            TestCase(input="5 3\n", expected_output="8\n"),
        ],
        hidden_tests=[
            TestCase(input="0 0\n", expected_output="0\n"),
            TestCase(input="-1 1\n", expected_output="0\n"),
        ],
        difficulty="introductory",
    )


@pytest.fixture
def correct_code():
    return "a, b = map(int, input().split())\nprint(a + b)"


@pytest.fixture
def shortcut_code():
    """Code that passes visible tests but fails on hidden test (0 0)."""
    return """a, b = map(int, input().split())
if a == 0 and b == 0:
    print(1)
else:
    print(a + b)"""


@pytest.fixture
def sample_activations():
    """Synthetic activation data for probe testing."""
    rng = np.random.default_rng(42)
    n_per_class = 100
    hidden_dim = 64

    X_shortcut = rng.normal(loc=1.0, scale=1.0, size=(n_per_class, hidden_dim))
    X_general = rng.normal(loc=-1.0, scale=1.0, size=(n_per_class, hidden_dim))

    X = np.vstack([X_shortcut, X_general])
    y = np.array([1] * n_per_class + [0] * n_per_class)

    return X, y


@pytest.fixture
def sample_activations_with_groups():
    """Synthetic activations with problem group labels."""
    rng = np.random.default_rng(42)
    n_per_class = 100
    hidden_dim = 64

    X_shortcut = rng.normal(loc=1.0, scale=1.0, size=(n_per_class, hidden_dim))
    X_general = rng.normal(loc=-1.0, scale=1.0, size=(n_per_class, hidden_dim))

    X = np.vstack([X_shortcut, X_general])
    y = np.array([1] * n_per_class + [0] * n_per_class)
    # 10 unique problem IDs
    groups = np.array([f"p{i % 10}" for i in range(n_per_class * 2)])

    return X, y, groups


@pytest.fixture
def sample_generations():
    """Synthetic generation data."""
    gens = []
    for i in range(100):
        gens.append(Generation(
            problem_id=f"p{i % 20}",
            code=f"print({i})",
            label="shortcut" if i % 3 == 0 else "general" if i % 3 == 1 else "failing",
            visible_passed=2 if i % 3 != 2 else 1,
            visible_total=2,
            hidden_passed=2 if i % 3 == 1 else 0,
            hidden_total=2,
        ))
    return gens


@pytest.fixture
def sample_checkpoint_activations():
    """Synthetic checkpoint activations for probe tracking."""
    rng = np.random.default_rng(42)
    hidden_dim = 64
    n_per_class = 50
    n_checkpoints = 5

    checkpoint_acts = []
    checkpoint_labels = []
    checkpoint_groups = []

    for ckpt in range(n_checkpoints):
        # Gradually reduce separation to simulate decoupling
        sep = 1.0 - ckpt * 0.15
        X_s = rng.normal(loc=sep, scale=1.0, size=(n_per_class, hidden_dim))
        X_g = rng.normal(loc=-sep, scale=1.0, size=(n_per_class, hidden_dim))
        X = np.vstack([X_s, X_g])
        y = np.array([1] * n_per_class + [0] * n_per_class)
        groups = np.array([f"p{i % 10}" for i in range(n_per_class * 2)])

        checkpoint_acts.append({9: X})  # layer 9
        checkpoint_labels.append(y)
        checkpoint_groups.append(groups)

    return checkpoint_acts, checkpoint_labels, checkpoint_groups
