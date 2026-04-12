"""Smoke tests for the reproduction analysis scripts.

These ensure the committed raw/ artifacts are enough to regenerate the
headline numbers from a fresh clone. Run with: pytest tests/test_reproduction.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "paper" / "raw"
CI_OUT = REPO / "results" / "paper" / "ci_and_permutation.json"


def test_raw_artifacts_present():
    """Every raw artifact the reproduction scripts depend on must exist."""
    required = [
        "steering_cc_trajectory.json",
        "alpha_r4_a2.json",
        "alpha_r4_a7.json",
        "alpha_r4_a10.json",
        "fresh_base.json",
        "fresh_r0.json",
        "fresh_r2.json",
        "fresh_r4.json",
        "probes_cc_results.json",
        "fresh_cosine.json",
        "problems_sha256.txt",
    ]
    missing = [f for f in required if not (RAW / f).exists()]
    assert not missing, f"missing raw artifacts: {missing}"


def test_problems_hash_matches_writeup():
    """The committed sha256 must match the hash cited in WRITEUP.md §3.1."""
    sha = (RAW / "problems_sha256.txt").read_text().strip().split()[0]
    assert sha == "f1186c1c4719fc09ecff9be27006de80d293590f370c42aff585b655404071f4", (
        f"problems.jsonl sha256 drifted: got {sha}"
    )


def test_bootstrap_cis_runs_and_has_headline_numbers():
    """bootstrap_cis.py writes ci_and_permutation.json with expected keys."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "bootstrap_cis.py")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"script failed:\n{result.stderr}"
    assert CI_OUT.exists()
    data = json.loads(CI_OUT.read_text())

    # Every load-bearing section must be present
    for key in (
        "frozen_alpha5_trajectory",
        "frozen_alpha_sweep_round4",
        "frozen_alpha_sweep_round2",
        "round_4_alpha10_significance",
        "round_4_alpha5_reproducibility",
        "u_shape_test",
        "fresh_vs_frozen_ab",
        "probe_trajectory",
        "strategy_trajectory",
        "strategy_stability_test",
    ):
        assert key in data, f"missing key {key} in ci_and_permutation.json"


def test_round_4_alpha10_significant():
    """The frozen direction at round 4 α=10 must remain significantly negative."""
    data = json.loads(CI_OUT.read_text())
    sig = data["round_4_alpha10_significance"]
    assert sig["abs_delta"] < -0.05, (
        f"round 4 α=10 absolute Δ unexpectedly weak: {sig['abs_delta']}"
    )
    assert sig["p_negative"] > 0.99, (
        f"round 4 α=10 no longer reliably negative: p={sig['p_negative']}"
    )


def test_round_2_alpha10_is_largest_effect():
    """Round 2 α=10 should produce the largest single-condition effect."""
    data = json.loads(CI_OUT.read_text())
    r2 = data["frozen_alpha_sweep_round2"]
    alpha10 = next(r for r in r2 if r["alpha"] == 10)
    assert alpha10["delta"]["rel_point"] < -0.20, (
        f"round 2 α=10 relative Δ unexpectedly weak: {alpha10['delta']['rel_point']}"
    )


def test_round_4_alpha5_dead_zone_was_noise():
    """The re-measurement of round 4 α=5 must disagree with the main trajectory
    by roughly the full binomial CI — this is the worked example in §4.5.3."""
    data = json.loads(CI_OUT.read_text())
    repro = data["round_4_alpha5_reproducibility"]
    # The two runs should differ by at least 5pp absolute
    main = repro["main_trajectory_abs_delta"]
    rerun = repro["rerun_abs_delta"]
    diff = abs(main - rerun)
    assert diff > 0.04, (
        f"expected at least 4pp disagreement between main and rerun "
        f"(the worked example of single-point noise); got {diff*100:.2f}pp"
    )


def test_frozen_auroc_drop_is_present():
    """Frozen AUROC trajectory should show the 0.98 → 0.89 drop."""
    data = json.loads(CI_OUT.read_text())
    frozen = data["probe_trajectory"]["frozen_aurocs"]
    assert frozen[0] > 0.97, f"base AUROC unexpectedly low: {frozen[0]}"
    assert frozen[-1] < 0.91, f"round_4 AUROC unexpectedly high: {frozen[-1]}"
    assert frozen[0] - frozen[-1] > 0.06, "drop smaller than expected"


def test_cosine_rotation_is_sustained():
    """Cosine should drop to ~0.37 by step_3 and stay there through step_5."""
    data = json.loads(CI_OUT.read_text())
    cosines = data["cosine_trajectory"]["cosines"]
    assert cosines["step_3"] < 0.5, f"cos(step_3) unexpectedly high: {cosines['step_3']}"
    assert cosines["step_4"] < 0.5, f"cos(step_4) unexpectedly high: {cosines['step_4']}"
    assert cosines["step_5"] < 0.5, f"cos(step_5) unexpectedly high: {cosines['step_5']}"


def test_strategy_stability():
    """Base vs round_4 iterative fraction should not be significantly different."""
    data = json.loads(CI_OUT.read_text())
    test = data["strategy_stability_test"]
    assert test["p_two_sided"] > 0.1, (
        f"strategy drift became significant: p={test['p_two_sided']}"
    )


def test_reproduce_script_runs():
    """reproduce_key_numbers.py must run cleanly after bootstrap_cis.py."""
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "reproduce_key_numbers.py")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"script failed:\n{result.stderr}"
    # Must print the load-bearing claims
    assert "Frozen AUROC degrades" in result.stdout
    assert "round 4" in result.stdout  # post-reframe wording
    assert "round 2" in result.stdout
    assert "monotonic dose-response" in result.stdout
    assert "reproducibility" in result.stdout.lower()
