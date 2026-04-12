"""Fetch the small raw JSON artifacts from the Modal volume into the repo.

This is a one-time setup script for readers who want to reproduce the
bootstrap CI analysis without touching Modal's raw activations / adapters.

Required: modal CLI auth with access to the `control-results` volume.

Writes all files under results/paper/raw/.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "results" / "paper" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# (remote_path, local_name)
FILES = [
    ("/steering_cc/results.json",              "steering_cc_trajectory.json"),
    ("/steering_cc_alpha_r4_a2/results.json",  "alpha_r4_a2.json"),
    ("/steering_cc_alpha_r4_a7/results.json",  "alpha_r4_a7.json"),
    ("/steering_cc_alpha_r4_a10/results.json", "alpha_r4_a10.json"),
    ("/steering_cc_fresh_base/results.json",   "fresh_base.json"),
    ("/steering_cc_fresh_r0/results.json",     "fresh_r0.json"),
    ("/steering_cc_fresh_r2/results.json",     "fresh_r2.json"),
    ("/steering_cc_fresh_r4/results.json",     "fresh_r4.json"),
    ("/probes_cc/results.json",                "probes_cc_results.json"),
    ("/probes_cc/fresh_cosine.json",           "fresh_cosine.json"),
    ("/probes_cc/validation_fresh_step5.json", "validation_fresh_step5.json"),
    ("/data_cc/problems.sha256",               "problems_sha256.txt"),
    # Strategy classification
    ("/strategy_cc/step_0.json",               "strategy_step_0.json"),
    ("/strategy_cc/step_1.json",               "strategy_step_1.json"),
    ("/strategy_cc/step_2.json",               "strategy_step_2.json"),
    ("/strategy_cc/step_3.json",               "strategy_step_3.json"),
    ("/strategy_cc/step_4.json",               "strategy_step_4.json"),
    ("/strategy_cc/step_5.json",               "strategy_step_5.json"),
    # Additional artifacts (may or may not exist depending on run state)
    ("/steering_cc_rerun_r4_a5/results.json",       "rerun_r4_a5.json"),
    ("/steering_cc_alpha_r2_a2/results.json",       "alpha_r2_a2.json"),
    ("/steering_cc_alpha_r2_a5/results.json",       "alpha_r2_a5.json"),
    ("/steering_cc_alpha_r2_a7/results.json",       "alpha_r2_a7.json"),
    ("/steering_cc_alpha_r2_a10/results.json",      "alpha_r2_a10.json"),
]


def main() -> None:
    for remote, local in FILES:
        dest = RAW / local
        print(f"fetching {remote} → {dest}")
        try:
            subprocess.run(
                ["modal", "volume", "get", "control-results", remote,
                 str(dest), "--force"],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"  skipped ({e.stderr.decode().strip().splitlines()[-1][:80]})")
            continue
    print(f"\nWrote raw artifacts under {RAW}")


if __name__ == "__main__":
    main()
