"""Download results from Modal Volume to local results/ directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def download(artifacts: str = "all"):
    """Download artifacts from Modal volume.

    artifacts: 'activations', 'analysis', 'data', 'all'
    """
    local_dir = Path("results")
    local_dir.mkdir(exist_ok=True)

    if artifacts in ("activations", "all"):
        print("[download] downloading activations...")
        subprocess.run([
            "modal", "volume", "get", "control-results",
            "activations/", str(local_dir / "activations/"),
        ], check=True)

    if artifacts in ("analysis", "all"):
        print("[download] downloading analysis...")
        subprocess.run([
            "modal", "volume", "get", "control-results",
            "analysis/", str(local_dir / "analysis/"),
        ], check=True)

    if artifacts in ("data", "all"):
        print("[download] downloading frozen data...")
        subprocess.run([
            "modal", "volume", "get", "control-results",
            "data/", str(local_dir / "data/"),
        ], check=True)

    print(f"[download] done. Files in {local_dir}/")


def main():
    artifacts = sys.argv[1] if len(sys.argv) > 1 else "all"
    download(artifacts)


if __name__ == "__main__":
    main()
