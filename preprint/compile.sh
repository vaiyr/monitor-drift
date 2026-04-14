#!/bin/bash
# One-shot build: generate figures and compile LaTeX.
set -euo pipefail
cd "$(dirname "$0")"

echo "--- Generating figures ---"
cd figure_scripts
python3 figure1_hero.py
python3 figure2_multi_method.py
python3 figure3_dose_response.py
python3 figure4_ablation.py
python3 figure5_ablation_trajectory.py
cd ..

echo "--- Compiling LaTeX ---"
tectonic main.tex

echo "--- Done: main.pdf ---"
ls -lh main.pdf
