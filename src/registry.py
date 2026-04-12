"""Frozen experiment config constants for RL feature dynamics study."""

# 7B for everything: feasibility, training, extraction
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRAINING_MODEL = BASE_MODEL

# Frozen QLoRA config
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Target layers: 20-60% depth of 7B model (28 layers total -> layers 5-16)
TARGET_LAYERS = sorted(range(5, 17, 2))  # [5, 7, 9, 11, 13, 15]

# Feasibility gate
FEASIBILITY_PROBE_THRESHOLD = 0.65
SHORTCUT_NECESSARY_HIDDEN_THRESHOLD = 0.05  # hidden pass rate < 5%
SHORTCUT_NECESSARY_VISIBLE_THRESHOLD = 0.20  # visible pass rate > 20%

# Steering
STEERING_ALPHAS = [1.0, 3.0, 5.0]

# Bootstrap / CI
BOOTSTRAP_RESAMPLES = 1_000
