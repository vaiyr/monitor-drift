# Results (CodeContests)

**Date:** 2026-04-10
**Status:** Full pipeline complete on CodeContests. Results support a detection/intervention dissociation interpretation.

---

## TL;DR

Under optimization pressure (shortcut rate ~65%) on a dataset with stable problem-identity controls (CodeContests, 242 shortcut-producing problems), we find that **the passive readability and the active dose-response curve of a shortcut-awareness feature drift independently during training**.

- **Direction drift:** cosine between frozen and fresh probe directions drops 1.00 → 0.37 by round_2; frozen AUROC drops 0.98 → 0.89; fresh AUROC rises 0.98 → 1.00
- **Frozen direction retains additive leverage:** the frozen direction produces monotonic dose-response curves at every checkpoint. At round_4: −17.1% at α=10, the largest single-condition effect measured anywhere in the experiment
- **But fails the ablation necessity test at round_4:** projecting the frozen direction out of the residual stream produces no behavioral change (Δ = −0.9%, p = 0.57). The direction is sufficient via magnitude but not causally necessary — the model has routed around it
- **Strategy distribution is stable** (±4pp random walk, net +1.6pp iterative base → round_4) — direction drift cannot be explained by the model shifting to different shortcut strategies

**The main finding is a compound silent failure.** A frozen monitor reports acceptable detection and intervention metrics while the monitored direction is no longer the causal axis the model uses. Standard evaluation misses this; only a projective ablation test catches it.

---

## Methodology note: GroupKFold in low-group regimes

GroupKFold cross-validation can mask significant problem-identity leakage when the number of groups is small. We recommend leave-one-group-out validation as a standard check for linear probes on grouped data whenever group counts fall below ~30. `scripts/validate_probe.py` implements both GroupKFold and LOO against the same data.

---

## Full results

### Phase 1: Feasibility (CodeContests, 1000 problems × 16 samples)

| Metric | Value |
|---|---|
| Total generations | 16,000 |
| Shortcuts | 721 |
| Distinct shortcut problems | 242 |
| Per-generation shortcut rate | 4.5% |
| Probe in-sample AUROC (layer 11) | 0.932 |
| Probe GroupKFold CV AUROC | 0.718 ± 0.118 |
| Confidence alignment | 0.027 |

### Phase 2: Iterative SFT training (5 rounds)

Conservative hyperparameters (rank-16 LoRA, lr=2e-5, 1 epoch, pool across rounds). Shortcut rate on visible-passing solutions stayed at 61–66% throughout:

| Round | Visible-passing | Shortcuts | Shortcut rate |
|---|---|---|---|
| 0 | 443 | 276 | 62.3% |
| 1 | 456 | 279 | 61.2% |
| 2 | 436 | 282 | 64.7% |
| 3 | 465 | 303 | 65.2% |
| 4 | 469 | 309 | 65.9% |

### Phase 3a/b: Probe tracking across 6 checkpoints

| Checkpoint | Frozen AUROC | Fresh AUROC | Cos(frozen, fresh) |
|---|---|---|---|
| step_0 (base) | 0.980 | 0.980 | 1.000 |
| step_1 (round_0) | 0.926 | 0.973 | 0.778 |
| step_2 (round_1) | 0.893 | 0.924 | 0.718 |
| step_3 (round_2) | 0.895 | **1.000** | **0.367** |
| step_4 (round_3) | 0.902 | 1.000 | 0.380 |
| step_5 (round_4) | 0.892 | 1.000 | 0.399 |

**Observations:**
- Frozen AUROC drops ~9pp — the original probe direction becomes less aligned with where the feature now lives
- Fresh AUROC rises to perfect — the feature is becoming *more* separable, just in a different direction
- Cosine similarity drops sharply at round_2 (0.72 → 0.37) — a phase transition in the representation
- Fresh probe direction norm also explodes at step_3 (0.46 → 32.4), confirming the sharp representational change

**LOO validation of the fresh probe at step_5** (across 60 CC problems): mean 0.955, median 1.0, min 0.328, std 0.124. The fresh 1.0 AUROC is not identity-confounded — the probe genuinely generalizes across problems. The 0.328 minimum is on a single problem (`cc_11_B._Jumping_Jack`) with only 8 shortcut samples, and that problem is structurally anomalous (shortcut is closed-form math with a bug; general is iterative brute force — inverted from the modal shortcut pattern).

### Phase 3c: Steering (frozen direction, α=5.0, 30 problems × 32 samples, n=960 per condition)

Alpha sweep at base model: 1.0, 3.0 showed no effect; **α=5.0 best at −1.4pp**. The main checkpoint trajectory was run with α=5.0:

| Checkpoint | Unsteered | Steered | Δ abs | Δ rel |
|---|---|---|---|---|
| base | 0.427 | 0.414 | −0.013 | −3.0% |
| round_0 | 0.459 | 0.420 | −0.039 | −8.5% |
| round_1 | 0.443 | 0.418 | −0.025 | −5.6% |
| round_2 | **0.474** | 0.418 | **−0.056** | **−11.9%** (peak) |
| round_3 | 0.443 | 0.406 | −0.036 | −8.2% |
| round_4 | 0.428 | 0.431 | **+0.003** | **+0.7%** (collapse) |

The round_4 α=5 collapse is a sample-size artifact — a re-measurement returned −9.8%, consistent with the monotonic dose-response at that checkpoint (see α-sweep below).

### Focused validation: fresh-probe steering + α sweep at round_4

**Fresh-probe steering (norm-matched to frozen, α=5.0) at 4 checkpoints:**

| Checkpoint | Cos(frozen, fresh) | Frozen Δ | Fresh Δ |
|---|---|---|---|
| base | 1.00 | −3.0% | −3.1% (sanity check passes) |
| round_0 | 0.78 | −8.5% | **−10.5%** (fresh slightly stronger) |
| round_2 | 0.37 | **−11.9%** | −9.0% (frozen stronger) |
| round_4 | 0.40 | +0.7% | **−7.2%** (fresh works where frozen appears dead) |

**Frozen α-sweep at round_4:**

| α | Baseline | Steered | Δ abs | Δ rel |
|---|---|---|---|---|
| 2.0 | 0.448 | 0.423 | −0.025 | −5.6% |
| 5.0 | 0.428 | 0.431 | +0.003 | +0.7% (noisy — rerun gives −9.8%) |
| 7.0 | 0.438 | 0.384 | −0.053 | **−12.1%** |
| 10.0 | 0.440 | 0.365 | −0.075 | **−17.1%** |

**Two key findings from these checks:**

1. **The frozen direction at round_4 retains additive leverage.** At α=10, it produces a −17.1% reduction, the largest single-condition effect measured anywhere in the experiment. The dose-response is monotonic.

2. **Fresh-probe steering and frozen-probe steering have different dose calibrations at each checkpoint.** At round_4, fresh works at α=5 (−7.2%) while frozen appears dead at α=5 (+0.7%). At round_2, frozen is stronger at α=5 (−11.9% vs −9.0%). At round_0, fresh is slightly stronger (−10.5% vs −8.5%).

### Directional ablation (Arditi et al. 2024 necessity test)

Projecting the frozen direction out of the residual stream at round_4 across three extraction methods:

| Extraction method | Ablation Δ rel | p(Δ<0) |
|---|---|---|
| CV-LR (default) | −0.9% | 0.565 |
| Deterministic LR (C=1.0) | +2.4% | 0.311 |
| Difference-of-means | −7.2% | 0.915 |

LR-family frozen directions fail the necessity test — removing them from the residual stream does not change behavior. Only difference-of-means retains a partial ablation effect. The frozen direction is sufficient to influence behavior via additive magnitude but is not the causal axis the model uses at round_4.

### Read/write dissociation at round_2

At round_2 (the peak steering checkpoint), we steered with the *fresh* probe direction trained on step_3 activations, after rescaling to match the frozen direction's L2 norm (both = 1.51):

| Direction at round_2 | Unsteered | Steered | Δ abs | Δ rel |
|---|---|---|---|---|
| **Frozen** (trained at step_0) | 0.474 | 0.418 | −0.056 | **−11.9%** |
| **Fresh** (trained at step_3, norm-matched) | 0.465 | 0.423 | −0.042 | **−9.0%** |

**The direction that reads the feature best (fresh, 1.0 AUROC) has *weaker* causal effect than the direction that reads it worse (frozen, 0.89 AUROC).** Same model, same problems, same α, norm-matched perturbation. Read-alignment and write-alignment are different axes.

### Strategy distribution (control for strategy drift)

| Checkpoint | Iterative | Non-iterative | Total shortcuts |
|---|---|---|---|
| step_0 (base) | 68.8% | 31.2% | 535 |
| step_1 (round_0) | 66.0% | 34.0% | 574 |
| step_2 (round_1) | 69.9% | 30.1% | 559 |
| step_3 (round_2) | 67.6% | 32.4% | 589 |
| step_4 (round_3) | 67.2% | 32.8% | 585 |
| step_5 (round_4) | 70.4% | 29.6% | 585 |

**Range 66–70.4%, random walk, no trend.** The probe dynamics and steering trajectory cannot be explained by strategy drift — the model is producing the same shortcut strategy mix throughout training.

---

## Interpretation

Under optimization pressure on a confound-controlled dataset, the shortcut representation undergoes rotation (cosine frozen↔fresh drops 1.0 → 0.37) while the feature becomes more sharply separable (fresh AUROC 0.98 → 1.00). The original probe direction becomes partially misaligned with the new representation, so its passive AUROC drops.

The frozen direction retains additive causal leverage at every checkpoint (monotonic dose-response, −17.1% at round_4 α=10). But this leverage is magnitude-driven, not direction-specific: projecting the frozen direction out of the residual stream at round_4 produces no behavioral change. The model has routed around the frozen direction while still being pushable by large additive perturbations along it.

### Strategy drift is ruled out

±2pp in iterative fraction across 5 rounds cannot account for a 9pp frozen AUROC drop, a 0.63 cosine rotation, or a 12-percentage-point swing in steering effect. The model is doing the same things, but representing and generating them through different internal pathways.

### Monitoring implications

1. **Passive probes silently lose direction calibration.** A frozen probe's AUROC degrades under training pressure even when the underlying concept is still being produced and is still causally intervenable. A deployed monitor using a fixed probe would emit false negatives at increasing rates.

2. **Additive steering can be misleading.** A frozen steering vector's effect at a given α can appear healthy while the direction has lost causal necessity. Additive steering success does not license necessity claims about the steered direction — only a projective ablation test discriminates between direction-specific and magnitude-driven effects.

3. **Best-read direction ≠ best-write direction ≠ constant across training.** At round_2, frozen direction is a better steering vector than fresh. At round_4, fresh direction is better at small α while frozen needs higher α. These relationships are not predictable from passive AUROC.

4. **Robust monitoring requires joint calibration.** Both the direction and the causal status must be periodically re-checked:
   - Refit the probe (catch direction rotation; restore passive detection)
   - Run a projective ablation test (verify the direction is still causally necessary, not just magnitude-sufficient)
   - Run both passive and active tests (neither alone captures the full failure mode)

### What this is NOT

- This is **not** a decoupling result. The frozen direction at round_4 produces a −17.1% steering effect at α=10. The feature does not disconnect from behavior.
- This is **not** a suppression result. The fresh probe still hits 1.0 — the feature is not being destroyed.
- This is **not** a strategy drift result. Iterative fraction is essentially flat (±4pp random walk, net +1.6pp).
- The Jumping Jack finding (fresh probe fails on problems with inverted strategy structure) is a narrower observation about what specific probes learn in distribution, not about the main result.

---

## Known limitations

1. **n=960 per steering condition gives ±3pp CI.** Individual checkpoint effects are borderline. The trend is clear but specific point estimates should not be over-interpreted.
2. **Single α (5.0) for the main checkpoint sweep.** The full α sweep was only run at round_4. The growing-then-collapsing trajectory at α=5.0 could look different at other α values.
3. **α-sweeps only at two checkpoints.** Both show monotonic dose-response, but full characterization across all checkpoints would take ~24h additional compute.
4. **Ablation only at round_4.** Whether the ablation necessity failure appears at earlier checkpoints is unknown and is a key measurement for the monitor implementation (see `NEXT.md`).
5. **CodeContests may have contamination with Qwen2.5-Coder training data.** This affects absolute shortcut rates but is less of a concern for representation dynamics.

---

## Key files and artifacts (Modal volume `control-results`)

### Data
- `/data_cc/problems.jsonl` — 1000 frozen CodeContests problems (sha256: f1186c1c4719fc09...)
- `/data_cc/training_problem_ids.json` — 150 problems used for training

### Feasibility
- `/feasibility_cc/generations.jsonl` — 16,000 generations
- `/feasibility_cc/result.json` — probe AUROC, layer
- `/probes/validation_cc.json` — LOO validation of feasibility probe

### Training
- `/checkpoints/ppo_cc/round_{0..4}/` — LoRA adapters
- `/checkpoints/ppo_cc/training_log.json` — per-round counts

### Probing
- `/activations_cc/step_{0..5}/` — layer-11 activations at each checkpoint
- `/probes_cc/frozen_probe_direction.npy` — step-0 probe direction (frozen)
- `/probes_cc/results.json` — frozen+fresh AUROC trajectories
- `/probes_cc/fresh_cosine.json` — cosine similarity frozen↔fresh per step
- `/probes_cc/fresh_directions/step_{0..5}.npy` — fresh probe directions per step
- `/probes_cc/fresh_directions/step_3_scaled.npy` — norm-matched to frozen for read/write check
- `/probes_cc/validation_fresh_step5.json` — LOO validation of fresh probe at step_5

### Steering
- `/steering_cc/results.json` — main steering trajectory (frozen direction, 6 checkpoints)
- `/steering_cc_fresh_r2/results.json` — fresh-direction steering at round_2

### Strategy classification
- `/strategy_cc/step_{0..5}.json` — iterative/closed-form/other counts per checkpoint
- `/strategy_cc/step_{0..5}_shortcuts.jsonl` — sample of raw shortcut codes (up to 400 per step)

---

## Code pointers

- `src/codecontests_env.py` — CC loader with cf_rating, checker, and hidden-test filters
- `src/probe.py` — linear probing, directions, cosine
- `src/extract.py` — activation extraction with LoRA merging
- `src/steer.py` — HF model.generate() + forward hook steering with micro-batching
- `src/train_ppo.py` — iterative rejection sampling + SFT with resume support
- `scripts/validate_probe.py` — LOO and drop-dominant validation for any feasibility probe
- `deploy/app.py` — Modal entry points for every phase

All vLLM workloads use single A100-80GB (`tensor_parallel_size=1`). Steering has resume logic (loads existing results.json at start, skips completed checkpoints) to survive container preemption.

## Running the full pipeline from scratch

```bash
modal run deploy/app.py --mode freeze-cc --n-problems 1000
modal run deploy/app.py --mode feasibility-cc --n-problems 1000
modal run deploy/app.py --mode validate-feasibility-probe-cc
modal run deploy/app.py --mode build-cc-train-ids
modal run deploy/app.py --mode train-ppo-cc --max-steps 5
modal run deploy/app.py --mode extract-checkpoints-cc --max-steps 5
modal run deploy/app.py --mode probe-checkpoints-cc
modal run deploy/app.py --mode fresh-directions-cc
modal run deploy/app.py --mode classify-strategies-cc
modal run deploy/app.py --mode steer-cc-checkpoints
# For the read/write dissociation check:
modal run deploy/app.py --mode steer-cc-fresh-r2
```
