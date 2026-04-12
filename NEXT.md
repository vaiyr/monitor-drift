# A Self-Recalibrating Monitor with Causal Validation

---

## 1. One-sentence pitch

The paper showed that frozen linear monitors fail silently — detection accuracy degrades, additive steering still "works" but via magnitude rather than direction specificity, and nothing in the standard evaluation protocol flags this. This follow-up builds and evaluates a monitor that knows when it's broken: it tracks detection drift, triggers automatic probe refits, and runs Arditi-style ablation checks to verify that the monitored direction is still causally necessary rather than merely sufficient via magnitude.

---

## 2. The problem

Three failure modes compound silently during non-adversarial training:

### 2.1 Detection drift (silent)

The frozen probe's AUROC drops from 0.980 to ~0.89 within one SFT round. Fresh probes at the same checkpoint recover AUROC >= 0.92, so the feature is still there — the probe is just pointed at the wrong angle. A deployed monitor reports "model looks fine" when it should report "I can no longer see clearly."

**Key numbers:** cos(base, round_k) = 0.35 under deterministic LR for every k >= 1. The rotation is ~70 degrees in a single training step and stays put.

### 2.2 Misleading intervention confidence (silent)

Additive steering with the frozen direction produces monotonic dose-response curves at every checkpoint (round 4 alpha=10: -17.1%, CI [-25.8%, -7.4%]). A naive evaluation says "steering works, the direction is reliable." But this result is driven by magnitude, not direction specificity.

### 2.3 Loss of causal necessity (silent, only detectable with ablation)

Projecting out the frozen direction at round 4 produces no behavioral change (CV-LR: delta = -0.9%, p = 0.57; deterministic LR: +2.4%, p = 0.31). Only difference-of-means retains partial ablation effect (-7.2%, p = 0.92). The frozen direction is sufficient to push behavior around via large additive perturbation, but the model is no longer *using* that direction — it has routed around it.

**The compound failure:** A standard monitor checks (1) detection AUROC and (2) steering effect size. Both look acceptable. The actual state — the direction is stale and the steering is magnitude-only — is invisible without an ablation test. No existing monitoring protocol runs ablation checks during deployment.

---

## 3. The monitor

### 3.1 Design: three-signal monitoring

The monitor emits a diagnostic triple at each evaluation point:

```
(detection_confidence, intervention_confidence, causal_necessity)
```

| Signal | What it measures | How it's computed | Observed baseline trajectory |
|---|---|---|---|
| **detection_confidence** | Can the probe still classify? | Frozen probe AUROC on a held-out labeled buffer | 0.98 -> 0.89 in round 0, stable |
| **intervention_confidence** | Does steering still change behavior? | Additive steering delta at current best alpha | -24.5% at round 2, -17.1% at round 4 |
| **causal_necessity** | Is the direction actually load-bearing? | Ablation delta: project out the direction, measure behavioral change | ~0% at round 4 (fails) |

A naive frozen monitor tracks only the first signal. A standard detection+steering evaluation tracks the first two. This monitor tracks all three, and the third is what catches the failure mode the experiment identified.

### 3.2 Components

**Component 1: Drift detector.**

Inputs: frozen probe direction, fresh activations from the current checkpoint, held-out labeled buffer (shortcut/general examples).

At each evaluation point:
1. Compute frozen probe AUROC on the held-out buffer.
2. Train a fresh probe on the current activations.
3. Compute cos(frozen, fresh).
4. If AUROC < threshold_detection (default 0.92) OR cosine < threshold_cosine (default 0.70), emit a DRIFT alert and trigger a refit.

The thresholds are calibrated from the experimental data: AUROC 0.92 is the lower bound of fresh-probe performance, and cosine 0.70 is below the difference-of-means trajectory (0.66-0.82) but above the deterministic LR rotation (0.35). The point isn't to pick the "right" threshold — it's to have any threshold at all, which the frozen-monitor protocol does not.

**Component 2: Automatic probe refit.**

On DRIFT alert:
1. Retrain probe on current-checkpoint activations using the held-out buffer.
2. Run leave-one-problem-out validation (GroupKFold can mask problem-identity leakage in low-group settings; LOO catches it).
3. If LOO mean AUROC >= 0.85, hot-swap the detection direction. Otherwise, flag REFIT_FAILED — the feature may no longer be linearly separable at this layer.
4. Log the old and new directions, their cosine, and the LOO scores.

**Component 3: Causal necessity check.**

After every refit (and periodically on a slower cadence even without refit):
1. Run additive steering at alpha in {5, 10} with the current direction. Record delta.
2. Run projective ablation (Arditi et al. 2024 formulation) with the current direction. Record delta.
3. Compare:
   - **Both positive:** Direction is causally necessary and sufficient. Healthy.
   - **Additive positive, ablation null:** Direction is sufficient via magnitude but not causally necessary. The model has routed around it. Emit NECESSITY_LOST alert.
   - **Both null:** Direction has lost all leverage. Emit LEVERAGE_LOST alert.
   - **Ablation positive, additive null:** Unusual — direction is necessary but additive steering doesn't exploit it. Flag for investigation.

The NECESSITY_LOST state is the specific failure mode the experiment discovered. No existing monitoring protocol detects it.

### 3.3 Output format

Each evaluation point produces a structured record:

```json
{
  "checkpoint": "round_4",
  "timestamp": "2026-04-12T14:30:00Z",
  "detection": {
    "frozen_auroc": 0.892,
    "fresh_auroc": 0.987,
    "cosine_frozen_fresh": 0.35,
    "drift_alert": true
  },
  "refit": {
    "triggered": true,
    "loo_mean_auroc": 0.924,
    "loo_min_auroc": 0.328,
    "loo_min_problem": "cc_11_B._Jumping_Jack",
    "new_direction_saved": true
  },
  "causal": {
    "additive_delta_alpha5": -0.098,
    "additive_delta_alpha10": -0.171,
    "ablation_delta": -0.009,
    "ablation_p_negative": 0.565,
    "status": "NECESSITY_LOST"
  }
}
```

### 3.4 What makes this different from "just retrain your probe"

Retraining the probe addresses detection drift (component 1-2). That's table stakes. The contribution is component 3: the causal necessity check operationalizes the additive/ablation dissociation as a continuous monitoring signal. The specific claim is:

> A monitor that passes standard detection and intervention checks can still be in a NECESSITY_LOST state where the model has routed around the monitored direction. The only way to detect this is to run a projective ablation test alongside the additive steering test. This monitor does that.

This is not a theoretical concern — it is exactly what happened at round 4 in our experiments.

---

## 4. Evaluation against existing checkpoints

The monitor can be evaluated retroactively against the six existing checkpoints (base, rounds 0-4) without any new training runs. This is the core demonstration.

### 4.1 Retroactive evaluation protocol

For each of the 6 checkpoints, run the full monitor pipeline and record the diagnostic triple. The evaluation answers: **at which checkpoint would the monitor have fired each alert, and would those alerts have been correct?**

Expected results from existing data:

| Checkpoint | detection_confidence | intervention_confidence | causal_necessity | Expected alerts |
|---|---|---|---|---|
| base | 0.98 (high) | N/A (no steering baseline) | N/A | None |
| round_0 | ~0.89 (degraded) | strong (alpha=10 effective) | unknown (not yet tested) | DRIFT |
| round_1 | ~0.89 | strong | unknown | DRIFT |
| round_2 | ~0.89 | -24.5% at alpha=10 | needs testing | DRIFT, possibly healthy causal |
| round_3 | ~0.89 | declining | needs testing | DRIFT |
| round_4 | ~0.89 | -17.1% at alpha=10 | FAILS (delta = -0.9%) | DRIFT + NECESSITY_LOST |

The key comparison: a naive frozen monitor at round 4 reports "AUROC 0.89 (slightly degraded), steering effect -17.1% (working)." This monitor reports "AUROC 0.89 (DRIFT), steering effect -17.1% (working), ablation effect -0.9% (NECESSITY_LOST)." The third signal is what catches the failure.

### 4.2 Fresh-direction causal check

After the refit at each drifted checkpoint, run the causal necessity check on the *fresh* direction. If the fresh direction passes both additive and ablation tests where the frozen direction failed ablation, that demonstrates the full monitoring loop: detect drift -> refit -> verify causal necessity on the new direction -> confirm recovery.

### 4.3 Comparison baselines

1. **Frozen-only monitor:** Never refits, never checks ablation. Reports AUROC only. This is the standard deployed protocol.
2. **Refit-only monitor:** Detects drift and refits, but doesn't check causal necessity. This is "just retrain your probe."
3. **Full monitor:** Detects drift, refits, and checks causal necessity. This is the proposed system.

The comparison shows what each additional component catches that the simpler versions miss.

### 4.4 New measurements needed

The retroactive evaluation requires ablation tests at checkpoints the initial experiment didn't cover:
- Round 0, round 1, round 2, round 3 ablation with the frozen direction (we only ran round 4).
- Round 0-4 ablation with the *fresh* direction at each checkpoint (to verify refit recovery).

This is approximately 8 additional ablation runs at n=960, each taking ~1h of A100 time. Total new compute: ~8-10h. No new training runs needed.

---

## 5. Implementation plan

### 5.1 Module structure

```
src/
  monitor.py              Core monitor: three-signal evaluation loop
  ablation.py             Projective ablation (factor out from existing code)
  # Existing modules used as-is:
  probe.py                Probe training, LOO validation, cosine tracking
  extract.py              Activation extraction
  steer.py                Additive steering

scripts/
  run_monitor_retroactive.py    Evaluate monitor against existing checkpoints
  plot_monitor_signals.py       Three-panel figure: detection / intervention / necessity
```

### 5.2 `src/monitor.py` — core module

```python
# Sketch — not pseudocode, this is the actual interface

@dataclass
class MonitorSignals:
    """Diagnostic triple emitted at each evaluation point."""
    checkpoint: str
    # Detection
    frozen_auroc: float
    fresh_auroc: float
    cosine_frozen_fresh: float
    drift_alert: bool
    # Refit
    refit_triggered: bool
    loo_mean_auroc: float | None
    loo_min_auroc: float | None
    loo_min_problem: str | None
    # Causal
    additive_deltas: dict[float, float]   # alpha -> delta
    ablation_delta: float | None
    ablation_p_negative: float | None
    causal_status: str  # "HEALTHY", "NECESSITY_LOST", "LEVERAGE_LOST"


class RecalibratingMonitor:
    def __init__(
        self,
        frozen_direction: np.ndarray,
        monitor_layer: int = 11,
        detection_threshold: float = 0.92,
        cosine_threshold: float = 0.70,
        ablation_n: int = 960,
        steering_alphas: tuple[float, ...] = (5.0, 10.0),
    ):
        self.current_direction = frozen_direction
        self.frozen_direction = frozen_direction  # keep original for comparison
        ...

    def evaluate(
        self,
        model,
        checkpoint_name: str,
        labeled_buffer: list[dict],   # held-out shortcut/general examples
        eval_problems: list[dict],    # problems for steering/ablation
    ) -> MonitorSignals:
        """Run the full three-signal evaluation."""
        # 1. Detection
        activations = extract_activations(model, labeled_buffer, self.monitor_layer)
        frozen_auroc = probe_auroc(activations, labels, self.frozen_direction)
        fresh_direction, fresh_auroc = train_fresh_probe(activations, labels)
        cosine = cos_sim(self.frozen_direction, fresh_direction)
        drift_alert = frozen_auroc < self.detection_threshold or cosine < self.cosine_threshold

        # 2. Refit (if drifted)
        refit_result = None
        if drift_alert:
            refit_result = self._refit(activations, labels, labeled_buffer)
            if refit_result.passed:
                self.current_direction = refit_result.new_direction

        # 3. Causal necessity check
        causal_result = self._check_causal(model, eval_problems)

        return MonitorSignals(...)

    def _refit(self, activations, labels, buffer):
        """Retrain probe with LOO validation."""
        ...

    def _check_causal(self, model, problems):
        """Additive steering + projective ablation comparison."""
        ...
```

### 5.3 `src/ablation.py` — projective ablation

Factor out the existing ablation code into a reusable module:

```python
def ablation_hook(direction: np.ndarray):
    """Forward hook that projects out `direction` from residual stream."""
    d_unit = direction / np.linalg.norm(direction)
    def hook(module, input, output):
        # output: (batch, seq, dim)
        proj = (output @ d_unit) * d_unit  # projection onto direction
        return output - proj              # remove it
    return hook

def run_ablation(
    model, problems, direction, layer, n_samples=960
) -> tuple[float, float]:
    """Returns (delta_rel, p_negative) from ablation experiment."""
    ...
```

### 5.4 `scripts/run_monitor_retroactive.py` — demonstration script

The headline demonstration: run the monitor against all six checkpoints and show it would have caught the NECESSITY_LOST state at round 4.

```python
def main():
    monitor = RecalibratingMonitor(
        frozen_direction=load_frozen_direction(),
        monitor_layer=11,
    )

    for checkpoint in ["base", "round_0", "round_1", "round_2", "round_3", "round_4"]:
        model = load_checkpoint(checkpoint)
        signals = monitor.evaluate(model, checkpoint, labeled_buffer, eval_problems)
        save_signals(signals, checkpoint)
        print(f"{checkpoint}: detection={signals.frozen_auroc:.3f}  "
              f"intervention={signals.additive_deltas.get(10.0, 'N/A')}  "
              f"causal={signals.causal_status}")
```

Expected output:

```
base:    detection=0.980  intervention=N/A       causal=HEALTHY
round_0: detection=0.889  intervention=-0.XXX    causal=???         [DRIFT -> REFIT]
round_1: detection=0.891  intervention=-0.XXX    causal=???         [DRIFT -> REFIT]
round_2: detection=0.887  intervention=-0.245    causal=???         [DRIFT -> REFIT]
round_3: detection=0.890  intervention=-0.XXX    causal=???         [DRIFT -> REFIT]
round_4: detection=0.892  intervention=-0.171    causal=NECESSITY_LOST  [DRIFT -> REFIT -> ALERT]
```

The `???` entries at rounds 0-3 are the measurements this project produces. If the fresh-direction ablation test passes at those checkpoints, the story is: "the monitor detects drift, refits, confirms causal necessity at early checkpoints, and correctly flags the loss of causal necessity at round 4."

### 5.5 `scripts/plot_monitor_signals.py` — the paper figure

Three-panel figure showing all three signals across checkpoints:

- **Panel A:** Frozen vs. fresh AUROC (detection). Shows the gap that triggers refit.
- **Panel B:** Additive steering delta at alpha=10 (intervention). Shows the misleadingly healthy signal.
- **Panel C:** Ablation delta (causal necessity). Shows the signal that only the full monitor catches — the drop to zero at round 4.

Overlay: mark where each monitor variant (frozen-only, refit-only, full) would have fired alerts. The full monitor fires NECESSITY_LOST at round 4; the others don't.

## 6. Risks and open questions

### 6.1 Ablation is expensive at scale

Projective ablation requires a forward pass per direction at each evaluation point. At 7B parameters this is manageable; at frontier scale it may be prohibitive for continuous monitoring. Mitigation: the ablation check runs on a slower cadence than detection (e.g., every 10 checkpoints), and only on the current active direction. The detection + refit components are cheap and run every step.

### 6.2 Ablation results at rounds 0-3 might not show a clean transition

The expected story is: causal necessity holds at early checkpoints and fails at round 4, with the monitor catching the transition. If ablation fails at round 0 (i.e., the direction is never causally necessary after the first SFT step), the story is still clean but harsher — the frozen direction was magnitude-only from the start. If ablation passes at all checkpoints, the monitor never fires NECESSITY_LOST, and the demonstration is weaker. Either outcome is reportable but the clean transition is the best figure.

### 6.3 The held-out labeled buffer requires labeled data

The detection component needs shortcut/general labels for the held-out buffer. In the CodeContests setting this is free (hidden tests provide labels). In a real deployment, the monitor needs some source of labeled examples — either human annotation, a separate classifier, or a trusted evaluation set. This is a limitation of the approach, not specific to this monitor.

### 6.4 Threshold sensitivity

The detection and cosine thresholds (0.92, 0.70) are calibrated from this experiment's data. Different features, models, or training regimes may need different thresholds. The monitor should ship with a calibration routine that sets thresholds from a baseline evaluation window, not hardcoded values.

### 6.5 Fresh-direction ablation might also fail

If the fresh probe's direction also fails the ablation test (i.e., no linear direction is causally necessary at late checkpoints), the refit doesn't help — the feature has genuinely become non-linearly encoded or distributed. The monitor would correctly report this as a failure state, but there's no automatic recovery. This would be an important negative result: linear monitoring has a fundamental ceiling at this checkpoint.
