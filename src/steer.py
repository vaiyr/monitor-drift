"""Steering interventions: activation patching at probe direction."""

from __future__ import annotations

import numpy as np
import torch

from concurrent.futures import ThreadPoolExecutor, as_completed

from src.registry import STEERING_ALPHAS


def _eval_parallel(codes: list[str], problems: list, eval_fn, max_workers: int = 32) -> list:
    """Evaluate code submissions in parallel. Returns list of EvalResult."""
    results = [None] * len(codes)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(eval_fn, code, prob): i
            for i, (code, prob) in enumerate(zip(codes, problems))
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception:
                results[idx] = None
    return results


def _strip_code_fences(code: str) -> str:
    """Strip markdown code fences from model output."""
    code = code.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    elif code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return code


def add_steering_hook(model, layer_idx: int, direction: np.ndarray, alpha: float):
    """Register a forward hook that adds alpha * direction to residual stream at layer.

    Returns the hook handle (call handle.remove() to deactivate).
    """
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16).to(model.device)

    def hook_fn(module, input, output):
        # Subtract direction to push AWAY from shortcuts (probe positive class)
        if isinstance(output, tuple):
            hidden = output[0]
            hidden = hidden - alpha * direction_tensor.unsqueeze(0).unsqueeze(0)
            return (hidden,) + output[1:]
        else:
            return output - alpha * direction_tensor.unsqueeze(0).unsqueeze(0)

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle


def add_ablation_hook(model, layer_idx: int, direction: np.ndarray):
    """Register a forward hook that projects `direction` out of residuals.

    Following Arditi et al. 2024: h' = h - (h · d̂) d̂, applied at every token
    position at the target layer. The direction is unit-normalized inside the
    hook, so the caller does not need to normalize it first.

    Returns the hook handle (call handle.remove() to deactivate).
    """
    d = np.asarray(direction, dtype=np.float32)
    norm = float(np.linalg.norm(d))
    if norm < 1e-8:
        raise ValueError("ablation direction has near-zero norm")
    d_unit = torch.tensor(d / norm, dtype=torch.bfloat16).to(model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # projection: (hidden @ d_unit) * d_unit, broadcasted over (batch, seq)
        proj = (hidden * d_unit).sum(dim=-1, keepdim=True) * d_unit
        new_hidden = hidden - proj
        if isinstance(output, tuple):
            return (new_hidden,) + output[1:]
        return new_hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    return handle


def random_direction(
    hidden_dim: int,
    seed: int,
    target_norm: float,
) -> np.ndarray:
    """Generate a reproducible Gaussian random direction with a target L2 norm."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(hidden_dim).astype(np.float32)
    current_norm = float(np.linalg.norm(v))
    return v * (target_norm / max(current_norm, 1e-12))


def generate_with_ablation(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    direction: np.ndarray,
    n_samples: int = 16,
    max_new_tokens: int = 1024,
    micro_batch_size: int = 64,
) -> list[str]:
    """Generate n_samples completions per prompt with directional ablation active.

    Uses an ablation hook at `layer_idx` that projects `direction` out of the
    residual stream (Arditi et al. 2024 formulation). Same generation loop as
    `generate_with_steering`, just with a different hook.
    """
    handle = add_ablation_hook(model, layer_idx, direction)
    device = next(model.parameters()).device
    try:
        completions = []
        for pi, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]

            remaining = n_samples
            prompt_completions = []
            while remaining > 0:
                this_batch = min(remaining, micro_batch_size)
                batch_ids = input_ids.expand(this_batch, -1)
                batch_mask = attention_mask.expand(this_batch, -1)
                with torch.no_grad():
                    output_ids = model.generate(
                        batch_ids,
                        attention_mask=batch_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for i in range(this_batch):
                    new_tokens = output_ids[i, prompt_len:]
                    text = _strip_code_fences(
                        tokenizer.decode(new_tokens, skip_special_tokens=True)
                    )
                    prompt_completions.append(text)
                remaining -= this_batch

            completions.extend(prompt_completions)
            print(
                f"  [gen-ablate] {pi+1}/{len(prompts)} (n={n_samples})",
                flush=True,
            )
        return completions
    finally:
        handle.remove()


def measure_ablation_effectiveness(
    model,
    tokenizer,
    prompts: list[str],
    problems: list,
    eval_fn,
    layer_idx: int,
    direction: np.ndarray,
    n_samples: int = 16,
    micro_batch_size: int = 64,
) -> dict:
    """Measure shortcut rate with directional ablation vs. no intervention.

    Returns a dict shaped like `measure_steering_effectiveness` but with
    'shortcut_rate_ablated' in place of 'shortcut_rate_steered'.
    """
    expanded_problems = [prob for prob in problems for _ in range(n_samples)]

    unsteered_codes = generate_without_steering(
        model, tokenizer, prompts,
        n_samples=n_samples, micro_batch_size=micro_batch_size,
    )
    ablated_codes = generate_with_ablation(
        model, tokenizer, prompts, layer_idx, direction,
        n_samples=n_samples, micro_batch_size=micro_batch_size,
    )

    print(
        f"[ablate-measure] evaluating {len(unsteered_codes)} unsteered + "
        f"{len(ablated_codes)} ablated...",
        flush=True,
    )
    unsteered_evals = _eval_parallel(unsteered_codes, expanded_problems, eval_fn)
    ablated_evals = _eval_parallel(ablated_codes, expanded_problems, eval_fn)

    unsteered_shortcuts = sum(
        1 for er in unsteered_evals if er is not None and er.is_shortcut
    )
    ablated_shortcuts = sum(
        1 for er in ablated_evals if er is not None and er.is_shortcut
    )

    n = len(unsteered_codes)
    return {
        "shortcut_rate_unsteered": unsteered_shortcuts / n if n else 0,
        "shortcut_rate_ablated": ablated_shortcuts / n if n else 0,
        "n_unsteered_shortcuts": unsteered_shortcuts,
        "n_ablated_shortcuts": ablated_shortcuts,
        "n_total": n,
        "n_problems": len(problems),
        "direction_norm": float(np.linalg.norm(direction)),
    }


def generate_with_steering(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
    n_samples: int = 16,
    max_new_tokens: int = 1024,
    micro_batch_size: int = 64,
) -> list[str]:
    """Generate n_samples completions per prompt with steering active.

    Splits the n_samples completions per prompt into micro-batches of size
    `micro_batch_size` so we can crank `n_samples` high (e.g. 128-256 for
    n≥1000 per condition) without blowing KV cache.
    """
    handle = add_steering_hook(model, layer_idx, direction, alpha)
    device = next(model.parameters()).device
    try:
        completions = []
        for pi, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            prompt_len = input_ids.shape[1]

            remaining = n_samples
            prompt_completions = []
            while remaining > 0:
                this_batch = min(remaining, micro_batch_size)
                batch_ids = input_ids.expand(this_batch, -1)
                batch_mask = attention_mask.expand(this_batch, -1)
                with torch.no_grad():
                    output_ids = model.generate(
                        batch_ids,
                        attention_mask=batch_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=0.8,
                        top_p=0.95,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for i in range(this_batch):
                    new_tokens = output_ids[i, prompt_len:]
                    text = _strip_code_fences(
                        tokenizer.decode(new_tokens, skip_special_tokens=True)
                    )
                    prompt_completions.append(text)
                remaining -= this_batch

            completions.extend(prompt_completions)
            print(
                f"  [gen-steered] {pi+1}/{len(prompts)} "
                f"(alpha={alpha:.1f}, n={n_samples})",
                flush=True,
            )
        return completions
    finally:
        handle.remove()


def generate_without_steering(
    model,
    tokenizer,
    prompts: list[str],
    n_samples: int = 16,
    max_new_tokens: int = 1024,
    micro_batch_size: int = 64,
) -> list[str]:
    """Generate n_samples completions per prompt without steering.

    Splits into micro-batches to support large n_samples.
    """
    device = next(model.parameters()).device
    completions = []
    for pi, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        remaining = n_samples
        while remaining > 0:
            this_batch = min(remaining, micro_batch_size)
            batch_ids = input_ids.expand(this_batch, -1)
            batch_mask = attention_mask.expand(this_batch, -1)
            with torch.no_grad():
                output_ids = model.generate(
                    batch_ids,
                    attention_mask=batch_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for i in range(this_batch):
                new_tokens = output_ids[i, prompt_len:]
                text = _strip_code_fences(
                    tokenizer.decode(new_tokens, skip_special_tokens=True)
                )
                completions.append(text)
            remaining -= this_batch
        print(f"  [gen-baseline] {pi+1}/{len(prompts)} (n={n_samples})", flush=True)
    return completions


def sweep_steering_alpha(
    model,
    tokenizer,
    prompts: list[str],
    problems: list,
    eval_fn,
    layer_idx: int,
    direction: np.ndarray,
    alphas: list[float] | None = None,
    n_samples: int = 16,
    micro_batch_size: int = 64,
) -> tuple[float, dict]:
    """Sweep alpha values to find one that meaningfully reduces shortcut rate.

    Generates n_samples per problem for statistical power.
    Returns (best_alpha, results_dict).
    """
    if alphas is None:
        alphas = STEERING_ALPHAS

    # Expand problems to match n_samples per prompt
    expanded_problems = [prob for prob in problems for _ in range(n_samples)]

    # Baseline shortcut rate
    baseline_codes = generate_without_steering(
        model, tokenizer, prompts,
        n_samples=n_samples, micro_batch_size=micro_batch_size,
    )
    print(f"[steer] evaluating {len(baseline_codes)} baseline completions...", flush=True)
    baseline_evals = _eval_parallel(baseline_codes, expanded_problems, eval_fn)
    baseline_shortcuts = sum(1 for er in baseline_evals if er is not None and er.is_shortcut)
    n_total = len(baseline_codes)
    baseline_rate = baseline_shortcuts / n_total if n_total else 0
    print(f"[steer] baseline: {baseline_shortcuts}/{n_total} shortcuts ({baseline_rate:.3f})", flush=True)

    results = {"baseline_shortcut_rate": baseline_rate, "n_total": n_total, "alphas": {}}
    best_alpha = alphas[0]
    best_reduction = 0.0

    for alpha in alphas:
        steered_codes = generate_with_steering(
            model, tokenizer, prompts, layer_idx, direction, alpha,
            n_samples=n_samples, micro_batch_size=micro_batch_size,
        )
        print(f"[steer] evaluating alpha={alpha:.1f}...", flush=True)
        steered_evals = _eval_parallel(steered_codes, expanded_problems, eval_fn)
        steered_shortcuts = sum(1 for er in steered_evals if er is not None and er.is_shortcut)
        steered_rate = steered_shortcuts / n_total if n_total else 0
        reduction = baseline_rate - steered_rate
        results["alphas"][alpha] = {
            "shortcut_rate": steered_rate,
            "reduction": reduction,
            "n_shortcuts": steered_shortcuts,
        }
        if reduction > best_reduction:
            best_reduction = reduction
            best_alpha = alpha
        print(f"[steer] alpha={alpha:.1f}: shortcut_rate={steered_rate:.3f} (reduction={reduction:.3f})", flush=True)

    return best_alpha, results


def measure_steering_effectiveness(
    model,
    tokenizer,
    prompts: list[str],
    problems: list,
    eval_fn,
    layer_idx: int,
    direction: np.ndarray,
    alpha: float,
    n_samples: int = 16,
    micro_batch_size: int = 64,
) -> dict:
    """Measure steering effectiveness at a single checkpoint with fixed alpha.

    Generates n_samples per problem for statistical power.
    Returns dict with shortcut rates steered vs unsteered.
    """
    expanded_problems = [prob for prob in problems for _ in range(n_samples)]

    unsteered_codes = generate_without_steering(
        model, tokenizer, prompts,
        n_samples=n_samples, micro_batch_size=micro_batch_size,
    )
    steered_codes = generate_with_steering(
        model, tokenizer, prompts, layer_idx, direction, alpha,
        n_samples=n_samples, micro_batch_size=micro_batch_size,
    )

    print(f"[steer-measure] evaluating {len(unsteered_codes)} unsteered + {len(steered_codes)} steered...", flush=True)
    unsteered_evals = _eval_parallel(unsteered_codes, expanded_problems, eval_fn)
    steered_evals = _eval_parallel(steered_codes, expanded_problems, eval_fn)

    unsteered_shortcuts = sum(1 for er in unsteered_evals if er is not None and er.is_shortcut)
    steered_shortcuts = sum(1 for er in steered_evals if er is not None and er.is_shortcut)

    n = len(unsteered_codes)
    return {
        "shortcut_rate_unsteered": unsteered_shortcuts / n if n else 0,
        "shortcut_rate_steered": steered_shortcuts / n if n else 0,
        "n_unsteered_shortcuts": unsteered_shortcuts,
        "n_steered_shortcuts": steered_shortcuts,
        "n_total": n,
        "n_problems": len(problems),
        "alpha": alpha,
    }
