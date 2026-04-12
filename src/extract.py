"""Activation extraction with checkpoint iteration support."""

from __future__ import annotations

from itertools import islice
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.registry import BASE_MODEL, TARGET_LAYERS


def batched(iterable, n):
    """Batch an iterable into chunks of size n."""
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def load_model_for_extraction(
    base_model: str = BASE_MODEL,
    adapter_path: str | None = None,
):
    """Load model for activation extraction. Merges LoRA if adapter_path given."""
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path is not None:
        from peft import AutoPeftModelForCausalLM
        print(f"[extract] loading and merging adapter from {adapter_path}")
        peft_model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path, device_map="auto", torch_dtype=torch.bfloat16,
        )
        model = peft_model.merge_and_unload()
    else:
        print(f"[extract] loading base model {base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16,
        )

    model.eval()
    return model, tokenizer


def extract_activations(
    model,
    tokenizer,
    texts: list[str],
    target_layers: list[int] | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
) -> dict[int, np.ndarray]:
    """Extract mean-pooled residual stream activations at target layers.

    Args:
        texts: list of text strings to extract activations from
        target_layers: which layers to extract from
        batch_size: batch size for forward passes
        max_length: max token length

    Returns:
        dict mapping layer_idx -> (n_samples, hidden_dim) numpy array
    """
    if target_layers is None:
        target_layers = TARGET_LAYERS

    all_acts = {layer: [] for layer in target_layers}

    for batch in batched(texts, batch_size):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        input_ids = inputs["input_ids"]
        if input_ids.numel() == 0:
            continue

        try:
            # With device_map="auto", tensors need to go to first parameter's device
            device = next(model.parameters()).device
            with torch.no_grad():
                outputs = model(
                    input_ids.to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states
            attention_mask = inputs["attention_mask"].to(hidden_states[0].device)
            for layer_idx in target_layers:
                hs_idx = layer_idx + 1
                if hs_idx < len(hidden_states):
                    h = hidden_states[hs_idx]  # (batch, seq, hidden)
                    # Masked mean pooling
                    mask = attention_mask.unsqueeze(-1).float()
                    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    all_acts[layer_idx].append(pooled.detach().cpu().float().numpy())
        except Exception as e:
            print(f"[extract] skipping batch: {e}")

    result = {}
    for layer_idx in target_layers:
        if all_acts[layer_idx]:
            result[layer_idx] = np.concatenate(all_acts[layer_idx], axis=0)
    return result


def extract_with_log_probs(
    model,
    tokenizer,
    texts: list[str],
    target_layers: list[int] | None = None,
    batch_size: int = 4,
    max_length: int = 2048,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Extract activations and mean log-probabilities for confidence estimation.

    Returns:
        (activations_dict, log_probs) where log_probs is (n_samples,)
    """
    if target_layers is None:
        target_layers = TARGET_LAYERS

    all_acts = {layer: [] for layer in target_layers}
    all_log_probs = []

    for batch in batched(texts, batch_size):
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        if input_ids.numel() == 0:
            continue

        try:
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            # Activations
            hidden_states = outputs.hidden_states
            mask = attention_mask.unsqueeze(-1).float()
            for layer_idx in target_layers:
                hs_idx = layer_idx + 1
                if hs_idx < len(hidden_states):
                    h = hidden_states[hs_idx]
                    pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                    all_acts[layer_idx].append(pooled.detach().cpu().float().numpy())

            # Log probabilities: shift logits and compute per-token log prob
            logits = outputs.logits  # (batch, seq, vocab)
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:].float()

            log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
            # Mean log prob per sequence (masked)
            mean_lp = (token_log_probs * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
            all_log_probs.append(mean_lp.detach().cpu().float().numpy())
        except Exception as e:
            print(f"[extract] skipping batch: {e}")

    result = {}
    for layer_idx in target_layers:
        if all_acts[layer_idx]:
            result[layer_idx] = np.concatenate(all_acts[layer_idx], axis=0)

    log_probs_arr = np.concatenate(all_log_probs, axis=0) if all_log_probs else np.array([])
    return result, log_probs_arr


def save_activations(acts: dict[int, np.ndarray], output_path: str):
    """Save activations as .npz file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"layer_{k}": v for k, v in acts.items()}
    np.savez_compressed(output_path, **arrays)
    print(f"[extract] saved activations to {output_path} ({len(arrays)} layers)")


def load_activations(npz_path: str) -> dict[int, np.ndarray]:
    """Load activations from .npz file."""
    data = np.load(npz_path)
    result = {}
    for key in data.files:
        layer_idx = int(key.replace("layer_", ""))
        result[layer_idx] = data[key]
    return result
