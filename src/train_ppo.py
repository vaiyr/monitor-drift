"""Iterative rejection sampling + SFT for visible-test-pass reward.

Each round:
1. Generate N completions per problem using vLLM (fast)
2. Evaluate against visible tests → reward
3. Filter to reward=1 completions (pass all visible tests)
4. SFT on filtered completions with QLoRA
5. Checkpoint frequently within SFT for probe tracking

This gives the same optimization pressure as PPO/RLOO (push toward
visible-test-passing solutions) but decouples generation from training.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from src.registry import (
    LORA_TARGET_MODULES,
    TRAINING_MODEL,
)
from src.types import ExecutionResult


def execute_code(code: str, test_input: str, timeout_sec: float = 5.0) -> ExecutionResult:
    """Run code in subprocess with resource limits."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return ExecutionResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(stdout="", stderr="timeout", exit_code=-1, timed_out=True)
    except Exception as e:
        return ExecutionResult(stdout="", stderr=str(e), exit_code=-1, timed_out=False)
    finally:
        os.unlink(tmp_path)


def build_prompt(problem: dict, tokenizer) -> str:
    """Build a generation prompt for a coding problem."""
    messages = [
        {"role": "system", "content": "Solve the programming problem. Output only Python code, no explanation."},
        {"role": "user", "content": f"Problem:\n{problem['prompt']}\n\nTest cases:\n" + "\n".join(
            f"  Input: {tc['input'].strip()}\n  Expected: {tc['expected_output'].strip()}"
            for tc in problem["visible_tests"]
        )},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_sft_example(problem: dict, code: str, tokenizer) -> str:
    """Format (prompt, code) pair for SFT training."""
    messages = [
        {"role": "system", "content": "Solve the programming problem. Output only Python code, no explanation."},
        {"role": "user", "content": f"Problem:\n{problem['prompt']}\n\nTest cases:\n" + "\n".join(
            f"  Input: {tc['input'].strip()}\n  Expected: {tc['expected_output'].strip()}"
            for tc in problem["visible_tests"]
        )},
        {"role": "assistant", "content": code},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)


class FrequentCheckpointCallback(TrainerCallback):
    """Save adapter every N steps for granular probe tracking."""

    def __init__(self, save_dir: str, save_every: int = 50):
        self.save_dir = save_dir
        self.save_every = save_every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        if state.global_step % self.save_every == 0 and state.global_step > 0:
            ckpt_dir = os.path.join(self.save_dir, f"step_{state.global_step}")
            model.save_pretrained(ckpt_dir)
            print(f"[sft] checkpoint at step {state.global_step}")


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


def _fast_eval(code: str, problem: dict) -> dict:
    """Fast-fail evaluation: bail on first visible test failure."""
    from src.types import TestCase

    vis_tests = [TestCase(**t) if isinstance(t, dict) else t for t in problem.get("visible_tests", [])]
    hid_tests = [TestCase(**t) if isinstance(t, dict) else t for t in problem.get("hidden_tests", [])]

    for tc in vis_tests:
        inp = tc.input if hasattr(tc, 'input') else tc['input']
        exp = tc.expected_output if hasattr(tc, 'expected_output') else tc['expected_output']
        result = execute_code(code, inp, timeout_sec=3.0)
        if result.exit_code != 0 or result.stdout.strip() != exp.strip():
            return {"visible_passed": False, "hidden_passed": False}

    # All visible passed — check hidden
    hid_pass = True
    for tc in hid_tests:
        inp = tc.input if hasattr(tc, 'input') else tc['input']
        exp = tc.expected_output if hasattr(tc, 'expected_output') else tc['expected_output']
        result = execute_code(code, inp, timeout_sec=3.0)
        if result.exit_code != 0 or result.stdout.strip() != exp.strip():
            hid_pass = False
            break  # fast-fail hidden too

    return {"visible_passed": True, "hidden_passed": hid_pass}


def generate_and_filter(
    problems: list[dict],
    eval_fn,
    llm,
    tokenizer,
    n_per_problem: int = 8,
    lora_request=None,
) -> list[dict]:
    """Generate completions and filter to those that pass visible tests.

    Uses fast-fail evaluation with parallel test execution.
    Returns list of {"problem": dict, "code": str, "is_shortcut": bool}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=1024, temperature=0.8, top_p=0.95)
    filtered = []

    for i, problem in enumerate(problems):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "Solve the programming problem. Output only Python code, no explanation."},
            {"role": "user", "content": f"Problem:\n{problem['prompt']}\n\nTest cases:\n" + "\n".join(
                f"  Input: {tc['input'].strip()}\n  Expected: {tc['expected_output'].strip()}"
                for tc in problem["visible_tests"]
            )},
        ], tokenize=False, add_generation_prompt=True)

        gen_kwargs = {"prompts": [prompt] * n_per_problem, "sampling_params": params}
        if lora_request:
            gen_kwargs["lora_request"] = lora_request

        try:
            outputs = llm.generate(**gen_kwargs)
        except Exception:
            continue

        codes = [_strip_code_fences(out.outputs[0].text) for out in outputs]

        # Parallel fast-fail evaluation
        results = [None] * len(codes)
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(_fast_eval, code, problem): j for j, code in enumerate(codes)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = {"visible_passed": False, "hidden_passed": False}

        for code, er in zip(codes, results):
            if er is None:
                continue
            if er["visible_passed"]:
                filtered.append({
                    "problem": problem,
                    "code": code,
                    "is_shortcut": not er["hidden_passed"],
                })

        if (i + 1) % 20 == 0:
            n_shortcuts = sum(1 for f in filtered if f["is_shortcut"])
            print(f"[gen] {i+1}/{len(problems)}: {len(filtered)} pass visible, {n_shortcuts} shortcuts")

    return filtered


def sft_on_filtered(
    filtered: list[dict],
    all_prior_filtered: list[dict],
    base_model: str,
    tokenizer,
    output_dir: str,
    round_num: int,
    adapter_path: str | None = None,
    num_epochs: int = 1,
    save_every: int = 50,
):
    """SFT train on filtered (reward=1) completions.

    Pools across all prior rounds to prevent catastrophic forgetting.
    Uses conservative hyperparameters: low LR, 1 epoch, rank 16.
    No quantization during training to match vLLM inference (bfloat16).
    """
    # Pool current + all prior filtered examples
    all_examples = all_prior_filtered + filtered
    train_data = []
    for item in all_examples:
        text = build_sft_example(item["problem"], item["code"], tokenizer)
        train_data.append({"text": text})

    if len(train_data) < 10:
        print(f"[sft] round {round_num}: only {len(train_data)} examples, skipping")
        return None

    train_dataset = Dataset.from_list(train_data)

    # Conservative LoRA: rank 16 to limit capacity relative to small datasets
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    round_dir = os.path.join(output_dir, f"round_{round_num}")
    os.makedirs(round_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=round_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,  # conservative LR to avoid capability degradation
        num_train_epochs=num_epochs,
        bf16=True,
        gradient_checkpointing=True,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        warmup_ratio=0.1,
    )

    # Always start from base model in bfloat16 (no quantization)
    # This matches vLLM inference, avoiding quantization mismatch
    model = AutoModelForCausalLM.from_pretrained(
        base_model, device_map="auto", torch_dtype=torch.bfloat16,
    )

    callbacks = [FrequentCheckpointCallback(round_dir, save_every=save_every)]

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model(round_dir)

    n_shortcuts = sum(1 for f in filtered if f["is_shortcut"])
    n_total = len(all_examples)
    print(f"[sft] round {round_num}: trained on {n_total} examples "
          f"(current round: {len(filtered)}, {n_shortcuts} shortcuts)")

    return round_dir


def train_iterative(
    problems: list[dict],
    eval_fn,
    base_model: str = TRAINING_MODEL,
    output_dir: str = "/results/checkpoints/ppo",
    n_rounds: int = 5,
    n_per_problem: int = 8,
    save_every: int = 50,
):
    """Run iterative rejection sampling + SFT.

    Each round generates completions, filters to reward=1, and SFTs.
    Checkpoints are saved frequently within each SFT round for probe tracking.
    """
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_log: list[dict] = []
    adapter_path = None
    all_prior_filtered: list[dict] = []  # pooled across all rounds

    # Resume support: load existing training_log.json and skip completed rounds.
    # If round_{k} has an adapter_model.safetensors and training_log.json has an
    # entry for round k, treat it as done and carry adapter_path forward.
    log_path = os.path.join(output_dir, "training_log.json")
    if os.path.exists(log_path):
        try:
            training_log = json.loads(Path(log_path).read_text())
        except Exception:
            training_log = []
    completed_rounds = set()
    for entry in training_log:
        r = entry.get("round")
        if r is None:
            continue
        rdir = os.path.join(output_dir, f"round_{r}")
        if os.path.exists(os.path.join(rdir, "adapter_model.safetensors")):
            completed_rounds.add(r)
    if completed_rounds:
        last_done = max(completed_rounds)
        adapter_path = os.path.join(output_dir, f"round_{last_done}")
        print(f"[train] resuming: rounds {sorted(completed_rounds)} already done, "
              f"using adapter from round {last_done}")

    for round_num in range(n_rounds):
        if round_num in completed_rounds:
            print(f"\n[train] skipping round {round_num} (already completed)")
            continue

        print(f"\n{'='*60}")
        print(f"  ROUND {round_num + 1}/{n_rounds}")
        print(f"{'='*60}")

        # Step 1: Generate with vLLM
        from vllm import LLM

        llm_kwargs = {
            "model": base_model,
            "dtype": "bfloat16",
            "max_model_len": 8192,  # CodeContests descriptions can exceed 2k tokens
            "trust_remote_code": True,
            "gpu_memory_utilization": 0.4,  # lowered to avoid sampler warmup OOM
            "max_num_seqs": 128,  # cap concurrent seqs to bound warmup memory
        }
        lora_req = None
        if adapter_path:
            from vllm.lora.request import LoRARequest
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 16
            lora_req = LoRARequest(f"round_{round_num}", 1, adapter_path)

        llm = LLM(**llm_kwargs)
        tok = llm.get_tokenizer()

        # Step 2: Generate and filter
        filtered = generate_and_filter(
            problems, eval_fn, llm, tok,
            n_per_problem=n_per_problem,
            lora_request=lora_req,
        )

        n_shortcuts = sum(1 for f in filtered if f["is_shortcut"])
        n_general = len(filtered) - n_shortcuts
        round_log = {
            "round": round_num,
            "n_filtered": len(filtered),
            "n_shortcuts": n_shortcuts,
            "n_general": n_general,
            "shortcut_rate": n_shortcuts / len(filtered) if filtered else 0,
            "pool_size": len(all_prior_filtered) + len(filtered),
        }
        training_log.append(round_log)
        print(f"[round {round_num}] {len(filtered)} pass visible "
              f"({n_shortcuts} shortcuts, {n_general} general), "
              f"pool size: {len(all_prior_filtered) + len(filtered)}")

        # Free vLLM GPU memory before SFT
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        if not filtered:
            print(f"[round {round_num}] no passing completions, skipping SFT")
            continue

        # Step 3: SFT on pooled filtered completions (current + all prior)
        adapter_path = sft_on_filtered(
            filtered, all_prior_filtered,
            base_model, tokenizer, output_dir,
            round_num=round_num,
            adapter_path=None,  # always start from base model
            save_every=save_every,
        )
        all_prior_filtered.extend(filtered)

        # Save round log
        log_path = os.path.join(output_dir, "training_log.json")
        Path(log_path).write_text(json.dumps(training_log, indent=2))

    print(f"\n[train] complete: {n_rounds} rounds")
    print(json.dumps(training_log, indent=2))
    return output_dir
