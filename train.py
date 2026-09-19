from __future__ import annotations

import argparse
import math
import os
import time
from functools import partial
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import (
    EquationDataset,
    Generation,
    ExpressionProblem,
    Problem,
    collate_batch,
    decode_model_answer,
    generate_problems,
    generate_expression_problems,
    parse_prompt,
    reverse_digit_runs,
    split_think_and_answer,
    to_model_prompt,
    LengthBucketBatchSampler,
)
from model import MathGPT
from tokenizer import CharTokenizer


def build_tokenizer(cfg: Config) -> CharTokenizer:
    # Preserve the original checkpoint vocabulary exactly, then append new tokens.
    base_specials = [cfg.think_start, cfg.think_end]
    tokens = (
        list(cfg.chars)
        + [cfg.pad_token, cfg.bos_token, cfg.eos_token]
        + base_specials
        + list("abcdefghijklmnopqrstuvwxyz/")
        + [cfg.error_token]
    )
    return CharTokenizer(
        chars=cfg.chars,
        pad_token=cfg.pad_token,
        bos_token=cfg.bos_token,
        eos_token=cfg.eos_token,
        extra_specials=cfg.extra_specials,
        tokens=tokens,
    )


def get_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_cuda_speedups() -> None:
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def resolve_amp_dtype(cfg: Config, device: torch.device) -> torch.dtype | None:
    if not cfg.use_amp or device.type != "cuda":
        return None
    if cfg.amp_dtype == "bf16":
        return torch.bfloat16
    if cfg.amp_dtype == "fp16":
        return torch.float16
    raise ValueError(f"Unknown amp_dtype: {cfg.amp_dtype!r} (expected bf16 or fp16)")


def dataloader_kwargs(cfg: Config, device: torch.device) -> dict:
    kwargs: dict = {}
    if cfg.num_workers > 0:
        kwargs["num_workers"] = cfg.num_workers
        kwargs["persistent_workers"] = True
    if device.type == "cuda":
        kwargs["pin_memory"] = True
    return kwargs


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_compile_model(
    model: MathGPT,
    device: torch.device,
    enabled: bool,
    *,
    max_seq_len: int,
    amp_dtype: torch.dtype | None,
) -> MathGPT:
    if not enabled or device.type != "cuda":
        return model

    try:
        import triton  # noqa: F401
    except ImportError:
        print(
            "warning: triton not installed; torch.compile skipped "
            "(inductor backend needs Triton, which is unavailable on most Windows setups)"
        )
        return model

    try:
        compiled = torch.compile(model)
        probe_b, probe_t = 2, max_seq_len - 1
        x = torch.randint(0, 10, (probe_b, probe_t), device=device)
        y = torch.full((probe_b, probe_t), -100, device=device, dtype=torch.long)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = compiled(x, y)
        loss.sum().backward()
        for p in compiled.parameters():
            if p.grad is not None:
                p.grad = None
    except Exception as exc:
        msg = str(exc).splitlines()[0]
        print(f"warning: torch.compile failed ({msg}); continuing without compile")
        return model

    print("torch.compile enabled")
    return compiled


def build_scheduler(optimizer: torch.optim.Optimizer, warmup: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(max(1, warmup))
        progress = (step - warmup) / float(max(1, total_steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate_loss(
    model: MathGPT,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["input_ids"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss = model(x, y)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    model.train()
    return total_loss / max(1, n)


@torch.no_grad()
def predict_result(
    model: MathGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    device: torch.device,
    reverse_digits: bool,
    max_new_tokens: int = 192,
    think_start: str = "<think>",
    think_end: str = "</think>",
    reverse_in_think: bool = False,
) -> Generation:
    """Accept human prompt like '123+45='; return think trace + decimal answer."""
    if reverse_in_think:
        model_prompt = prompt.strip().replace(" ", "")
        if not model_prompt.endswith("="):
            model_prompt += "="
    else:
        a, op, b = parse_prompt(prompt)
        model_prompt = to_model_prompt(a, op, b, reverse_digits)
    ids = tokenizer.encode(model_prompt, add_bos=True, add_eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(x, max_new_tokens=max_new_tokens, eos_id=tokenizer.eos_id)
    gen = tokenizer.decode(out[0].tolist()[len(ids) :], skip_special=True)
    think, answer_raw = split_think_and_answer(gen, think_start, think_end)
    answer = decode_model_answer(answer_raw, reverse_digits=False if reverse_in_think else reverse_digits)
    think_show = think if reverse_in_think else (reverse_digit_runs(think) if reverse_digits else think)
    return Generation(answer=answer, think=think_show, raw=gen)


def format_prediction(prompt: str, pred: Generation) -> str:
    if pred.think:
        return f"{prompt}\n<think>\n{pred.think}\n</think>\n{pred.answer}"
    return f"{prompt}{pred.answer}"


@torch.no_grad()
def exact_match_accuracy(
    model: MathGPT,
    problems: list[Problem],
    tokenizer: CharTokenizer,
    device: torch.device,
    reverse_digits: bool,
    max_new_tokens: int = 192,
    limit: int | None = 500,
    think_start: str = "<think>",
    think_end: str = "</think>",
    reverse_in_think: bool = False,
) -> tuple[float, dict[str, float]]:
    model.eval()
    probs = problems if limit is None else problems[:limit]
    correct = 0
    bucket_ok: dict[str, int] = defaultdict(int)
    bucket_n: dict[str, int] = defaultdict(int)

    for p in probs:
        pred = predict_result(
            model,
            tokenizer,
            p.human_prompt,
            device,
            reverse_digits,
            max_new_tokens,
            think_start=think_start,
            think_end=think_end,
            reverse_in_think=reverse_in_think,
        )
        try:
            if hasattr(p, "answer_text"):
                expected = p.answer_text(False)
            else:
                # ExpressionProblem stores exact results as Fraction.
                from dataset import format_fraction

                expected = format_fraction(p.result)
            ok = pred.answer == expected if pred.answer else False
        except ValueError:
            ok = False

        if hasattr(p, "a"):
            digs = max(len(str(p.a)), len(str(p.b)))
            key = f"{digs}d"
        else:
            key = f"{p.expression.count('+') + p.expression.count('-') + p.expression.count('*') + 1}terms"
        bucket_n[key] += 1
        if ok:
            correct += 1
            bucket_ok[key] += 1

    by_digits = {
        k: bucket_ok[k] / bucket_n[k]
        for k in sorted(
            bucket_n,
            key=lambda s: int(s.split("d")[0] if "d" in s else s.split("terms")[0]),
        )
    }
    model.train()
    return correct / max(1, len(probs)), by_digits


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MathGPT on + / - / * with scratchpads")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify the pipeline")
    args = parser.parse_args()

    cfg = Config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.train_size is not None:
        cfg.train_size = args.train_size
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.use_amp = False
    if args.compile:
        cfg.compile_model = True
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.smoke:
        cfg.train_size = 20_000
        cfg.val_size = 1_000
        cfg.epochs = 3
        cfg.batch_size = 128
        cfg.eval_every = 200
        cfg.log_every = 20
        cfg.warmup_steps = 100

    device = get_device(cfg.device)
    setup_cuda_speedups()
    amp_dtype = resolve_amp_dtype(cfg, device)
    dl_kwargs = dataloader_kwargs(cfg, device)
    print(f"device: {device}")
    amp_label = "off" if amp_dtype is None else cfg.amp_dtype
    print(
        f"speed: amp={amp_label}  compile={cfg.compile_model}  "
        f"num_workers={cfg.num_workers}  pin_memory={dl_kwargs.get('pin_memory', False)}  "
        f"micro_batch={cfg.batch_size}  grad_accum={cfg.grad_accum_steps}  "
        f"effective_batch={cfg.batch_size * cfg.grad_accum_steps}"
    )
    print(
        f"target: +/-(<= {cfg.max_number})  *(<= {cfg.mul_max_operand})  "
        f"ood +/-( {cfg.ood_min_digits}-{cfg.ood_max_digits}d )  "
        f"ood *( {cfg.ood_mul_min_digits}d x <={cfg.ood_mul_b_max_digits}d )  "
        f"reverse_digits={cfg.reverse_digits}  scratchpad={cfg.use_scratchpad}  "
        f"scratchpad_ops={cfg.scratchpad_ops}  answer_only_loss={cfg.answer_only_loss}"
    )

    tokenizer = build_tokenizer(cfg)
    print(f"vocab_size: {tokenizer.vocab_size}  tokens: {tokenizer.id_to_token}")

    print("generating eval data...")
    gen_kwargs = dict(
        include_addition=cfg.include_addition,
        include_subtraction=cfg.include_subtraction,
        include_multiplication=cfg.include_multiplication,
        include_division=cfg.include_division,
        mul_max_operand=cfg.mul_max_operand,
        negative_fraction=cfg.negative_fraction,
    )
    # Train never reuses an (a, op, b); also keep train disjoint from val.
    used_keys: set[tuple[int, str, int]] = set()
    val_probs = generate_problems(
        cfg.val_size,
        cfg.max_number,
        seed=cfg.seed + 1,
        exclude=used_keys,
        **gen_kwargs,
    )
    hard_probs = generate_problems(
        cfg.hard_eval_size,
        cfg.ood_max_number,
        seed=cfg.seed + 7,
        min_digits=cfg.ood_min_digits,
        max_digits=cfg.ood_max_digits,
        ops_filter=[
            op
            for op, on in (("+", cfg.include_addition), ("-", cfg.include_subtraction), ("/", cfg.include_division))
            if on
        ],
        **gen_kwargs,
    )
    mul_probs = generate_problems(
        cfg.mul_eval_size,
        cfg.max_number,
        seed=cfg.seed + 9,
        ops_filter=["*"],
        **gen_kwargs,
    )
    ood_mul_probs = generate_problems(
        cfg.ood_mul_size,
        cfg.ood_max_number,
        seed=cfg.seed + 11,
        ops_filter=["*"],
        min_digits=cfg.ood_mul_min_digits,
        max_digits=cfg.ood_mul_min_digits,
        b_min_digits=1,
        b_max_digits=cfg.ood_mul_b_max_digits,
        include_addition=cfg.include_addition,
        include_subtraction=cfg.include_subtraction,
        include_multiplication=cfg.include_multiplication,
        include_division=cfg.include_division,
        mul_max_operand=cfg.ood_mul_max_operand,
    )
    expression_probs = generate_expression_problems(
        cfg.expression_eval_size,
        cfg.ood_max_number,
        seed=cfg.seed + 13,
        max_terms=cfg.expression_max_terms,
        parentheses_fraction=cfg.expression_parentheses_fraction,
    )
    for p in hard_probs + mul_probs + ood_mul_probs:
        used_keys.add((p.a, p.op, p.b))

    ds_kwargs = dict(
        tokenizer=tokenizer,
        max_seq_len=cfg.max_seq_len,
        reverse_digits=cfg.reverse_digits,
        answer_only_loss=cfg.answer_only_loss,
        use_scratchpad=cfg.use_scratchpad,
        scratchpad_ops=cfg.scratchpad_ops,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
        reverse_in_think=cfg.reverse_in_think,
        reverse_start=cfg.reverse_start,
        reverse_end=cfg.reverse_end,
        postfix_start=cfg.postfix_start,
        postfix_end=cfg.postfix_end,
        eval_start=cfg.eval_start,
        eval_end=cfg.eval_end,
    )
    val_ds = EquationDataset(val_probs, **ds_kwargs)
    val_loader = DataLoader(
        val_ds,
        batch_sampler=LengthBucketBatchSampler(val_ds, cfg.batch_size, shuffle=False),
        collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
        **dl_kwargs,
    )
    print(
        f"val: {len(val_ds)}  ood+/-(6-7d): {len(hard_probs)}  mul: {len(mul_probs)}  "
        f"ood_mul: {len(ood_mul_probs)}  expressions(<= {cfg.expression_max_terms} terms): "
        f"{len(expression_probs)}  (train samples are unique and generated each epoch)"
    )

    model = MathGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_seq_len=cfg.max_seq_len,
    ).to(device)
    n_params = model.count_parameters()
    print(f"parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    resume_path = cfg.resume_checkpoint
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        state_dict = checkpoint["model"]
        # A checkpoint saved from torch.compile() has this wrapper prefix.
        if any(key.startswith("_orig_mod.") for key in state_dict):
            state_dict = {
                key.removeprefix("_orig_mod."): value
                for key, value in state_dict.items()
            }
        current = model.state_dict()
        compatible = {
            k: v for k, v in state_dict.items()
            if k in current and current[k].shape == v.shape
        }
        model.load_state_dict(compatible, strict=False)
        # The new vocabulary is larger; copy all old rows into the tied matrix.
        for key in ("tok_emb.weight", "lm_head.weight"):
            if key in state_dict and key in current:
                rows = min(state_dict[key].shape[0], current[key].shape[0])
                with torch.no_grad():
                    current[key][:rows].copy_(state_dict[key][:rows])
        print(
            f"resumed from {resume_path} "
            f"(previous step={checkpoint.get('step', 'unknown')})"
        )
    else:
        print("starting 300M model from random initialization")
    model = maybe_compile_model(
        model,
        device,
        cfg.compile_model,
        max_seq_len=cfg.max_seq_len,
        amp_dtype=amp_dtype,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95)
    )
    scaler = make_grad_scaler(enabled=amp_dtype == torch.float16)
    micro_steps_per_epoch = max(1, math.ceil(cfg.train_size / cfg.batch_size))
    steps_per_epoch = max(1, math.ceil(micro_steps_per_epoch / cfg.grad_accum_steps))
    total_steps = cfg.epochs * steps_per_epoch
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    global_step = 0
    best_val = float("inf")
    best_acc = -1.0
    t0 = time.time()
    eval_kwargs = dict(
        think_start=cfg.think_start,
        think_end=cfg.think_end,
        max_new_tokens=cfg.max_new_tokens,
        reverse_in_think=cfg.reverse_in_think,
    )

    def run_eval(tag: str, full: bool = False) -> float:
        nonlocal best_val, best_acc
        val_loss = evaluate_loss(model, val_loader, device, amp_dtype=amp_dtype)
        mid_val = min(400, len(val_probs))
        mid_hard = min(200, len(hard_probs))
        mid_mul = min(200, len(mul_probs))
        mid_ood_mul = min(200, len(ood_mul_probs))
        mid_expr = min(200, len(expression_probs))
        limit = None if full else mid_val
        acc, by_d = exact_match_accuracy(
            model,
            val_probs,
            tokenizer,
            device,
            reverse_digits=cfg.reverse_digits,
            limit=limit,
            **eval_kwargs,
        )
        hard_acc, _ = exact_match_accuracy(
            model,
            hard_probs,
            tokenizer,
            device,
            reverse_digits=cfg.reverse_digits,
            limit=None if full else mid_hard,
            **eval_kwargs,
        )
        mul_acc, _ = exact_match_accuracy(
            model,
            mul_probs,
            tokenizer,
            device,
            reverse_digits=cfg.reverse_digits,
            limit=None if full else mid_mul,
            **eval_kwargs,
        )
        ood_mul_acc, _ = exact_match_accuracy(
            model,
            ood_mul_probs,
            tokenizer,
            device,
            reverse_digits=cfg.reverse_digits,
            limit=None if full else mid_ood_mul,
            **eval_kwargs,
        )
        expr_acc, _ = exact_match_accuracy(
            model,
            expression_probs,
            tokenizer,
            device,
            reverse_digits=cfg.reverse_digits,
            limit=None if full else mid_expr,
            **eval_kwargs,
        )
        by_d_str = " ".join(f"{k}={v:.2f}" for k, v in by_d.items())
        print(
            f"\n{tag}: val_loss={val_loss:.4f}  acc={acc:.3f}  "
            f"ood+/-(6-7d)={hard_acc:.3f}  mul={mul_acc:.3f}  "
            f"ood_mul={ood_mul_acc:.3f}  expr={expr_acc:.3f}  [{by_d_str}]"
        )
        if acc >= best_acc or (acc == best_acc and val_loss < best_val):
            best_acc = acc
            best_val = val_loss
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": {k: v for k, v in cfg.__dict__.items() if k != "extra_specials"},
                    "tokenizer_chars": cfg.chars,
                    "tokenizer_tokens": tokenizer.id_to_token,
                    "step": global_step,
                    "val_loss": val_loss,
                    "acc": acc,
                    "hard_acc": hard_acc,
                    "mul_acc": mul_acc,
                    "ood_mul_acc": ood_mul_acc,
                    "expr_acc": expr_acc,
                },
                ckpt_path,
            )
            print(f"saved checkpoint -> {ckpt_path}")
        return acc

    for epoch in range(1, cfg.epochs + 1):
        print(f"generating unique train data for epoch {epoch}/{cfg.epochs}...")
        binary_count = int(cfg.train_size * (1.0 - cfg.expression_fraction))
        hard_count = int(binary_count * cfg.train_hard_fraction)
        easy_count = binary_count - hard_count
        train_probs = generate_problems(
            easy_count,
            cfg.train_easy_max_number,
            seed=cfg.seed + epoch * 1_000_003,
            exclude=used_keys,
            **gen_kwargs,
        )
        train_probs.extend(
            generate_problems(
                hard_count,
                cfg.ood_max_number,
                seed=cfg.seed + epoch * 1_000_003 + 31,
                min_digits=cfg.train_hard_min_digits,
                max_digits=cfg.train_hard_max_digits,
                exclude=used_keys,
                **gen_kwargs,
            )
        )
        expression_count = cfg.train_size - len(train_probs)
        if expression_count:
            expression_max_terms = min(
                cfg.expression_max_terms,
                cfg.expression_start_max_terms
                + (epoch - 1) // cfg.expression_growth_every,
            )
            print(f"expression curriculum: max_terms={expression_max_terms}")
            train_probs.extend(
                generate_expression_problems(
                    expression_count,
                    cfg.max_number,
                    seed=cfg.seed + epoch * 1_000_003 + 17,
                    min_terms=cfg.expression_min_terms,
                    max_terms=expression_max_terms,
                    parentheses_fraction=cfg.expression_parentheses_fraction,
                )
            )
        train_ds = EquationDataset(train_probs, **ds_kwargs)
        train_loader = DataLoader(
            train_ds,
            batch_sampler=LengthBucketBatchSampler(
                train_ds,
                cfg.batch_size,
                shuffle=True,
                seed=cfg.seed + epoch,
            ),
            collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
            **dl_kwargs,
        )
        print(f"epoch {epoch} train examples: {len(train_ds)}  unique keys: {len(used_keys)}")

        model.train()
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.epochs}")
        for micro_step, batch in enumerate(pbar):
            x = batch["input_ids"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                _, loss = model(x, y)
            scaler.scale(loss / cfg.grad_accum_steps).backward()

            is_last_micro = micro_step == len(train_loader) - 1
            should_step = ((micro_step + 1) % cfg.grad_accum_steps == 0) or is_last_micro
            if not should_step:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            if global_step % cfg.log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % cfg.eval_every == 0:
                run_eval(f"step {global_step}", full=False)

        run_eval(f"epoch {epoch} done (elapsed={time.time() - t0:.1f}s)", full=True)

    print("training finished.")
    print(f"best acc={best_acc:.3f}  best val_loss={best_val:.4f}")
    print(f"checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
