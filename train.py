from __future__ import annotations

import argparse
import math
import os
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Callable, Protocol

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from curriculum import (
    CurriculumState,
    ExamRecord,
    default_curriculum,
    generate_grade_exam_problems,
    generate_grade_train_problems,
    merge_curriculum_into_config,
    smoke_curriculum,
    starting_grade,
)
from dataset import (
    EquationDataset,
    Generation,
    ExpressionProblem,
    LinearEquationProblem,
    Problem,
    collate_batch,
    decode_model_answer,
    parse_prompt,
    reverse_digit_runs,
    split_think_and_answer,
    to_model_prompt,
    LengthBucketBatchSampler,
)
from model import MathGPT
from tokenizer import CharTokenizer


def build_tokenizer(cfg: Config) -> CharTokenizer:
    return CharTokenizer(
        chars=cfg.chars,
        pad_token=cfg.pad_token,
        bos_token=cfg.bos_token,
        eos_token=cfg.eos_token,
        extra_specials=cfg.extra_specials,
    )


def load_tokenizer_from_checkpoint(cfg: Config, checkpoint: dict | None) -> CharTokenizer:
    tokenizer = build_tokenizer(cfg)
    if checkpoint is None:
        return tokenizer
    saved = checkpoint.get("tokenizer_tokens")
    if saved is not None and list(saved) != tokenizer.id_to_token:
        raise ValueError(
            "Checkpoint vocabulary does not match current config (chars / special tokens). "
            "Delete the old checkpoint or run with --fresh."
        )
    return tokenizer


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


def optimizer_steps_per_epoch(cfg: Config, train_size: int) -> int:
    micro_batches = max(1, math.ceil(train_size / cfg.batch_size))
    return max(1, math.ceil(micro_batches / cfg.grad_accum_steps))


def build_grade_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup: int,
    segment_steps: int,
    min_ratio: float,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Warmup + cosine decay over one curriculum grade (floor = min_ratio * peak)."""
    warmup = max(1, warmup)
    segment_steps = max(warmup + 1, segment_steps)
    min_ratio = min(1.0, max(0.0, min_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / float(max(1, segment_steps - warmup))
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def start_grade_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Config,
    grade,
    completed_grade_epochs: int,
    *,
    cold_start: bool,
) -> torch.optim.lr_scheduler.LambdaLR:
    steps_per_epoch = optimizer_steps_per_epoch(cfg, grade.train_size)
    segment_steps = max(1, grade.max_epochs * steps_per_epoch)
    if cold_start:
        warmup = cfg.warmup_steps
    else:
        warmup = cfg.grade_warmup_steps
    completed_steps = max(0, completed_grade_epochs * steps_per_epoch)
    last_epoch = completed_steps - 1 if completed_steps > 0 else -1
    sched = build_grade_scheduler(
        optimizer,
        warmup,
        segment_steps,
        cfg.lr_min_ratio,
        last_epoch=last_epoch,
    )
    lr_now = sched.get_last_lr()[0]
    print(
        f"LR schedule: grade {grade.id}  segment_steps={segment_steps}  "
        f"warmup={warmup}  resume_step={completed_steps}  lr={lr_now:.2e}"
    )
    return sched


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
        from dataset import normalize_human_prompt

        model_prompt = normalize_human_prompt(prompt)
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
            if isinstance(p, LinearEquationProblem):
                expected = p.answer_text(False)
            elif isinstance(p, ExpressionProblem):
                from dataset import format_fraction

                expected = format_fraction(p.result)
            elif hasattr(p, "answer_text"):
                expected = p.answer_text(False)
            else:
                from dataset import format_fraction

                expected = format_fraction(p.result)
            ok = pred.answer == expected if pred.answer else False
        except ValueError:
            ok = False

        if isinstance(p, LinearEquationProblem):
            key = "eq"
        elif hasattr(p, "a"):
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


def normalize_checkpoint_state_dict(state_dict: dict) -> dict:
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    return state_dict


def load_model_from_checkpoint(model: MathGPT, checkpoint: dict) -> None:
    state_dict = normalize_checkpoint_state_dict(checkpoint["model"])
    model.load_state_dict(state_dict, strict=True)


def build_checkpoint_payload(
    *,
    model: MathGPT,
    cfg: Config,
    tokenizer: CharTokenizer,
    global_step: int,
    curriculum_state: CurriculumState | None,
    benchmark: dict | None,
) -> dict:
    payload: dict = {
        "model": model.state_dict(),
        "config": {k: v for k, v in cfg.__dict__.items() if k != "extra_specials"},
        "tokenizer_chars": cfg.chars,
        "tokenizer_tokens": tokenizer.id_to_token,
        "step": global_step,
        "benchmark": benchmark or {},
    }
    if curriculum_state is not None:
        payload["curriculum"] = curriculum_state.to_dict()
    return payload


def grade_checkpoint_path(cfg: Config, grade_id: int) -> str:
    name = cfg.grade_checkpoint_name_template.format(grade_id=grade_id)
    return os.path.join(cfg.checkpoint_dir, name)


def save_training_checkpoint(
    path: str,
    *,
    model: MathGPT,
    cfg: Config,
    tokenizer: CharTokenizer,
    global_step: int,
    curriculum_state: CurriculumState | None = None,
    benchmark: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> None:
    payload = build_checkpoint_payload(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        global_step=global_step,
        curriculum_state=curriculum_state,
        benchmark=benchmark,
    )
    if cfg.save_optimizer_state and optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if cfg.save_optimizer_state and scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if cfg.save_optimizer_state and scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, path)
    print(f"saved checkpoint -> {path}")


def infer_curriculum_state_from_checkpoint(checkpoint: dict | None) -> CurriculumState:
    if not checkpoint or "curriculum" not in checkpoint:
        return CurriculumState()
    return CurriculumState.from_dict(checkpoint["curriculum"])


class InterruptHandler(Protocol):
    def __call__(self, step: int, *, rewind_grade_epoch: bool) -> None: ...


def run_training_steps(
    model: MathGPT,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    device: torch.device,
    cfg: Config,
    amp_dtype: torch.dtype | None,
    global_step: int,
    desc: str,
    on_step: Callable[[int], None] | None = None,
    on_interrupt: InterruptHandler | None = None,
) -> int:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc=desc)
    try:
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
            if on_step is not None:
                on_step(global_step)
    except KeyboardInterrupt:
        pbar.close()
        if on_interrupt is not None:
            on_interrupt(global_step, rewind_grade_epoch=True)
        raise SystemExit(130) from None

    return global_step


def run_curriculum_training(
    *,
    cfg: Config,
    model: MathGPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    dl_kwargs: dict,
    curriculum,
    curriculum_state: CurriculumState,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler,
    ckpt_path: str,
) -> None:
    eval_kwargs = dict(
        think_start=cfg.think_start,
        think_end=cfg.think_end,
        max_new_tokens=cfg.max_new_tokens,
        reverse_in_think=cfg.reverse_in_think,
    )
    if len(curriculum_state.passed_grades) >= len(curriculum):
        print(
            f"curriculum already complete (grades passed: {curriculum_state.passed_grades}); "
            "use --fresh to retrain from 一年级."
        )
        return

    t0 = time.time()
    start_id = starting_grade(curriculum_state, curriculum)
    curriculum_state.current_grade = start_id

    print(
        f"curriculum: passed={curriculum_state.passed_grades or 'none'}  "
        f"start at grade {start_id}  grade_epoch={curriculum_state.grade_epoch}"
    )
    print("tip: Ctrl+C saves a checkpoint and exits; rerun train.py to resume.")

    sched_holder: list[torch.optim.lr_scheduler.LRScheduler] = [scheduler]

    def save_on_interrupt(step: int, *, rewind_grade_epoch: bool) -> None:
        nonlocal global_step
        global_step = step
        if rewind_grade_epoch and curriculum_state.grade_epoch > 0:
            curriculum_state.grade_epoch -= 1
        save_training_checkpoint(
            ckpt_path,
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            global_step=global_step,
            curriculum_state=curriculum_state,
            benchmark={
                "interrupted": True,
                "reason": "keyboard_interrupt",
                "grade": curriculum_state.current_grade,
                "grade_epoch": curriculum_state.grade_epoch,
                "passed_grades": list(curriculum_state.passed_grades),
                "step": global_step,
            },
            optimizer=optimizer,
            scheduler=sched_holder[0],
            scaler=scaler,
        )
        print("\ntraining interrupted; resume with: python train.py --device cuda")

    try:
        global_step = _run_curriculum_loop(
            cfg=cfg,
            model=model,
            tokenizer=tokenizer,
            device=device,
            amp_dtype=amp_dtype,
            dl_kwargs=dl_kwargs,
            curriculum=curriculum,
            curriculum_state=curriculum_state,
            global_step=global_step,
            optimizer=optimizer,
            sched_holder=sched_holder,
            scaler=scaler,
            ckpt_path=ckpt_path,
            eval_kwargs=eval_kwargs,
            start_id=start_id,
            save_on_interrupt=save_on_interrupt,
            cold_start=(global_step == 0),
        )
    except KeyboardInterrupt:
        save_on_interrupt(global_step, rewind_grade_epoch=True)
        raise SystemExit(130) from None

    elapsed = time.time() - t0
    if len(curriculum_state.passed_grades) >= len(curriculum):
        print(f"curriculum complete (all {len(curriculum)} grades passed) in {elapsed:.1f}s")
    else:
        print(f"curriculum paused after {elapsed:.1f}s")
    print(f"checkpoint: {ckpt_path}")


def _run_curriculum_loop(
    *,
    cfg: Config,
    model: MathGPT,
    tokenizer: CharTokenizer,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    dl_kwargs: dict,
    curriculum,
    curriculum_state: CurriculumState,
    global_step: int,
    optimizer: torch.optim.Optimizer,
    sched_holder: list[torch.optim.lr_scheduler.LRScheduler],
    scaler,
    ckpt_path: str,
    eval_kwargs: dict,
    start_id: int,
    save_on_interrupt: InterruptHandler,
    cold_start: bool,
) -> int:
    for grade in curriculum:
        if grade.id in curriculum_state.passed_grades:
            continue
        if grade.id < start_id:
            continue

        merge_curriculum_into_config(cfg, grade)
        exam_probs = generate_grade_exam_problems(
            grade,
            seed=cfg.curriculum_exam_seed_base + grade.id * 10_007,
        )
        print(
            f"\n=== {grade.name}（{grade.description}）"
            f"  ops={''.join(grade.ops)}  pass>={grade.pass_accuracy:.0%} ==="
        )

        grade_epoch = curriculum_state.grade_epoch if grade.id == curriculum_state.current_grade else 0
        sched_holder[0] = start_grade_scheduler(
            optimizer,
            cfg,
            grade,
            grade_epoch,
            cold_start=cold_start and grade.id == start_id and grade_epoch == 0,
        )
        cold_start = False
        passed = False

        while grade_epoch < grade.max_epochs:
            grade_epoch += 1
            curriculum_state.current_grade = grade.id
            curriculum_state.grade_epoch = grade_epoch

            train_probs = generate_grade_train_problems(
                grade,
                seed=cfg.seed + grade.id * 1_000_003 + grade_epoch * 97,
            )

            ds_kwargs = dict(
                tokenizer=tokenizer,
                max_seq_len=cfg.max_seq_len,
                reverse_digits=cfg.reverse_digits,
                answer_only_loss=cfg.answer_only_loss,
                use_scratchpad=cfg.use_scratchpad,
                scratchpad_ops=grade.scratchpad_ops,
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
            train_ds = EquationDataset(train_probs, **ds_kwargs)
            train_loader = DataLoader(
                train_ds,
                batch_sampler=LengthBucketBatchSampler(
                    train_ds,
                    cfg.batch_size,
                    shuffle=True,
                    seed=cfg.seed + grade.id * 1000 + grade_epoch,
                ),
                collate_fn=partial(collate_batch, pad_id=tokenizer.pad_id),
                **dl_kwargs,
            )
            print(
                f"{grade.name} epoch {grade_epoch}/{grade.max_epochs}: "
                f"train={len(train_ds)}  exam={len(exam_probs)}"
            )

            global_step = run_training_steps(
                model,
                train_loader,
                optimizer,
                sched_holder[0],
                scaler,
                device,
                cfg,
                amp_dtype,
                global_step,
                desc=f"{grade.name} ep {grade_epoch}",
                on_interrupt=save_on_interrupt,
            )

            if grade_epoch < grade.min_epochs:
                save_training_checkpoint(
                    ckpt_path,
                    model=model,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    curriculum_state=curriculum_state,
                    benchmark={"phase": "warmup", "grade": grade.id, "grade_epoch": grade_epoch},
                    optimizer=optimizer,
                    scheduler=sched_holder[0],
                    scaler=scaler,
                )
                continue

            try:
                exam_acc, exam_by_bucket = exact_match_accuracy(
                    model,
                    exam_probs,
                    tokenizer,
                    device,
                    reverse_digits=cfg.reverse_digits,
                    limit=None,
                    **eval_kwargs,
                )
            except KeyboardInterrupt:
                save_on_interrupt(global_step, rewind_grade_epoch=False)
                raise SystemExit(130) from None
            passed = exam_acc >= grade.pass_accuracy
            record = ExamRecord(
                grade_id=grade.id,
                accuracy=exam_acc,
                passed=passed,
                step=global_step,
                n_problems=len(exam_probs),
            )
            curriculum_state.last_exam = record.to_dict()
            curriculum_state.exam_history.append(record.to_dict())
            benchmark = {
                "grade": grade.id,
                "grade_name": grade.name,
                "exam_accuracy": exam_acc,
                "exam_pass_threshold": grade.pass_accuracy,
                "exam_passed": passed,
                "exam_by_bucket": exam_by_bucket,
                "grade_epoch": grade_epoch,
                "passed_grades": list(curriculum_state.passed_grades),
            }
            save_training_checkpoint(
                ckpt_path,
                model=model,
                cfg=cfg,
                tokenizer=tokenizer,
                global_step=global_step,
                curriculum_state=curriculum_state,
                benchmark=benchmark,
                optimizer=optimizer,
                scheduler=sched_holder[0],
                scaler=scaler,
            )
            bucket_str = " ".join(f"{k}={v:.2f}" for k, v in exam_by_bucket.items())
            status = "PASS" if passed else "RETRY"
            print(
                f"exam {grade.name}: acc={exam_acc:.3f}  need>={grade.pass_accuracy:.2f}  "
                f"[{bucket_str}]  -> {status}"
            )
            if passed:
                curriculum_state.passed_grades.append(grade.id)
                curriculum_state.grade_epoch = 0
                passed_benchmark = {
                    **benchmark,
                    "phase": "grade_passed",
                    "grade_checkpoint_for": grade.id,
                    "passed_grades": list(curriculum_state.passed_grades),
                }
                save_training_checkpoint(
                    ckpt_path,
                    model=model,
                    cfg=cfg,
                    tokenizer=tokenizer,
                    global_step=global_step,
                    curriculum_state=curriculum_state,
                    benchmark=passed_benchmark,
                    optimizer=optimizer,
                    scheduler=sched_holder[0],
                    scaler=scaler,
                )
                if cfg.save_grade_checkpoints:
                    grade_path = grade_checkpoint_path(cfg, grade.id)
                    save_training_checkpoint(
                        grade_path,
                        model=model,
                        cfg=cfg,
                        tokenizer=tokenizer,
                        global_step=global_step,
                        curriculum_state=curriculum_state,
                        benchmark=passed_benchmark,
                        optimizer=optimizer,
                        scheduler=sched_holder[0],
                        scaler=scaler,
                    )
                break

        if not passed:
            print(
                f"stopped: {grade.name} did not reach {grade.pass_accuracy:.0%} "
                f"after {grade.max_epochs} epochs; resume with train.py to continue."
            )
            break

    return global_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MathGPT on + / - / * / with scratchpads")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader worker processes")
    parser.add_argument("--smoke", action="store_true", help="Tiny run to verify the pipeline")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint weights and curriculum state",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint path to resume (default: config resume_checkpoint)",
    )
    args = parser.parse_args()

    cfg = Config()
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.device is not None:
        cfg.device = args.device
    if args.no_amp:
        cfg.use_amp = False
    if args.compile:
        cfg.compile_model = True
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.resume is not None:
        cfg.resume_checkpoint = args.resume
    if args.smoke:
        cfg.batch_size = 128
        cfg.log_every = 20
        cfg.warmup_steps = 100
        cfg.grade_warmup_steps = 50

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

    resume_path = cfg.resume_checkpoint
    checkpoint: dict | None = None
    if not args.fresh and os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    tokenizer = load_tokenizer_from_checkpoint(cfg, checkpoint)
    print(f"vocab_size: {tokenizer.vocab_size}  tokens: {tokenizer.id_to_token}")

    curriculum = smoke_curriculum() if args.smoke else default_curriculum()
    print(
        "curriculum training: grades 1–6 with exams; "
        "checkpoint stores passed grades and latest exam benchmark"
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

    curriculum_state = CurriculumState()
    global_step = 0
    if checkpoint is not None:
        load_model_from_checkpoint(model, checkpoint)
        global_step = int(checkpoint.get("step") or 0)
        curriculum_state = infer_curriculum_state_from_checkpoint(checkpoint)
        print(
            f"resumed from {resume_path} (step={global_step}, "
            f"passed grades={curriculum_state.passed_grades or 'none'})"
        )
    else:
        if args.fresh:
            print("starting from random initialization (--fresh)")
        else:
            print("starting from random initialization (no checkpoint found)")
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
    scheduler = build_grade_scheduler(optimizer, 1, 1, cfg.lr_min_ratio)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)

    if checkpoint and cfg.save_optimizer_state:
        if "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                print(f"warning: could not restore optimizer ({exc})")
        if "scheduler" in checkpoint:
            print(
                "note: per-grade LR schedule is rebuilt from curriculum progress "
                "(checkpoint scheduler state ignored)"
            )
        if "scaler" in checkpoint and amp_dtype == torch.float16:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except ValueError as exc:
                print(f"warning: could not restore grad scaler ({exc})")

    run_curriculum_training(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        device=device,
        amp_dtype=amp_dtype,
        dl_kwargs=dl_kwargs,
        curriculum=curriculum,
        curriculum_state=curriculum_state,
        global_step=global_step,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
