from __future__ import annotations

import argparse
import os

import torch

from config import Config
from dataset import generate_problems
from model import MathGPT
from tokenizer import CharTokenizer
from train import (
    build_tokenizer,
    exact_match_accuracy,
    format_prediction,
    get_device,
    predict_result,
)


def load_model(ckpt_path: str, device: torch.device) -> tuple[MathGPT, CharTokenizer, Config]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = ckpt["config"]
    legacy_format = "reverse_in_think" not in raw
    # Backward-compatible: old ckpts used reverse_answer
    if "reverse_digits" not in raw and "reverse_answer" in raw:
        raw = dict(raw)
        raw["reverse_digits"] = raw.pop("reverse_answer")
    if legacy_format:
        raw = dict(raw)
        raw["legacy_format"] = True
        raw["reverse_in_think"] = False
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    tokenizer = CharTokenizer(
        tokens=ckpt.get("tokenizer_tokens"),
        extra_specials=cfg.extra_specials,
    ) if ckpt.get("tokenizer_tokens") else build_tokenizer(cfg)
    model = MathGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        d_ff=cfg.d_ff,
        dropout=0.0,
        max_seq_len=cfg.max_seq_len,
    ).to(device)
    state_dict = ckpt["model"]
    # torch.compile wraps the module and prefixes saved parameter names with
    # `_orig_mod.`. Inference constructs the normal (uncompiled) module.
    # Normalize compiled checkpoints so both save formats load transparently.
    if any(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }
    current = model.state_dict()
    compatible = {k: v for k, v in state_dict.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    for key in ("tok_emb.weight", "lm_head.weight"):
        if key in state_dict and key in current:
            rows = min(state_dict[key].shape[0], current[key].shape[0])
            with torch.no_grad():
                current[key][:rows].copy_(state_dict[key][:rows])
    model.eval()
    return model, tokenizer, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate / interact with MathGPT")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "mathgpt.pt"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="e.g. 123+45=")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--hard-size", type=int, default=200)
    parser.add_argument("--mul-size", type=int, default=100)
    parser.add_argument("--ood-mul-size", type=int, default=100)
    args = parser.parse_args()

    cfg_default = Config()
    device = get_device(args.device or cfg_default.device)
    model, tokenizer, cfg = load_model(args.ckpt, device)
    print(f"loaded {args.ckpt} on {device}")
    print(f"parameters: {model.count_parameters():,}")
    print(f"max_number={cfg.max_number}  reverse_digits={cfg.reverse_digits}")

    gen_kw = dict(
        include_addition=cfg.include_addition,
        include_subtraction=cfg.include_subtraction,
        include_multiplication=cfg.include_multiplication,
        mul_max_operand=cfg.mul_max_operand,
    )
    val = generate_problems(args.val_size, cfg.max_number, seed=999, **gen_kw)
    hard = generate_problems(
        args.hard_size,
        cfg.ood_max_number,
        seed=1001,
        min_digits=cfg.ood_min_digits,
        max_digits=cfg.ood_max_digits,
        ops_filter=[
            op
            for op, on in (("+", cfg.include_addition), ("-", cfg.include_subtraction))
            if on
        ],
        **gen_kw,
    )
    mul = generate_problems(
        args.mul_size,
        cfg.max_number,
        seed=1002,
        ops_filter=["*"],
        **gen_kw,
    )
    ood_mul = generate_problems(
        args.ood_mul_size,
        cfg.ood_max_number,
        seed=1003,
        ops_filter=["*"],
        min_digits=cfg.ood_mul_min_digits,
        max_digits=cfg.ood_mul_min_digits,
        b_min_digits=1,
        b_max_digits=cfg.ood_mul_b_max_digits,
        include_addition=cfg.include_addition,
        include_subtraction=cfg.include_subtraction,
        include_multiplication=cfg.include_multiplication,
        mul_max_operand=cfg.ood_mul_max_operand,
    )
    acc, by_d = exact_match_accuracy(
        model,
        val,
        tokenizer,
        device,
        reverse_digits=cfg.reverse_digits,
        limit=None,
        max_new_tokens=cfg.max_new_tokens,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
    )
    hard_acc, _ = exact_match_accuracy(
        model,
        hard,
        tokenizer,
        device,
        reverse_digits=cfg.reverse_digits,
        limit=None,
        max_new_tokens=cfg.max_new_tokens,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
    )
    mul_acc, _ = exact_match_accuracy(
        model,
        mul,
        tokenizer,
        device,
        reverse_digits=cfg.reverse_digits,
        limit=None,
        max_new_tokens=cfg.max_new_tokens,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
    )
    ood_mul_acc, _ = exact_match_accuracy(
        model,
        ood_mul,
        tokenizer,
        device,
        reverse_digits=cfg.reverse_digits,
        limit=None,
        max_new_tokens=cfg.max_new_tokens,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
    )
    print(f"acc (in-dist mixed): {acc:.3f}")
    print(f"ood +/- {cfg.ood_min_digits}-{cfg.ood_max_digits}d: {hard_acc:.3f}")
    print(f"mul in-dist (<= {cfg.mul_max_operand}): {mul_acc:.3f}")
    print(
        f"ood mul {cfg.ood_mul_min_digits}d x <={cfg.ood_mul_b_max_digits}d: {ood_mul_acc:.3f}"
    )
    print("by digits:", " ".join(f"{k}={v:.3f}" for k, v in by_d.items()))

    demos = [
        "3+5=",
        "1-2=",
        "12*7=",
        "100-23=",
        "5-10=",
        "99*99=",
        "1234*56=",
        "12345*56789=",
        "123456*78=",
        "99999*99999=",
        "999+1=",
        "1234+5678=",
        "999999+1=",
        "1000000-1=",
        "654321+123456=",
        "1000000-999999=",
    ]
    print("\ndemos:")
    for p in demos:
        pred = predict_result(
            model,
            tokenizer,
            p,
            device,
            reverse_digits=cfg.reverse_digits,
            max_new_tokens=cfg.max_new_tokens,
            think_start=cfg.think_start,
            think_end=cfg.think_end,
            reverse_in_think=cfg.reverse_in_think,
        )
        print(format_prediction(p, pred))
        print()

    if args.prompt:
        prompt = args.prompt.strip().replace(" ", "")
        if not prompt.endswith("="):
            prompt += "="
        pred = predict_result(
            model,
            tokenizer,
            prompt,
            device,
            cfg.reverse_digits,
            max_new_tokens=cfg.max_new_tokens,
            think_start=cfg.think_start,
            think_end=cfg.think_end,
            reverse_in_think=cfg.reverse_in_think,
        )
        print(format_prediction(prompt, pred))

    if args.interactive:
        print("\ninteractive mode (Ctrl+C to quit). type prompts like 42+8=")
        while True:
            try:
                p = input("> ").strip().replace(" ", "")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not p:
                continue
            if not p.endswith("="):
                p += "="
            print(
                format_prediction(
                    p,
                    predict_result(
                        model,
                        tokenizer,
                        p,
                        device,
                        cfg.reverse_digits,
                        max_new_tokens=cfg.max_new_tokens,
                        think_start=cfg.think_start,
                        think_end=cfg.think_end,
                        reverse_in_think=cfg.reverse_in_think,
                    ),
                )
            )


if __name__ == "__main__":
    main()
