from __future__ import annotations

import argparse
import os

import torch

from config import Config
from dataset import (
    generate_expression_problems,
    generate_linear_equation_problems,
    generate_problems,
)
from model import MathGPT
from train import (
    exact_match_accuracy,
    format_prediction,
    get_device,
    load_model_from_checkpoint,
    load_tokenizer_from_checkpoint,
    predict_result,
)


def load_model(ckpt_path: str, device: torch.device) -> tuple[MathGPT, object, Config]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["config"]
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    tokenizer = load_tokenizer_from_checkpoint(cfg, ckpt)
    model = MathGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg.d_model,
        n_head=cfg.n_head,
        n_layer=cfg.n_layer,
        d_ff=cfg.d_ff,
        dropout=0.0,
        max_seq_len=cfg.max_seq_len,
    ).to(device)
    load_model_from_checkpoint(model, ckpt)
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
    parser.add_argument("--div-size", type=int, default=100)
    parser.add_argument("--expr-size", type=int, default=100)
    parser.add_argument("--eq-size", type=int, default=100)
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
        include_division=cfg.include_division,
        mul_max_operand=cfg.mul_max_operand,
    )
    eval_kw = dict(
        reverse_digits=cfg.reverse_digits,
        limit=None,
        max_new_tokens=cfg.max_new_tokens,
        think_start=cfg.think_start,
        think_end=cfg.think_end,
        reverse_in_think=cfg.reverse_in_think,
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
            for op, on in (
                ("+", cfg.include_addition),
                ("-", cfg.include_subtraction),
                ("/", cfg.include_division),
            )
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
        include_division=cfg.include_division,
        mul_max_operand=cfg.ood_mul_max_operand,
    )
    div = generate_problems(
        args.div_size,
        cfg.max_number,
        seed=1004,
        ops_filter=["/"],
        **gen_kw,
    )
    expressions = generate_expression_problems(
        args.expr_size,
        cfg.max_number,
        seed=1005,
        min_terms=cfg.expression_eval_min_terms,
        max_terms=cfg.expression_eval_max_terms,
        parentheses_fraction=cfg.expression_parentheses_fraction,
        min_digits=cfg.expression_eval_min_digits,
        max_digits=cfg.expression_eval_max_digits,
    )
    equations = generate_linear_equation_problems(
        args.eq_size,
        seed=1006,
        max_abs=99,
        max_coef=12,
        err_fraction=0.15,
    )
    acc, by_d = exact_match_accuracy(model, val, tokenizer, device, **eval_kw)
    hard_acc, _ = exact_match_accuracy(model, hard, tokenizer, device, **eval_kw)
    mul_acc, _ = exact_match_accuracy(model, mul, tokenizer, device, **eval_kw)
    ood_mul_acc, _ = exact_match_accuracy(model, ood_mul, tokenizer, device, **eval_kw)
    div_acc, _ = exact_match_accuracy(model, div, tokenizer, device, **eval_kw)
    expr_acc, _ = exact_match_accuracy(model, expressions, tokenizer, device, **eval_kw)
    eq_acc, _ = exact_match_accuracy(model, equations, tokenizer, device, **eval_kw)
    print(f"acc (in-dist mixed +/-*): {acc:.3f}")
    print(f"ood {cfg.ood_min_digits}-{cfg.ood_max_digits}d (+/-/): {hard_acc:.3f}")
    print(f"mul in-dist (<= {cfg.mul_max_operand}): {mul_acc:.3f}")
    print(
        f"ood mul {cfg.ood_mul_min_digits}d x <={cfg.ood_mul_b_max_digits}d: {ood_mul_acc:.3f}"
    )
    print(f"div in-dist (<= {cfg.max_number}): {div_acc:.3f}")
    print(
        f"expressions ({cfg.expression_eval_min_terms}-{cfg.expression_eval_max_terms} terms): "
        f"{expr_acc:.3f}"
    )
    print(f"equations (linear, x= integer): {eq_acc:.3f}")
    print("by digits (mixed val):", " ".join(f"{k}={v:.3f}" for k, v in by_d.items()))

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
        "8/2=",
        "7/3=",
        "2*3+4*5=",
        "3*x+5=14",
        "0*x+3=5",
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
