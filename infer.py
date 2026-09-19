from __future__ import annotations

import argparse
import os
import sys

import torch

from config import Config
from evaluate import load_model
from dataset import normalize_human_prompt
from train import format_prediction, get_device, predict_result


def normalize_prompt(text: str) -> str:
    return normalize_human_prompt(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast inference for MathGPT (no benchmark)")
    parser.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "mathgpt.pt"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--prompt", "-p", type=str, default=None, help="e.g. 12*7= or 123+45")
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="REPL mode (default if no --prompt)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"checkpoint not found: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    cfg_default = Config()
    device = get_device(args.device or cfg_default.device)
    model, tokenizer, cfg = load_model(args.ckpt, device)

    interactive = args.interactive or args.prompt is None

    def answer(prompt: str) -> str:
        pred = predict_result(
            model,
            tokenizer,
            prompt,
            device,
            reverse_digits=cfg.reverse_digits,
            max_new_tokens=cfg.max_new_tokens,
            think_start=cfg.think_start,
            think_end=cfg.think_end,
            reverse_in_think=cfg.reverse_in_think,
        )
        return format_prediction(prompt, pred)

    if args.prompt is not None:
        prompt = normalize_prompt(args.prompt)
        print(answer(prompt))

    if interactive:
        if args.prompt is None:
            print(f"loaded {args.ckpt} on {device}  ({model.count_parameters():,} params)")
            print("type an equation like 12*7= or 29832+43431=  (Ctrl+C to quit)")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line:
                continue
            try:
                prompt = normalize_prompt(line)
            except ValueError as e:
                print(f"error: {e}")
                continue
            try:
                print(answer(prompt))
            except Exception as e:
                print(f"error: {e}")


if __name__ == "__main__":
    main()
