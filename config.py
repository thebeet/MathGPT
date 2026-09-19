from dataclasses import dataclass


@dataclass
class Config:
    # Vocabulary: digits + operators + scratchpad punctuation + special tokens
    # NOTE: changing chars / think tags invalidates old checkpoints (embedding size changes)
    # Semicolons separate compact reasoning steps.
    # New characters are appended after the original vocabulary when migrating.
    chars: str = "0123456789+-*=;()"
    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    think_start: str = "<think>"
    think_end: str = "</think>"
    reverse_start: str = "<reverse>"
    reverse_end: str = "</reverse>"
    postfix_start: str = "<postfix>"
    postfix_end: str = "</postfix>"
    eval_start: str = "<eval>"
    eval_end: str = "</eval>"
    error_token: str = "<err>"

    # Model (~303M params with weight tying)
    # 1024 x 24 layers x 4096 FFN, with 16 attention heads.
    d_model: int = 1024
    n_head: int = 16
    n_layer: int = 24
    d_ff: int = 4096
    dropout: float = 0.0
    # Long mixed expressions need substantially more room than binary problems.
    # Keep enough context for several 6-digit operands and their postfix stages.
    max_seq_len: int = 768
    max_new_tokens: int = 640

    # Train operands are length-balanced in 1..digits(max_*). Leave 7+ digits for OOD eval.
    max_number: int = 999_999
    # Inclusive max factor. 5-digit * 5-digit traces still fit in max_seq_len=256;
    # 6-digit * 6-digit does not (needs ~300 tokens).
    mul_max_operand: int = 999_999
    include_addition: bool = True
    include_subtraction: bool = True
    include_multiplication: bool = True
    include_division: bool = True
    # Write numbers least-significant-digit first (helps + / -; still used for *)
    reverse_digits: bool = True
    # Keep the user expression normal-order; teach reversal inside <think>.
    reverse_in_think: bool = True
    # Set automatically when loading checkpoints produced before structured think.
    legacy_format: bool = False
    # Supervised scratchpad between <think> tags, then the final answer
    use_scratchpad: bool = True
    # Digit-wise traces for all ops — the algorithm, not the max width, is what extrapolates
    scratchpad_ops: tuple[str, ...] = ("+", "-", "*")
    # Fraction of each training epoch made from multi-operation expressions.
    expression_fraction: float = 0.8
    expression_max_terms: int = 8
    expression_min_terms: int = 5
    expression_start_max_terms: int = 5
    expression_growth_every: int = 1
    expression_parentheses_fraction: float = 0.8
    division_fraction: float = 0.15
    # Continued training: mix familiar examples with longer 6-7 digit problems.
    train_easy_max_number: int = 999_999
    train_hard_min_digits: int = 6
    train_hard_max_digits: int = 7
    train_hard_fraction: float = 0.7
    # Probability applied independently to each binary operand.
    negative_fraction: float = 0.25
    # Only backprop on tokens after the prompt '=', including think + answer
    answer_only_loss: bool = True
    train_size: int = 1_000_000
    val_size: int = 500
    hard_eval_size: int = 100
    expression_eval_size: int = 100
    mul_eval_size: int = 100
    # True length-OOD: longer than anything in training
    ood_max_number: int = 9_999_999
    ood_min_digits: int = 6
    ood_max_digits: int = 7
    ood_mul_size: int = 100
    ood_mul_max_operand: int = 999_999
    ood_mul_min_digits: int = 6
    ood_mul_b_max_digits: int = 3
    seed: int = 42

    # Training. Each epoch samples a fresh unique set; no example is reused.
    # RTX 4090 24GB-safe micro-batch for the ~303M model at seq_len 768.
    # Keep the effective batch at 256 through gradient accumulation.
    batch_size: int = 16
    grad_accum_steps: int = 16
    lr: float = 6e-5
    weight_decay: float = 0.1
    epochs: int = 10
    warmup_steps: int = 1000
    grad_clip: float = 1.0
    log_every: int = 100
    eval_every: int = 2000
    device: str = "cuda"
    # Performance: SDPA is always used in model.py; these tune the training loop.
    use_amp: bool = True
    amp_dtype: str = "bf16"  # bf16 | fp16
    compile_model: bool = False
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "mathgpt.pt"
    resume_checkpoint: str = "checkpoints/mathgpt.pt"

    @property
    def extra_specials(self) -> list[str]:
        return [self.think_start, self.think_end, self.error_token]
