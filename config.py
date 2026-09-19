from dataclasses import dataclass


@dataclass
class Config:
    # Single-character vocabulary (multi-char tokens come from pad/bos/eos and extra_specials).
    # Semicolons separate reasoning steps; c marks carry in digit-wise scratchpad traces.
    chars: str = (
        "0123456789+-*/=;()"
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
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
    # Total vocabulary size (chars + pad/bos/eos + extra_specials including <unuseN> pads).
    vocab_size: int = 120
    reserved_token_prefix: str = "<unuse"

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

    # Train on larger operands so the model practices the same regime as OOD eval.
    max_number: int = 9_999_999
    # Large-factor multiplication is still short enough for max_seq_len=768.
    mul_max_operand: int = 9_999_999
    include_addition: bool = True
    include_subtraction: bool = True
    include_multiplication: bool = True
    include_division: bool = True
    # Write numbers least-significant-digit first (helps + / -; still used for *)
    reverse_digits: bool = True
    # Keep the user expression normal-order; teach reversal inside <think>.
    reverse_in_think: bool = True
    # Supervised scratchpad between <think> tags, then the final answer
    use_scratchpad: bool = True
    # Digit-wise traces for all ops — the algorithm, not the max width, is what extrapolates
    scratchpad_ops: tuple[str, ...] = ("+", "-", "*")
    # Fraction of each training epoch made from multi-operation expressions.
    expression_fraction: float = 0.8
    expression_max_terms: int = 10
    expression_min_terms: int = 6
    expression_start_max_terms: int = 6
    expression_growth_every: int = 1
    expression_parentheses_fraction: float = 0.8
    expression_train_min_digits: int = 5
    expression_train_max_digits: int = 7
    division_fraction: float = 0.15
    # Continued training: mix familiar examples with longer 6-7 digit problems.
    train_easy_max_number: int = 9_999_999
    train_hard_min_digits: int = 7
    train_hard_max_digits: int = 8
    train_hard_fraction: float = 0.8
    # Probability applied independently to each binary operand.
    negative_fraction: float = 0.25
    # Only backprop on tokens after the prompt '=', including think + answer
    answer_only_loss: bool = True
    train_size: int = 1_000_000
    # Evaluation deliberately emphasizes difficult, large-number cases.
    val_size: int = 1_000
    hard_eval_size: int = 300
    expression_eval_size: int = 300
    expression_eval_min_terms: int = 7
    expression_eval_max_terms: int = 12
    expression_eval_min_digits: int = 7
    expression_eval_max_digits: int = 9
    mul_eval_size: int = 300
    # True length-OOD: 8-9 digit operands are beyond the 6-7 digit training mix.
    ood_max_number: int = 999_999_999
    ood_min_digits: int = 8
    ood_max_digits: int = 9
    ood_mul_size: int = 300
    ood_mul_max_operand: int = 99_999_999
    ood_mul_min_digits: int = 7
    ood_mul_b_max_digits: int = 4
    seed: int = 42

    # Training. Each epoch samples a fresh unique set; no example is reused.
    # RTX 4090 24GB-safe micro-batch for the ~303M model at seq_len 768.
    # Keep the effective batch at 256 through gradient accumulation.
    batch_size: int = 16
    grad_accum_steps: int = 16
    lr: float = 5e-5
    weight_decay: float = 0.05
    # Fine-tune the existing checkpoint on the updated fraction scratchpad.
    epochs: int = 2
    # Full warmup only for grade 1 at global step 0; each later grade uses grade_warmup_steps.
    warmup_steps: int = 1000
    grade_warmup_steps: int = 400
    # Cosine floor within a grade segment: min_lr = lr * lr_min_ratio.
    lr_min_ratio: float = 0.3
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
    # After passing a grade exam, also write checkpoints/mathgpt_grade{N}.pt
    save_grade_checkpoints: bool = True
    grade_checkpoint_name_template: str = "mathgpt_grade{grade_id}.pt"
    equation_fraction: float = 0.0
    curriculum_exam_seed_base: int = 900_001
    save_optimizer_state: bool = True

    @property
    def active_specials(self) -> list[str]:
        """Multi-char tokens used in training data."""
        return [self.think_start, self.think_end, self.error_token]

    @property
    def reserved_specials(self) -> list[str]:
        """Placeholder slots <unuse1>.. for future vocabulary extensions."""
        used = len(self.chars) + 3 + len(self.active_specials)
        n = max(0, self.vocab_size - used)
        p = self.reserved_token_prefix
        return [f"{p}{i}>" for i in range(1, n + 1)]

    @property
    def extra_specials(self) -> list[str]:
        return self.active_specials + self.reserved_specials
