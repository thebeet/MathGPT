from __future__ import annotations

import random
import re
from fractions import Fraction
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, Sampler

from tokenizer import CharTokenizer


def rev_digits(n: int) -> str:
    """
    Least-significant digit first.
    Positive: 123 -> '321'
    Negative: -12 -> '21-'  (digits reversed, sign trailing)
    """
    return str(n)[::-1]


def fmt_num(n: int, reverse_digits: bool) -> str:
    return rev_digits(n) if reverse_digits else str(n)


def format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_raw_fraction(numerator: int, denominator: int) -> str:
    """Format a fraction without reducing it for use in reasoning traces."""
    if denominator == 0:
        raise ZeroDivisionError("division by zero")
    if numerator == 0:
        return "0"
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    if denominator == 1:
        return str(numerator)
    return f"{numerator}/{denominator}"


def evaluate_postfix_raw(
    postfix: list[str],
) -> tuple[Fraction, list[str]]:
    """Evaluate postfix while retaining unreduced intermediate fractions.

    The exact Fraction result is kept separately for correctness/final answers;
    trace values use the arithmetic numerator and denominator directly.
    """
    stack: list[tuple[int, int]] = []
    steps: list[str] = []
    for tok in postfix:
        if tok.isdigit():
            stack.append((int(tok), 1))
            continue
        if len(stack) < 2:
            raise ValueError("Invalid postfix expression")
        right_n, right_d = stack.pop()
        left_n, left_d = stack.pop()
        if tok == "/" and right_n == 0:
            raise ZeroDivisionError("division by zero")
        if tok == "+":
            result_n, result_d = left_n * right_d + right_n * left_d, left_d * right_d
        elif tok == "-":
            result_n, result_d = left_n * right_d - right_n * left_d, left_d * right_d
        elif tok == "*":
            result_n, result_d = left_n * right_n, left_d * right_d
        elif tok == "/":
            result_n, result_d = left_n * right_d, left_d * right_n
        else:
            raise ValueError(f"Unknown operator: {tok!r}")
        # Zero is the one numerator-based simplification we allow. Keep all
        # other common factors intact (for example, retain 8/8 as 8/8).
        if result_n == 0:
            result_n, result_d = 0, 1
        elif result_d < 0:
            result_n, result_d = -result_n, -result_d
        left = format_raw_fraction(left_n, left_d)
        right = format_raw_fraction(right_n, right_d)
        result = format_raw_fraction(result_n, result_d)
        # Do not spend a reasoning step restating an already unchanged value,
        # e.g. 12/5=12/5 when a division merely produces that fraction.
        if f"{left}{tok}{right}" != result:
            if tok in {"+", "-"} and (left_d != 1 or right_d != 1):
                sign = "+" if tok == "+" else "-"
                expanded = (
                    f"({left_n}*{right_d}{sign}{right_n}*{left_d})/"
                    f"({left_d}*{right_d})"
                )
                numerator = f"({left_n * right_d}{sign}{right_n * left_d})"
                numeric = f"{numerator}/{left_d * right_d}"
                steps.append(f"{left}{tok}{right}={expanded}={numeric}={result}")
            else:
                steps.append(f"{left}{tok}{right}={result}")
        stack.append((result_n, result_d))
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    numerator, denominator = stack[0]
    return Fraction(numerator, denominator), steps


def parse_prompt(prompt: str) -> tuple[int, str, int]:
    """Parse human prompt '123+45=' / '12*3=' into (a, op, b)."""
    body = prompt.strip().replace(" ", "").rstrip("=")
    match = re.fullmatch(r"(-?\d+)([+*/])(-?\d+)", body)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))
    match = re.fullmatch(r"(-?\d+)-(-?\d+)", body)
    if match:
        return int(match.group(1)), "-", int(match.group(2))
    raise ValueError(f"Invalid prompt: {prompt!r}")


def to_model_prompt(a: int, op: str, b: int, reverse_digits: bool) -> str:
    if reverse_digits:
        return f"{rev_digits(a)}{op}{rev_digits(b)}="
    return f"{a}{op}{b}="


def _lex_expression(expr: str) -> list[str]:
    tokens = re.findall(r"\d+|[()+*/-]", expr.replace(" ", ""))
    if "".join(tokens) != expr.replace(" ", ""):  # reject unsupported syntax
        raise ValueError(f"Invalid expression: {expr!r}")
    return tokens


def expression_postfix(expr: str) -> list[str]:
    """Convert a binary +,-,* expression to canonical postfix tokens."""
    output: list[str] = []
    ops: list[str] = []
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    for tok in _lex_expression(expr):
        if tok.isdigit():
            output.append(tok)
        elif tok in precedence:
            while ops and ops[-1] != "(" and precedence[ops[-1]] >= precedence[tok]:
                output.append(ops.pop())
            ops.append(tok)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if not ops:
                raise ValueError(f"Unbalanced parentheses: {expr!r}")
            ops.pop()
    while ops:
        if ops[-1] == "(":
            raise ValueError(f"Unbalanced parentheses: {expr!r}")
        output.append(ops.pop())
    return output


def evaluate_postfix(postfix: list[str]) -> tuple[Fraction, list[tuple[Fraction, str, Fraction, Fraction]]]:
    """Evaluate postfix and return result plus (left, op, right, result) steps."""
    stack: list[Fraction] = []
    steps: list[tuple[Fraction, str, Fraction, Fraction]] = []
    for tok in postfix:
        if tok.isdigit():
            stack.append(Fraction(int(tok)))
            continue
        if len(stack) < 2:
            raise ValueError("Invalid postfix expression")
        right, left = stack.pop(), stack.pop()
        if tok == "/" and right == 0:
            raise ZeroDivisionError("division by zero")
        result = {"+": left + right, "-": left - right, "*": left * right, "/": left / right}[tok]
        steps.append((left, tok, right, result))
        stack.append(result)
    if len(stack) != 1:
        raise ValueError("Invalid postfix expression")
    return stack[0], steps


def _reverse_stage(numbers: list[str]) -> str:
    return ";".join(s[::-1] for s in numbers)


def _postfix_stage(postfix: list[str]) -> str:
    return ";".join(
        format_fraction(Fraction(tok)) if re.fullmatch(r"-?\d+(?:/\d+)?", tok) else tok
        for tok in postfix
    )


def _postfix_reduction_stages(
    postfix: list[str],
    steps: list[tuple[int, str, int, int]],
) -> list[str]:
    """Return the LSD-first postfix expression after each binary reduction."""
    remaining = postfix[:]
    stages = [_postfix_stage(postfix)]
    for _, _, _, value in steps:
        for index in range(2, len(remaining)):
            if (
                remaining[index] in {"+", "-", "*"}
                and remaining[index - 1].lstrip("-").isdigit()
                and remaining[index - 2].lstrip("-").isdigit()
            ):
                remaining[index - 2 : index + 1] = [str(value)]
                stages.append(_postfix_stage(remaining))
                break
        else:
            raise ValueError("Invalid postfix expression")

    return stages


def expression_train_text(
    expr: str,
    reverse_start: str = "<reverse>",
    reverse_end: str = "</reverse>",
    postfix_start: str = "<postfix>",
    postfix_end: str = "</postfix>",
    eval_start: str = "<eval>",
    eval_end: str = "</eval>",
    think_start: str = "<think>",
    think_end: str = "</think>",
) -> str:
    postfix = expression_postfix(expr)
    result, steps = evaluate_postfix_raw(postfix)
    return (
        f"{expr}={think_start}"
        + ";".join(steps)
        + f"{think_end}{result}"
    )


def reverse_digit_runs(text: str) -> str:
    """Reverse each contiguous run of digits (for displaying LSD-first traces)."""
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i].isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            out.append(text[i:j][::-1])
            i = j
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def decode_model_answer(text: str, reverse_digits: bool) -> str:
    """
    Convert model answer tokens back to a normal decimal string.
    reverse_digits format: '<digits>' or '<digits>-' for negatives.
    """
    if not text:
        return ""
    if text.startswith("<err>"):
        return "<err>"
    if not reverse_digits and re.fullmatch(r"-?\d+/\d+", text.strip()):
        try:
            return format_fraction(Fraction(text.strip()))
        except ValueError:
            return ""

    if reverse_digits:
        i = 0
        while i < len(text) and text[i].isdigit():
            i += 1
        raw = text[:i]
        if not raw:
            return ""
        neg = i < len(text) and text[i] == "-"
        value = int(raw[::-1])
        return str(-value if neg else value)

    neg = text[0] == "-"
    start = 1 if neg else 0
    digits: list[str] = []
    for ch in text[start:]:
        if ch.isdigit():
            digits.append(ch)
        else:
            break
    if not digits:
        return ""
    value = int("".join(digits))
    return str(-value if neg else value)


def split_think_and_answer(generated: str, think_start: str, think_end: str) -> tuple[str, str]:
    """Return (think_body, answer_text) from a model continuation after the prompt."""
    start = generated.find(think_start)
    end = generated.find(think_end)
    if start != -1 and end != -1 and end > start:
        think = generated[start + len(think_start) : end]
        answer = generated[end + len(think_end) :]
        return think, answer
    return "", generated


def mul_scratchpad(a: int, b: int, reverse_digits: bool) -> str:
    """Schoolbook multiply: each digit of b times a (with place), then add partials."""
    steps: list[str] = []
    partials: list[int] = []
    n = b
    k = 0
    a_s = fmt_num(a, reverse_digits)
    while True:
        d = n % 10
        p = a * d * (10**k)
        steps.append(f"{a_s}*{d}={fmt_num(p, reverse_digits)}")
        partials.append(p)
        n //= 10
        k += 1
        if n == 0:
            break

    total = partials[0]
    for p in partials[1:]:
        prev = total
        total = total + p
        steps.append(
            f"{fmt_num(prev, reverse_digits)}+{fmt_num(p, reverse_digits)}="
            f"{fmt_num(total, reverse_digits)}"
        )
    return ";".join(steps)


def add_scratchpad(a: int, b: int, reverse_digits: bool) -> str:
    """Digit-wise add with carry. Digits are already in LSD-first order."""
    sa = str(a)[::-1]
    sb = str(b)[::-1]
    n = max(len(sa), len(sb))
    sa = sa.ljust(n, "0")
    sb = sb.ljust(n, "0")
    carry = 0
    steps: list[str] = []
    for i in range(n):
        da, db = int(sa[i]), int(sb[i])
        sm = da + db + carry
        digit, new_c = sm % 10, sm // 10
        if carry:
            steps.append(f"{da}+{db}+{carry}={digit}c{new_c}")
        else:
            steps.append(f"{da}+{db}={digit}c{new_c}")
        carry = new_c
    if carry:
        steps.append(f"c{carry}")
    return ";".join(steps)


def sub_scratchpad(a: int, b: int, reverse_digits: bool) -> str:
    """Digit-wise subtract with borrow on the larger operand."""
    if a < b:
        return sub_scratchpad(b, a, reverse_digits)

    sx = str(a)[::-1]
    sy = str(b)[::-1]
    n = max(len(sx), len(sy))
    sx = sx.ljust(n, "0")
    sy = sy.ljust(n, "0")
    borrow = 0
    steps: list[str] = []
    for i in range(n):
        dx, dy = int(sx[i]), int(sy[i])
        raw = dx - borrow - dy
        if raw < 0:
            digit = raw + 10
            new_b = 1
        else:
            digit = raw
            new_b = 0
        steps.append(f"{dx}-{dy}-{borrow}={digit}c{new_b}")
        borrow = new_b
    return ";".join(steps)


def make_scratchpad(a: int, op: str, b: int, reverse_digits: bool) -> str:
    if op not in {"+", "-", "*"}:
        raise ValueError(f"Unknown op: {op!r}")
    if op == "/":
        if b == 0:
            return f"{fmt_num(a, reverse_digits)}{op}{fmt_num(b, reverse_digits)}=<err>"
        result = Fraction(a, b)
    else:
        result = {"+": a + b, "-": a - b, "*": a * b}[op]
    return (
        f"{fmt_num(a, reverse_digits)}{op}{fmt_num(b, reverse_digits)}="
        f"{format_fraction(result) if isinstance(result, Fraction) else fmt_num(result, reverse_digits)}"
    )


@dataclass
class Generation:
    answer: str
    think: str
    raw: str


@dataclass(frozen=True)
class Problem:
    a: int
    op: str
    b: int
    result: Fraction | int | str

    @property
    def human_prompt(self) -> str:
        return f"{self.a}{self.op}{self.b}="

    def model_prompt(self, reverse_digits: bool) -> str:
        return to_model_prompt(self.a, self.op, self.b, reverse_digits)

    def answer_text(self, reverse_digits: bool) -> str:
        if self.result == "<err>":
            return "<err>"
        if isinstance(self.result, Fraction):
            return format_fraction(self.result)
        return fmt_num(self.result, reverse_digits)

    def scratchpad(self, reverse_digits: bool) -> str:
        return make_scratchpad(self.a, self.op, self.b, reverse_digits)

    def train_text(
        self,
        reverse_digits: bool,
        use_scratchpad: bool = True,
        scratchpad_ops: tuple[str, ...] = ("*",),
        think_start: str = "<think>",
        think_end: str = "</think>",
        reverse_in_think: bool = False,
        reverse_start: str = "<reverse>",
        reverse_end: str = "</reverse>",
        postfix_start: str = "<postfix>",
        postfix_end: str = "</postfix>",
        eval_start: str = "<eval>",
        eval_end: str = "</eval>",
    ) -> str:
        # New format keeps the user-facing prompt and final answer normal-order;
        # reversal is an explicit learned step inside <think>.
        if reverse_in_think:
            prompt = self.human_prompt
            return (
                f"{prompt}{think_start}"
                f"{self.a}{self.op}{self.b}={self.answer_text(False)}"
                f"{think_end}{self.answer_text(False)}"
            )

        prompt = self.model_prompt(reverse_digits)
        answer = self.answer_text(reverse_digits)
        if not use_scratchpad or self.op not in scratchpad_ops:
            return prompt + answer
        think = self.scratchpad(reverse_digits)
        return f"{prompt}{think_start}{think}{think_end}{answer}"


@dataclass(frozen=True)
class ExpressionProblem:
    expression: str
    result: int

    @property
    def human_prompt(self) -> str:
        return f"{self.expression}="

    def train_text(
        self,
        reverse_digits: bool = True,
        use_scratchpad: bool = True,
        scratchpad_ops: tuple[str, ...] = ("+", "-", "*"),
        think_start: str = "<think>",
        think_end: str = "</think>",
        reverse_in_think: bool = True,
        reverse_start: str = "<reverse>",
        reverse_end: str = "</reverse>",
        postfix_start: str = "<postfix>",
        postfix_end: str = "</postfix>",
        eval_start: str = "<eval>",
        eval_end: str = "</eval>",
    ) -> str:
        return expression_train_text(
            self.expression,
            reverse_start,
            reverse_end,
            postfix_start,
            postfix_end,
            eval_start,
            eval_end,
            think_start,
            think_end,
        )


def generate_expression_problems(
    n: int,
    max_number: int,
    seed: int = 0,
    min_terms: int = 2,
    max_terms: int = 4,
    parentheses_fraction: float = 0.5,
    min_digits: int | None = None,
    max_digits: int | None = None,
) -> list[ExpressionProblem]:
    """Generate valid mixed expressions for supervised postfix learning."""
    rng = random.Random(seed)
    out: list[ExpressionProblem] = []
    seen: set[str] = set()
    ops = ("+", "-", "*", "/")
    min_terms = max(2, min_terms)
    max_terms = max(min_terms, max_terms)
    while len(out) < n:
        terms = rng.randint(min_terms, max_terms)
        values = [
            sample_number(rng, max_number, min_digits=min_digits, max_digits=max_digits)
            for _ in range(terms)
        ]
        operators = [rng.choice(ops) for _ in range(terms - 1)]
        parts = [str(values[0])]
        for op, value in zip(operators, values[1:]):
            parts.extend((op, str(value)))
        expr = "".join(parts)
        # Parenthesize one local subexpression, e.g. 1+(3-2)*2-1.
        if terms >= 3 and rng.random() < parentheses_fraction:
            group_len = 2 if terms < 5 or rng.random() < 0.75 else 3
            start = rng.randint(0, terms - group_len)
            tokens: list[str] = []
            for i, value in enumerate(values):
                if i == start:
                    tokens.append("(")
                tokens.append(str(value))
                if i == start + group_len - 1:
                    tokens.append(")")
                if i < terms - 1:
                    tokens.append(operators[i])
            expr = "".join(tokens)
        if expr in seen:
            continue
        try:
            result, _ = evaluate_postfix(expression_postfix(expr))
        except (ValueError, ZeroDivisionError):
            continue
        seen.add(expr)
        out.append(ExpressionProblem(expr, result))
    return out


def _digit_bounds(max_number: int, digits: int) -> tuple[int, int]:
    lo = 0 if digits == 1 else 10 ** (digits - 1)
    hi = min(max_number, 10**digits - 1)
    return lo, hi


def sample_number(
    rng: random.Random,
    max_number: int,
    min_digits: int | None = None,
    max_digits: int | None = None,
) -> int:
    """Uniform over digit lengths, then uniform within that length (helps length generalization)."""
    cap_d = len(str(max_number))
    lo_d = 1 if min_digits is None else min_digits
    hi_d = cap_d if max_digits is None else max_digits
    lo_d = max(1, min(lo_d, cap_d))
    hi_d = max(lo_d, min(hi_d, cap_d))
    d = rng.randint(lo_d, hi_d)
    lo, hi = _digit_bounds(max_number, d)
    return rng.randint(lo, hi)


def sample_operand_pair(
    rng: random.Random,
    max_number: int,
    min_digits: int | None = None,
    max_digits: int | None = None,
    b_min_digits: int | None = None,
    b_max_digits: int | None = None,
    swap: bool = True,
    negative_fraction: float = 0.0,
) -> tuple[int, int]:
    a = sample_number(rng, max_number, min_digits, max_digits)
    b = sample_number(
        rng,
        max_number,
        b_min_digits if b_min_digits is not None else min_digits,
        b_max_digits if b_max_digits is not None else max_digits,
    )
    if rng.random() < negative_fraction:
        a = -a
    if rng.random() < negative_fraction:
        b = -b
    if swap and rng.random() < 0.5:
        return b, a
    return a, b


def enabled_ops(
    include_addition: bool = True,
    include_subtraction: bool = True,
    include_multiplication: bool = True,
    include_division: bool = False,
) -> list[str]:
    ops: list[str] = []
    if include_addition:
        ops.append("+")
    if include_subtraction:
        ops.append("-")
    if include_multiplication:
        ops.append("*")
    if include_division:
        ops.append("/")
    if not ops:
        raise ValueError("At least one operation must be enabled")
    return ops


def make_problem(
    max_number: int,
    rng: random.Random,
    include_addition: bool = True,
    include_subtraction: bool = True,
    include_multiplication: bool = True,
    include_division: bool = False,
    mul_max_operand: int = 99_999,
) -> Problem:
    op = rng.choice(
        enabled_ops(include_addition, include_subtraction, include_multiplication, include_division)
    )
    if op == "*":
        a, b = sample_operand_pair(rng, mul_max_operand)
        return Problem(a, "*", b, a * b)
    if op == "/":
        a, b = sample_operand_pair(rng, max_number)
        return Problem(a, "/", b, "<err>" if b == 0 else Fraction(a, b))
    a, b = sample_operand_pair(rng, max_number)
    if op == "-":
        return Problem(a, "-", b, a - b)
    return Problem(a, "+", b, a + b)


def generate_problems(
    n: int,
    max_number: int,
    include_subtraction: bool = True,
    seed: int = 0,
    min_digits: int | None = None,
    max_digits: int | None = None,
    b_min_digits: int | None = None,
    b_max_digits: int | None = None,
    include_addition: bool = True,
    include_multiplication: bool = True,
    include_division: bool = False,
    mul_max_operand: int = 99_999,
    negative_fraction: float = 0.25,
    ops_filter: list[str] | None = None,
    exclude: set[tuple[int, str, int]] | None = None,
) -> list[Problem]:
    """
    Generate unique problems. If exclude is provided, skip those keys and add new ones
    into the same set (so callers can keep uniqueness across multiple calls).
    """
    rng = random.Random(seed)
    out: list[Problem] = []
    seen = exclude if exclude is not None else set()
    attempts = 0
    max_attempts = max(n * 80, 10_000)
    default_ops = enabled_ops(include_addition, include_subtraction, include_multiplication, include_division)

    while len(out) < n and attempts < max_attempts:
        attempts += 1
        if ops_filter is not None:
            op = rng.choice(ops_filter)
        else:
            op = rng.choice(default_ops)

        if op == "*":
            a, b = sample_operand_pair(
                rng,
                mul_max_operand,
                min_digits,
                max_digits,
                b_min_digits,
                b_max_digits,
                negative_fraction=negative_fraction,
            )
            p = Problem(a, "*", b, a * b)
        elif op == "/":
            a, b = sample_operand_pair(rng, max_number, min_digits, max_digits, b_min_digits, b_max_digits, negative_fraction=negative_fraction)
            p = Problem(a, "/", b, "<err>" if b == 0 else Fraction(a, b))
        else:
            a, b = sample_operand_pair(
                rng,
                max_number,
                min_digits,
                max_digits,
                b_min_digits,
                b_max_digits,
                negative_fraction=negative_fraction,
            )
            if op == "-":
                p = Problem(a, "-", b, a - b)
            else:
                p = Problem(a, "+", b, a + b)

        key = (p.a, p.op, p.b)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


class EquationDataset(Dataset):
    """Next-token prediction over equations; optional answer-only loss mask."""

    def __init__(
        self,
        problems: list[Problem | ExpressionProblem],
        tokenizer: CharTokenizer,
        max_seq_len: int,
        reverse_digits: bool = True,
        answer_only_loss: bool = True,
        use_scratchpad: bool = True,
        scratchpad_ops: tuple[str, ...] = ("*",),
        think_start: str = "<think>",
        think_end: str = "</think>",
        reverse_in_think: bool = False,
        reverse_start: str = "<reverse>",
        reverse_end: str = "</reverse>",
        postfix_start: str = "<postfix>",
        postfix_end: str = "</postfix>",
        eval_start: str = "<eval>",
        eval_end: str = "</eval>",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reverse_digits = reverse_digits
        self.answer_only_loss = answer_only_loss
        self.use_scratchpad = use_scratchpad
        self.scratchpad_ops = scratchpad_ops
        self.eq_id = tokenizer.token_to_id["="]
        self.examples: list[list[int]] = []

        for p in problems:
            text = p.train_text(
                reverse_digits,
                use_scratchpad=use_scratchpad,
                scratchpad_ops=scratchpad_ops,
                think_start=think_start,
                think_end=think_end,
                reverse_in_think=reverse_in_think,
                reverse_start=reverse_start,
                reverse_end=reverse_end,
                postfix_start=postfix_start,
                postfix_end=postfix_end,
                eval_start=eval_start,
                eval_end=eval_end,
            )
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) <= max_seq_len:
                self.examples.append(ids)

    def __len__(self) -> int:
        return self.examples.__len__()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.examples[idx]
        # Leave padding to collate_batch so attention only sees the longest
        # sequence in the current batch, rather than max_seq_len every time.
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        if self.answer_only_loss:
            try:
                eq_pos = ids.index(self.eq_id)
            except ValueError:
                eq_pos = 0
            for j in range(len(y)):
                pred_index = j + 1
                if pred_index <= eq_pos:
                    y[j] = -100

        return {"input_ids": x, "labels": y}


def collate_batch(
    batch: list[dict[str, torch.Tensor]], pad_id: int = 0
) -> dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for item in batch:
        x, y = item["input_ids"], item["labels"]
        pad_len = max_len - x.numel()
        if pad_len:
            x = torch.nn.functional.pad(x, (0, pad_len), value=pad_id)
            y = torch.nn.functional.pad(y, (0, pad_len), value=-100)
        input_ids.append(x)
        labels.append(y)
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


class LengthBucketBatchSampler(Sampler[list[int]]):
    """Shuffle examples while grouping similarly sized sequences together."""

    def __init__(
        self,
        dataset: EquationDataset,
        batch_size: int,
        shuffle: bool = True,
        bucket_multiplier: int = 20,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)
        self.seed = seed
        self._iteration = 0

    def __iter__(self):
        import random

        indices = list(range(len(self.dataset)))
        rng = random.Random(self.seed + self._iteration)
        self._iteration += 1
        if self.shuffle:
            rng.shuffle(indices)

        batches: list[list[int]] = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start : start + self.bucket_size]
            bucket.sort(key=lambda index: len(self.dataset.examples[index]))
            batches.extend(
                bucket[pos : pos + self.batch_size]
                for pos in range(0, len(bucket), self.batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
