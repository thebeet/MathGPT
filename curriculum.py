"""Primary-school style curriculum (grades 1–6) with per-grade exams."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dataset import (
    ExpressionProblem,
    LinearEquationProblem,
    Problem,
    generate_expression_problems,
    generate_linear_equation_problems,
    generate_problems,
)


@dataclass
class CurriculumGrade:
    """One training stage: difficulty increases from grade 1 to 6."""

    id: int
    name: str
    pass_accuracy: float
    min_epochs: int
    max_epochs: int
    train_size: int
    max_number: int
    mul_max_operand: int
    ops: tuple[str, ...]
    min_digits: int | None = None
    max_digits: int | None = None
    negative_fraction: float = 0.0
    expression_fraction: float = 0.0
    expression_min_terms: int = 2
    expression_max_terms: int = 3
    expression_min_digits: int = 1
    expression_max_digits: int = 2
    expression_parentheses_fraction: float = 0.5
    scratchpad_ops: tuple[str, ...] = ("+", "-", "*")
    exam_size: int = 200
    exam_min_digits: int | None = None
    exam_max_digits: int | None = None
    description: str = ""
    equation_fraction: float = 0.0
    equation_max_abs: int = 20
    equation_max_coef: int = 9
    equation_err_fraction: float = 0.12
    equation_easy_max_abs: int | None = None
    equation_easy_max_coef: int | None = None
    equation_easy_err_fraction: float | None = None


@dataclass
class ExamRecord:
    grade_id: int
    accuracy: float
    passed: bool
    step: int
    n_problems: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumState:
    """Serialized in checkpoints so training can resume at the right grade."""

    passed_grades: list[int] = field(default_factory=list)
    current_grade: int = 1
    grade_epoch: int = 0
    exam_history: list[dict[str, Any]] = field(default_factory=list)
    last_exam: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed_grades": list(self.passed_grades),
            "current_grade": self.current_grade,
            "grade_epoch": self.grade_epoch,
            "exam_history": list(self.exam_history),
            "last_exam": self.last_exam,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CurriculumState:
        if not raw:
            return cls()
        return cls(
            passed_grades=list(raw.get("passed_grades") or []),
            current_grade=int(raw.get("current_grade") or 1),
            grade_epoch=int(raw.get("grade_epoch") or 0),
            exam_history=list(raw.get("exam_history") or []),
            last_exam=raw.get("last_exam"),
        )


def default_curriculum() -> list[CurriculumGrade]:
    """Six stages: merged G1–2 → … → G6 expressions + merged equations."""
    return [
        CurriculumGrade(
            id=1,
            name="一年级",
            description="一位数加法与两位数加减（原一、二年级合并）",
            pass_accuracy=0.98,
            min_epochs=1,
            max_epochs=60,
            train_size=120_000,
            max_number=99,
            mul_max_operand=99,
            ops=("+", "-"),
            min_digits=1,
            max_digits=2,
            negative_fraction=0.05,
            scratchpad_ops=("+", "-"),
            exam_size=200,
            exam_min_digits=1,
            exam_max_digits=2,
        ),
        CurriculumGrade(
            id=2,
            name="二年级",
            description="乘法表与加减巩固",
            pass_accuracy=0.90,
            min_epochs=1,
            max_epochs=60,
            train_size=150_000,
            max_number=99,
            mul_max_operand=99,
            ops=("+", "-", "*"),
            min_digits=1,
            max_digits=2,
            negative_fraction=0.1,
            scratchpad_ops=("+", "-", "*"),
            exam_size=200,
            exam_min_digits=1,
            exam_max_digits=2,
        ),
        CurriculumGrade(
            id=3,
            name="三年级",
            description="除法与三位数四则",
            pass_accuracy=0.90,
            min_epochs=1,
            max_epochs=70,
            train_size=180_000,
            max_number=999,
            mul_max_operand=999,
            ops=("+", "-", "*", "/"),
            min_digits=1,
            max_digits=3,
            negative_fraction=0.15,
            scratchpad_ops=("+", "-", "*"),
            exam_size=250,
            exam_min_digits=1,
            exam_max_digits=3,
        ),
        CurriculumGrade(
            id=4,
            name="四年级",
            description="多位数四则与简单混合式",
            pass_accuracy=0.90,
            min_epochs=1,
            max_epochs=90,
            train_size=220_000,
            max_number=9_999,
            mul_max_operand=9_999,
            ops=("+", "-", "*", "/"),
            min_digits=2,
            max_digits=4,
            negative_fraction=0.2,
            expression_fraction=0.35,
            expression_min_terms=2,
            expression_max_terms=4,
            expression_min_digits=1,
            expression_max_digits=3,
            expression_parentheses_fraction=0.6,
            scratchpad_ops=("+", "-", "*"),
            exam_size=250,
            exam_min_digits=2,
            exam_max_digits=4,
        ),
        CurriculumGrade(
            id=5,
            name="五年级",
            description="长数字与复杂混合表达式",
            pass_accuracy=0.90,
            min_epochs=1,
            max_epochs=120,
            train_size=300_000,
            max_number=9_999_999,
            mul_max_operand=9_999_999,
            ops=("+", "-", "*", "/"),
            min_digits=3,
            max_digits=7,
            negative_fraction=0.25,
            expression_fraction=0.75,
            expression_min_terms=4,
            expression_max_terms=10,
            expression_min_digits=3,
            expression_max_digits=7,
            expression_parentheses_fraction=0.75,
            scratchpad_ops=("+", "-", "*"),
            exam_size=300,
            exam_min_digits=3,
            exam_max_digits=7,
        ),
        CurriculumGrade(
            id=6,
            name="六年级",
            description="五年级混合表达式 + 一元方程（原六、七年级方程合并）",
            pass_accuracy=0.90,
            min_epochs=1,
            max_epochs=90,
            train_size=300_000,
            max_number=9_999_999,
            mul_max_operand=9_999_999,
            ops=("+", "-", "*", "/"),
            min_digits=3,
            max_digits=7,
            negative_fraction=0.25,
            expression_fraction=0.75,
            expression_min_terms=4,
            expression_max_terms=10,
            expression_min_digits=3,
            expression_max_digits=7,
            expression_parentheses_fraction=0.75,
            scratchpad_ops=("+", "-", "*"),
            exam_size=300,
            exam_min_digits=3,
            exam_max_digits=7,
            equation_fraction=0.25,
            equation_max_abs=99,
            equation_max_coef=12,
            equation_err_fraction=0.15,
            equation_easy_max_abs=20,
            equation_easy_max_coef=9,
            equation_easy_err_fraction=0.10,
        ),
    ]


def smoke_curriculum() -> list[CurriculumGrade]:
    """Tiny curriculum for pipeline checks."""
    base = default_curriculum()
    out: list[CurriculumGrade] = []
    for g in base:
        out.append(
            CurriculumGrade(
                **{
                    **asdict(g),
                    "pass_accuracy": 0.5,
                    "max_epochs": 3,
                    "train_size": 2_000,
                    "exam_size": 40,
                }
            )
        )
    return out


def get_grade(curriculum: list[CurriculumGrade], grade_id: int) -> CurriculumGrade:
    for g in curriculum:
        if g.id == grade_id:
            return g
    raise KeyError(f"Unknown grade id {grade_id}")


def _ops_to_flags(ops: tuple[str, ...]) -> dict[str, bool]:
    return {
        "include_addition": "+" in ops,
        "include_subtraction": "-" in ops,
        "include_multiplication": "*" in ops,
        "include_division": "/" in ops,
    }


def _equation_kwargs(grade: CurriculumGrade) -> dict:
    return dict(
        max_abs=grade.equation_max_abs,
        max_coef=grade.equation_max_coef,
        err_fraction=grade.equation_err_fraction,
    )


def _equation_easy_kwargs(grade: CurriculumGrade) -> dict:
    return dict(
        max_abs=grade.equation_easy_max_abs or grade.equation_max_abs,
        max_coef=grade.equation_easy_max_coef or grade.equation_max_coef,
        err_fraction=grade.equation_easy_err_fraction or grade.equation_err_fraction,
    )


def generate_linear_equations_for_grade(
    n: int,
    seed: int,
    grade: CurriculumGrade,
) -> list[LinearEquationProblem]:
    """One or two tiers of a*x+k=c when equation_easy_* is set on the grade."""
    if n <= 0:
        return []
    if grade.equation_easy_max_abs is not None:
        # Small-coefficient tier has limited unique strings; cap easy count.
        easy_cap = 12_000
        n_easy = min(n // 2, easy_cap)
        n_hard = n - n_easy
        out: list[LinearEquationProblem] = []
        if n_easy:
            out.extend(
                generate_linear_equation_problems(
                    n_easy,
                    seed=seed,
                    **_equation_easy_kwargs(grade),
                )
            )
        if n_hard:
            out.extend(
                generate_linear_equation_problems(
                    n_hard,
                    seed=seed + 991,
                    **_equation_kwargs(grade),
                )
            )
        return out
    return generate_linear_equation_problems(n, seed=seed, **_equation_kwargs(grade))


def generate_grade_exam_problems(
    grade: CurriculumGrade,
    seed: int,
) -> list[Problem | ExpressionProblem | LinearEquationProblem]:
    """Fixed-difficulty exam set for one grade (held out from training keys)."""
    if grade.equation_fraction >= 1.0 and grade.expression_fraction <= 0.0:
        return generate_linear_equations_for_grade(grade.exam_size, seed, grade)

    flags = _ops_to_flags(grade.ops)
    ops_filter = list(grade.ops)
    expr_n = int(grade.exam_size * grade.expression_fraction)
    eq_n = int(grade.exam_size * grade.equation_fraction)
    binary_n = max(0, grade.exam_size - expr_n - eq_n)
    problems: list[Problem | ExpressionProblem | LinearEquationProblem] = generate_problems(
        binary_n,
        grade.max_number,
        seed=seed,
        min_digits=grade.exam_min_digits,
        max_digits=grade.exam_max_digits,
        ops_filter=ops_filter,
        mul_max_operand=grade.mul_max_operand,
        negative_fraction=grade.negative_fraction,
        **flags,
    )
    if expr_n > 0:
        problems.extend(
            generate_expression_problems(
                expr_n,
                grade.max_number,
                seed=seed + 17,
                min_terms=grade.expression_min_terms,
                max_terms=grade.expression_max_terms,
                parentheses_fraction=grade.expression_parentheses_fraction,
                min_digits=grade.expression_min_digits,
                max_digits=grade.expression_max_digits,
            )
        )
    if eq_n > 0:
        problems.extend(
            generate_linear_equations_for_grade(eq_n, seed + 23, grade),
        )
    return problems


def generate_grade_train_problems(
    grade: CurriculumGrade,
    seed: int,
) -> list[Problem | ExpressionProblem | LinearEquationProblem]:
    if grade.equation_fraction >= 1.0 and grade.expression_fraction <= 0.0:
        return generate_linear_equations_for_grade(grade.train_size, seed, grade)

    flags = _ops_to_flags(grade.ops)
    ops_filter = list(grade.ops)
    expression_count = int(grade.train_size * grade.expression_fraction)
    equation_count = int(grade.train_size * grade.equation_fraction)
    binary_count = max(0, grade.train_size - expression_count - equation_count)
    train_probs: list[Problem | ExpressionProblem | LinearEquationProblem] = generate_problems(
        binary_count,
        grade.max_number,
        seed=seed,
        min_digits=grade.min_digits,
        max_digits=grade.max_digits,
        ops_filter=ops_filter,
        mul_max_operand=grade.mul_max_operand,
        negative_fraction=grade.negative_fraction,
        **flags,
    )
    if expression_count > 0:
        train_probs.extend(
            generate_expression_problems(
                expression_count,
                grade.max_number,
                seed=seed + 31,
                min_terms=grade.expression_min_terms,
                max_terms=grade.expression_max_terms,
                parentheses_fraction=grade.expression_parentheses_fraction,
                min_digits=grade.expression_min_digits,
                max_digits=grade.expression_max_digits,
            )
        )
    if equation_count > 0:
        train_probs.extend(
            generate_linear_equations_for_grade(equation_count, seed + 37, grade),
        )
    return train_probs


def starting_grade(state: CurriculumState, curriculum: list[CurriculumGrade]) -> int:
    """First grade that is not yet passed."""
    passed = set(state.passed_grades)
    for g in curriculum:
        if g.id not in passed:
            return g.id
    return curriculum[-1].id


def merge_curriculum_into_config(cfg: Any, grade: CurriculumGrade) -> None:
    """Apply grade limits onto the shared Config object for logging / generation."""
    flags = _ops_to_flags(grade.ops)
    cfg.include_addition = flags["include_addition"]
    cfg.include_subtraction = flags["include_subtraction"]
    cfg.include_multiplication = flags["include_multiplication"]
    cfg.include_division = flags["include_division"]
    cfg.max_number = grade.max_number
    cfg.mul_max_operand = grade.mul_max_operand
    cfg.train_size = grade.train_size
    cfg.expression_fraction = grade.expression_fraction
    cfg.expression_min_terms = grade.expression_min_terms
    cfg.expression_max_terms = grade.expression_max_terms
    cfg.expression_train_min_digits = grade.expression_min_digits
    cfg.expression_train_max_digits = grade.expression_max_digits
    cfg.expression_parentheses_fraction = grade.expression_parentheses_fraction
    cfg.negative_fraction = grade.negative_fraction
    cfg.scratchpad_ops = grade.scratchpad_ops
    cfg.equation_fraction = grade.equation_fraction
