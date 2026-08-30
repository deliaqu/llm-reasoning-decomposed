import pandas as pd
import argparse
import pandas as pd
from fractions import Fraction
from sympy import simplify, sympify, nsimplify, SympifyError,  Eq
from typing import Union
import re
from typing import Union
from tqdm import tqdm
tqdm.pandas()

import os
import csv

from paths import DATA_DIR, RESULT_DIR


GROUND_TRUTH_COLUMNS = (
    "answer",
    "numerical_abstraction_answer",
    "symbolic_abstraction_answer",
    "symbol_binding",
)


def canonical_dataset_path(dataset_name: str, set_id):
    """Return the canonical dataset CSV path for (dataset_name, set_id).

    Mirrors the resolution in run_inference_transformers_direct.resolve_data_path
    so we can re-load fresh ground-truth columns at evaluation time, instead of
    trusting the snapshot copied into the inference result CSV.
    """
    if set_id is not None:
        per_set = {
            "gsm_p1": ("split_datasets_p1", "test_gsm_p1"),
            "gsm_p1_from_symbolic": ("split_datasets_p1_from_symbolic", "test_gsm_p1_from_symbolic"),
            "gsm_p1_from_symbolic_strict_valid": ("split_datasets_p1_from_symbolic_strict_valid", "test_gsm_p1_from_symbolic_strict_valid"),
            "gsm_p2": ("split_datasets_p2", "test_gsm_p2"),
            "gsm_noop": ("split_datasets_noop", "test_gsm_noop"),
            "gsm_noop_clean": ("clean_noop/split_datasets_noop", "test_gsm_noop"),
            "gsm_filler": ("split_datasets_filler", "test_gsm_filler"),
            "gsm_symbolic_padded_to_p1_len_delta_abs_lt2": ("split_datasets_symbolic_padded_to_p1_len_delta_abs_lt2", "test_gsm_symbolic_padded_to_p1"),
            "gsm_symbolic_padded_to_p1_len_delta_0": ("split_datasets_symbolic_padded_to_p1_len_delta_0", "test_gsm_symbolic_padded_to_p1"),
            "gsm_symbolic_to_p1": ("split_datasets_symbolic_to_p1", "test_gsm_symbolic_to_p1"),
            "gsm_symbolic": ("split_datasets", "test_gsm_symbolic"),
        }
        if dataset_name not in per_set:
            return None
        split_dir, file_stem = per_set[dataset_name]
        return f"{DATA_DIR}/gsm_symbolic/{split_dir}/{file_stem}_set_{set_id}.csv"

    if dataset_name == "gsm_noop_clean":
        return f"{DATA_DIR}/gsm_symbolic/clean_noop/test_gsm_noop.csv"
    if dataset_name in {
        "gsm_p1", "gsm_p2", "gsm_symbolic", "gsm_noop", "gsm_filler",
        "gsm_symbolic_padded_to_p1_len_delta_abs_lt2",
        "gsm_symbolic_padded_to_p1_len_delta_0",
        "gsm_symbolic_to_p1",
        "gsm_p1_from_symbolic",
        "gsm_p1_from_symbolic_strict_valid",
    }:
        return f"{DATA_DIR}/gsm_symbolic/test_{dataset_name}.csv"
    return f"{DATA_DIR}/test_{dataset_name}.csv"


def refresh_ground_truth(df: pd.DataFrame, dataset_path: str) -> pd.DataFrame:
    """Overwrite ground-truth columns in `df` with the values currently in
    `dataset_path`, joining on (original_id, instance). Reports per-column
    deltas. Returns the df with refreshed columns.
    """
    if dataset_path is None or not os.path.exists(dataset_path):
        print(f"Canonical dataset not found ({dataset_path}); skipping ground-truth refresh.")
        return df
    if not {"original_id", "instance"}.issubset(df.columns):
        print("Result CSV lacks (original_id, instance); skipping ground-truth refresh.")
        return df
    canonical = pd.read_csv(dataset_path)
    keep = ["original_id", "instance"] + [c for c in GROUND_TRUTH_COLUMNS if c in canonical.columns]
    canonical = canonical[keep].drop_duplicates(["original_id", "instance"])
    merged = df.merge(canonical, on=["original_id", "instance"], how="left", suffixes=("", "__canonical"))
    deltas = {}
    for col in GROUND_TRUTH_COLUMNS:
        canon_col = f"{col}__canonical"
        if canon_col not in merged.columns:
            continue
        if col in merged.columns:
            old = merged[col].astype(str)
            new = merged[canon_col].astype(str)
            mask = (old != new) & merged[canon_col].notna()
            deltas[col] = int(mask.sum())
            merged.loc[mask, col] = merged.loc[mask, canon_col]
        else:
            merged[col] = merged[canon_col]
            deltas[col] = int(merged[canon_col].notna().sum())
        merged = merged.drop(columns=[canon_col])
    if any(deltas.values()):
        deltas_str = ", ".join(f"{c}: {n}" for c, n in deltas.items() if n)
        print(f"Refreshed ground-truth from {dataset_path} (rows changed — {deltas_str})")
    else:
        print(f"Ground-truth in result CSV already matches {dataset_path}.")
    return merged
from evaluation_utils import (
    normalize,
    extract_last_number,
    extract_lhs,
    clean_expression,
    solve_for_x,
    latex_to_python_math
)
from instructions import TASK_CONFIG

def postprocess_generation(generation, verbose=False):
    '''
        Parse the content after '####'.
        Always takes the text after the LAST '####' as the answer, because model CoT may
        quote '####' (from the instruction) before using it as the actual answer delimiter.
    '''
    parts = generation.split('####')
    if len(parts) < 2:
        if verbose:
            print("Instruction Following Error, did not write in the format of ####.")
        # Fallback: extract last \boxed{} content (e.g. Qwen-Math style output)
        boxed = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', generation)
        if boxed:
            return boxed[-1].strip()
        return generation

    # Take the text after the last '####'
    response = parts[-1].strip()
    if not response:
        # Model wrote answer before '####' (e.g., "answer ####") — take preceding part
        response = parts[-2].strip()
    return response


def evaluate_final_answer(generation, reference, verbose=False):
    """
    Extracts the final numeric answer from the generation and compares it to the reference
    after normalization. Lenient with rounding: also accepts if floor(model_value) matches
    the reference, since ground truth answers are derived from expressions containing floor().
    Returns pd.NA when extraction/normalization fails.
    """
    import math
    generation = str(generation)
    reference = str(reference)
    
    # Some datasets leave "#### " in the reference answer.
    if "####" in reference:
        extr = extract_last_number(reference)
        if extr is not None:
            reference = extr

    try:
        number = extract_last_number(generation)
    except Exception as e:
        if verbose:
            print(e)
        return False  # instruction following error
    else:
        try:
            if normalize(number) == normalize(reference):
                return True
            # Lenient: accept if floor(model_value) == reference (model computed the exact
            # fractional value but skipped the final integer rounding step).
            try:
                if math.floor(float(normalize(number))) == float(normalize(reference)):
                    return True
            except Exception:
                pass
            return False
        except Exception as e:
            if verbose:
                print(e, generation, reference)
            return pd.NA
    

def evaluate_final_answer_text(generation_answer, reference, verbose=False):
    """
    Text-answer counterpart of evaluate_final_answer for datasets whose gold
    answers are strings (e.g. phantomwiki: names, dates, counts). Compares the
    already-extracted '####' answer to the reference after normalization
    (whitespace collapse, casefold, trailing-punctuation strip); falls back to
    numeric equality when both sides parse as numbers ("2" == "2.0").
    """
    def _norm_text(s):
        s = re.sub(r"\s+", " ", str(s)).strip()
        s = s.strip("\"'").rstrip(".")
        return s.casefold()

    gen, ref = _norm_text(generation_answer), _norm_text(reference)
    if not gen:
        return False
    if gen == ref:
        return True
    try:
        return float(gen) == float(ref)
    except (TypeError, ValueError):
        return False


_EXPR_MAX_LEN = 500  # chars; garbage model outputs can be 100k+ chars


def _extract_sympy_expr(raw: str):
    """
    Convert a raw expression (LaTeX, natural language, or plain math) to a sympy expression.
    Returns None if parsing fails.
    """
    expr = str(raw).strip()

    # Bail early on garbage — real math expressions are short.
    if len(expr) > _EXPR_MAX_LEN:
        return None

    # Normalize Unicode math characters to ASCII equivalents before anything else.
    expr = expr.replace('\u2212', '-')   # Unicode minus sign '−' → '-'
    expr = expr.replace('\u00d7', '*')   # Unicode multiplication '×' → '*' (also caught below)
    expr = expr.replace('\u00f7', '/')   # Unicode division '÷' → '/'
    expr = expr.replace('\u2019', '')    # Right single quotation mark (noise)

    # Unicode floor/ceiling brackets: ⌊x⌋ → floor(x), ⌈x⌉ → ceiling(x)
    expr = expr.replace('\u2308', 'ceiling(').replace('\u2309', ')')  # ⌈ ⌉
    expr = expr.replace('\u230a', 'floor(').replace('\u230b', ')')    # ⌊ ⌋

    # Remove LaTeX display/inline delimiters BEFORE multiline split so trailing \] lines
    # don't get chosen as the "last line" of a multiline answer.
    expr = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', expr)
    expr = expr.replace('$', '')

    # Multi-line: the model may write the expression first then explain, or explain then give the
    # expression last. Try lines from last to first, pick the first one that contains at least
    # one math character (digit, operator, or paren) — this skips pure natural-language lines.
    if '\n' in expr:
        lines = [line.strip() for line in expr.split('\n') if line.strip()]
        if not lines:
            return None
        math_lines = [l for l in lines if re.search(r'[\d+\-*/()^=]', l)]
        expr = math_lines[-1] if math_lines else lines[-1]

    # Unwrap \boxed{...}
    expr = re.sub(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', r'\1', expr)

    # Convert square brackets to standard parentheses
    expr = expr.replace('[', '(').replace(']', ')')

    # Remove \left, \right BEFORE \lceil/\rceil so "\left\lceil" doesn't become "\leftceiling("
    expr = re.sub(r'\\(?:left|right)\b', '', expr)

    # LaTeX ceiling/floor bracket notation: \lceil X \rceil -> ceiling(X), \lfloor X \rfloor -> floor(X)
    expr = re.sub(r'\\lceil\b', 'ceiling(', expr)
    expr = re.sub(r'\\rceil\b', ')', expr)
    expr = re.sub(r'\\lfloor\b', 'floor(', expr)
    expr = re.sub(r'\\rfloor\b', ')', expr)

    # Strip backslash from LaTeX named functions: \min -> min, \max -> max, etc.
    # These are then caught and capitalized by the _FUNC_MAP protection step below.
    expr = re.sub(r'\\(min|max|gcd|lcm|log|exp|sin|cos|tan|sqrt|abs)\b', lambda m: m.group(1), expr)

    # LaTeX operators
    expr = expr.replace('\\times', '*').replace('×', '*')
    expr = expr.replace('\\cdot', '*').replace('·', '*')
    expr = expr.replace('\\div', '/').replace('÷', '/')

    # LaTeX fractions: \frac{a}{b} -> ((a)/(b))
    # Must be before ^{...} conversion so that ^{\frac{u}{z}} → ^{((u)/(z))} → **((u)/(z)).
    # Apply iteratively with [^{}]+ (no nested braces) to handle nested \frac correctly.
    while True:
        new_expr = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'((\1)/(\2))', expr)
        if new_expr == expr:
            break
        expr = new_expr

    # LaTeX exponent with braces: x^{expr} -> x**(expr)  (after \frac so inner fracs are plain)
    while True:
        new_expr = re.sub(r'\^\{([^{}]+)\}', r'**(\1)', expr)
        if new_expr == expr:
            break
        expr = new_expr

    # Caret exponentiation: x^2 -> x**2
    expr = expr.replace('^', '**')

    # Percent sign after a variable or number: x% -> (x/100), 20% -> (20/100)
    expr = re.sub(r'([a-zA-Z])\s*%', r'(\1/100)', expr)
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)

    # Absolute value bars: |expr| -> strip bars
    expr = expr.replace('|', '')

    # Handle equation form "M = 2(y+x)" -> take RHS when LHS is a single identifier
    if '=' in expr:
        parts = expr.split('=', 1)
        lhs = parts[0].strip()
        if re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', lhs):
            expr = parts[1].strip()

    # Protect known math function calls from multi-letter expansion.
    # e.g., "min(" -> placeholder "__0__(" so "m","i","n" don't get split into "m*i*n"
    # The placeholder __N__ uses only underscores+digit so it's invisible to letter-pair regex.
    _FUNC_MAP = {
        'min': 'Min', 'max': 'Max',
        'floor': 'floor', 'ceil': 'ceiling', 'ceiling': 'ceiling',
        'abs': 'Abs', 'sqrt': 'sqrt', 'log': 'log', 'exp': 'exp',
        'sin': 'sin', 'cos': 'cos', 'tan': 'tan',
        'asin': 'asin', 'acos': 'acos', 'atan': 'atan',
        'gcd': 'gcd', 'lcm': 'lcm',
    }
    _stored_funcs = {}
    # Match only known function names (not arbitrary identifiers like x, y, M)
    _known_func_pattern = r'\b(' + '|'.join(sorted(_FUNC_MAP.keys(), key=len, reverse=True)) + r')\b\s*(?=\()'

    def _protect_func(m):
        name = m.group(1).lower()
        idx = len(_stored_funcs)
        key = f'__{idx}__'
        sympy_name = _FUNC_MAP.get(name, name)
        _stored_funcs[key] = sympy_name  # store the sympy name; original '(' stays in expression
        return key  # replace word with placeholder; lookahead keeps '(' intact

    expr = re.sub(_known_func_pattern, _protect_func, expr, flags=re.IGNORECASE)

    # Multi-letter variable products: "xz" -> "x*z" (applied iteratively for 3+ letter sequences like "uxy" -> "u*x*y")
    # Must run BEFORE math-flag extraction so tokens like "yz", "vz" get "*" inserted
    # and are then recognized as math by the extraction step below.
    # Function names are already replaced by placeholders so they won't be split.
    while True:
        new_expr = re.sub(r'([a-zA-Z])([a-zA-Z])', r'\1*\2', expr)
        if new_expr == expr:
            break
        expr = new_expr

    # Extract math from natural language: find the largest contiguous block of
    # math-like tokens (contain a digit, are a single letter, or are pure operators)
    tokens = expr.split()
    math_flags = [
        bool(re.search(r'[\d+\-*/()^.=]', t)) or  # contains any math character
        (len(t) == 1 and t.isalpha()) or          # single-letter variable
        bool(re.match(r'^[a-zA-Z]\d+$', t))       # variables like x1, y2
        for t in tokens
    ]
    if not all(math_flags):
        best, current = [], []
        for token, is_math in zip(tokens, math_flags):
            if is_math:
                current.append(token)
            else:
                if len(current) > len(best):
                    best = current[:]
                current = []
        if len(current) > len(best):
            best = current
        expr = ' '.join(best) if best else expr

    # If the expression contains a top-level comma (not inside parentheses), take the last segment.
    # Models sometimes output "expr1, expr2" with the final answer last.
    depth = 0
    last_top_comma = -1
    for ci, ch in enumerate(expr):
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        elif ch == ',' and depth == 0:
            last_top_comma = ci
    if last_top_comma >= 0:
        expr = expr[last_top_comma + 1:].strip()

    # Implicit multiplication (placeholders keep function names safe)
    expr = re.sub(r'(\d)\s*([a-zA-Z])', r'\1*\2', expr)
    expr = re.sub(r'(\d)\s*\(', r'\1*(', expr)
    expr = re.sub(r'\)\s*\(', r')*(', expr)
    expr = re.sub(r'\)\s*([a-zA-Z])', r')*\1', expr)
    expr = re.sub(r'([a-zA-Z])\s*\(', r'\1*(', expr)
    # Letter/digit space letter → implicit multiplication (e.g., "z u" → "z*u", "(x*y)z u" → handled)
    expr = re.sub(r'([a-zA-Z\d])\s+([a-zA-Z])', r'\1*\2', expr)

    # Restore protected function names: __N__(  →  sympy_name(
    for key, val in _stored_funcs.items():
        expr = expr.replace(key + '(', val + '(')

    try:
        # Override sympy's reserved single-letter names (S, I, E, N, Q, etc.) with plain Symbols
        # so that model outputs using them as variables don't silently produce wrong results.
        from sympy import symbols as _syms
        _safe_locals = {c: _syms(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'}
        try:
            return sympify(expr, locals=_safe_locals)
        except Exception:
            # Fallback: auto-balance mismatched parentheses (common in dataset/model outputs)
            open_count = expr.count('(')
            close_count = expr.count(')')
            balanced = expr
            if open_count > close_count:
                balanced = expr + ')' * (open_count - close_count)
            elif close_count > open_count:
                balanced = '(' * (close_count - open_count) + expr
            if balanced != expr:
                try:
                    return sympify(balanced, locals=_safe_locals)
                except Exception:
                    pass
            return None
    except Exception:
        return None


def _strip_rounding(e):
    """Strip floor() and ceiling() from a sympy expression, replacing each with its argument."""
    from sympy import floor as _floor, ceiling as _ceiling
    return e.replace(_floor, lambda x: x).replace(_ceiling, lambda x: x)


def _multi_binding_numeric_check(expr1, expr2, n_trials: int = 15) -> Union[bool, "pd.NAType"]:
    """
    Try multiple random integer bindings for all free symbols in expr1/expr2.
    Returns True if all successful trials agree (≥3 required), False if any trial disagrees,
    pd.NA if too few trials succeed (expressions may not be evaluable at random points).
    Used as a fallback when sympy's algebraic simplify() cannot handle an expression
    (e.g., floor(), ceiling(), piecewise, conditional operators).
    """
    import random
    all_vars = list(expr1.free_symbols | expr2.free_symbols)
    if not all_vars:
        return pd.NA
    matches = 0
    for _ in range(n_trials):
        # Use non-integer floats so floor(x) != x, distinguishing expressions that incorrectly
        # drop floor()/ceiling(). Values in (2, 19) avoid edge cases near 0.
        bindings = {v: random.uniform(2.1, 18.9) for v in all_vars}
        try:
            v1 = float(expr1.subs(bindings).evalf())
            v2 = float(expr2.subs(bindings).evalf())
            if round(v1, 4) != round(v2, 4):
                return False
            matches += 1
        except Exception:
            continue
    if matches >= 3:
        return True
    return pd.NA


def evaluate_symbolic_abstraction(symbolic_answer, abstract_answer, instance_id=None, symbol_binding=None, numerical_answer=None):
    """
    Deterministically checks whether two symbolic expressions are equivalent using sympy.
    Handles LaTeX notation, natural language, and implicit multiplication.
    Also validates via numeric substitution into original dataset bindings.
    Returns True, False, or pd.NA if parsing fails.
    """
    id_tag = f" [id={instance_id}]" if instance_id is not None else ""
    try:
        # Model outputs with comparison operators (>, <) indicate conditional/boolean logic.
        # GT expressions never use these operators, so such model answers are definitively wrong.
        if re.search(r'[><]', str(abstract_answer)):
            return False

        expr1 = _extract_sympy_expr(str(symbolic_answer))
        expr2 = _extract_sympy_expr(str(abstract_answer))
        
        if expr1 is None or expr2 is None:
            return pd.NA

        # Strip floor()/ceiling() from both expressions upfront. Ground truth uses floor() to
        # represent int() truncation from the original templates.
        expr1 = _strip_rounding(expr1)
        expr2 = _strip_rounding(expr2)

        # Check equivalence through numeric substitution first (extremely fast and solves variable restructuring)
        if pd.notna(symbol_binding) and pd.notna(numerical_answer):
            try:
                bindings = {
                    sympify(k.strip()): sympify(v.strip()) 
                    for pair in str(symbol_binding).split(',') if '=' in pair 
                    for k, v in [pair.split('=')]
                }
                
                if expr1.free_symbols:
                    v1 = float(expr1.subs(bindings).evalf())
                    v2 = float(expr2.subs(bindings).evalf())
                    if round(v1, 4) != round(v2, 4):
                        return False  # Numerically different → definitively wrong
            except Exception:
                pass

        # Try symbolic equivalence check (guarded: simplify() can hang forever on
        # pathological expressions produced by degenerate model outputs).
        import signal

        class _Timeout(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise _Timeout()

        _SIMPLIFY_TIMEOUT_SEC = 10

        def _safe_simplify(e):
            old = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(_SIMPLIFY_TIMEOUT_SEC)
            try:
                return simplify(e)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)

        try:
            if _safe_simplify(expr1 - expr2) == 0:
                return True

            # Check for common percentage and fractional mismatches (e.g. y vs y/100, y vs 1/y)
            symbols = expr1.free_symbols | expr2.free_symbols
            for s in symbols:
                for transform in [s/100, 100*s, 1/s]:
                    if _safe_simplify(expr1.subs(s, transform) - expr2) == 0 or _safe_simplify(expr1 - expr2.subs(s, transform)) == 0:
                        return True

            print(f"Symbolic expressions not equivalent{id_tag}: '{symbolic_answer}' vs '{abstract_answer}'")
            return False

        except _Timeout:
            print(f"Symbolic simplify timed out{id_tag}: '{str(symbolic_answer)[:80]}' vs '{str(abstract_answer)[:80]}'")
            result = _multi_binding_numeric_check(expr1, expr2)
            if result is not pd.NA:
                return result
        except Exception:
            # simplify() threw (e.g. due to Min/Max) — try numeric fallback
            result = _multi_binding_numeric_check(expr1, expr2)
            if result is not pd.NA:
                return result

    except Exception:
        pass
        
    print(f"Error comparing symbolic expressions{id_tag}: '{symbolic_answer}' vs '{abstract_answer}'")
    return pd.NA


def evaluate_numerical_abstraction(expr: str, reference_expr: Union[str, int, float], verbose: bool = False, strict: bool = True) -> Union[bool, pd.NA]:
    """
    Evaluates whether the generated expression is equivalent to the reference expression.
    When strict=True (default), two conditions must both hold:
      1. The expression evaluates to the same numeric value as the reference.
      2. Every number used in the model expression is also present in the reference
         expression (as a set — rejects pre-computed intermediates like 9*2 for
         ref (16-3-4)*2, but allows reorderings like 2*(16-3-4)).
    When strict=False, only condition 1 is checked (value equality only).
    Returns `pd.NA` if parsing or evaluation fails.
    """
    expr = str(expr)
    reference_expr = str(reference_expr)
    try:
        # Model outputs with comparison operators indicate conditional/boolean logic rather than
        # a pure arithmetic expression — these are definitively wrong for this task.
        if re.search(r'[><]', expr):
            return False

        # Model outputs where numbers are separated by spaces without any operator between them
        # (e.g., "(150  80000  50000)" instead of "(150/100)*80000") are unparseable and wrong.
        if re.search(r'\d\s+\d', expr):
            return False

        # Strip LaTeX display delimiters (\[ \] \( \)) that may appear as trailing lines
        # after postprocess_generation, so they don't trigger latex_to_python_math unnecessarily.
        expr = re.sub(r'\\\(|\\\)|\\\[|\\\]', '', expr).strip()

        # Multiline model prediction: take the first line that contains math characters.
        # CoT models sometimes append explanation after the expression (e.g. "expr\nThe answer is...")
        if '\n' in expr:
            lines = [l.strip() for l in expr.split('\n') if l.strip()]
            math_lines = [l for l in lines if re.search(r'[\d+\-*/()=.]', l)]
            expr = math_lines[0] if math_lines else lines[0]

        # Convert LaTeX to plain math
        if "\\" in expr:
            try:
                expr = latex_to_python_math(expr)
            except:
                return pd.NA

        if 'x' in expr and '=' in expr:
            return solve_for_x(expr, reference_expr, verbose)

        expr_lhs = extract_lhs(expr)
        expr_clean = clean_expression(expr_lhs)
        ref_clean = clean_expression(reference_expr)

        if verbose:
            print(f"Cleaned expression: {expr_clean}")
            print(f"Cleaned reference:  {ref_clean}")

        try:
            e1 = _strip_rounding(sympify(expr_clean, evaluate=False))
            e2 = _strip_rounding(sympify(ref_clean, evaluate=False))

            # Check 1: values must match
            # Use numeric tolerance to handle float-vs-Rational mismatches
            # (e.g., model writes 0.25 instead of 25/100 — sympy returns ~1e-14 noise)
            diff = simplify(e1 - e2)
            if diff != 0:
                try:
                    if abs(float(diff.evalf())) > 1e-6:
                        if verbose:
                            print("Values do not match.")
                        return False
                except Exception:
                    if verbose:
                        print("Values do not match.")
                    return False

            if strict:
                # Check 2: reject bare final answer
                if re.fullmatch(r"^-?\d+(\.\d+)?$", expr_clean.strip()):
                    if verbose:
                        print("Model output is a bare number.")
                    return False

                # Check 3: model numbers must be a subset of reference numbers.
                # Also include decimal equivalents of fractional rates in the reference
                # (e.g. 40/100 → 0.4) so a model writing 0.4 is not penalized.
                # Only non-integer results are included: integer-valued fractions (e.g. 12/4=3)
                # represent pre-computed division and must NOT be exempted.
                ref_nums = {float(m) for m in re.findall(r'\d+\.?\d*', ref_clean)}
                # Forward: non-integer a/b fractions in ref → add decimal equivalent
                # (e.g. 40/100 → 0.4, so model writing 0.4 is not penalized)
                for num, den in re.findall(r'(\d+\.?\d*)/(\d+\.?\d*)', ref_clean):
                    d = float(den)
                    if d != 0:
                        val = float(num) / d
                        if val != int(val):
                            ref_nums.add(val)
                # Backward: non-integer decimals in ref → add numerator/denominator of simplest fraction
                # (e.g. 0.5 → 1/2, so model writing x/2 instead of 0.5*x is not penalized)
                for m in re.findall(r'\d+\.\d+', ref_clean):
                    val = float(m)
                    if val != int(val):
                        frac = Fraction(val).limit_denominator(100)
                        ref_nums.add(float(frac.numerator))
                        ref_nums.add(float(frac.denominator))
                ref_nums.add(1.0)  # 1 is the multiplicative identity; allow factored forms like a*(1-r)
                # Percentage rates: when a round base (10/100/1000) appears in the expanded ref set,
                # allow x/base and its complement (base-x)/base for any x also in the expanded set.
                # This lets the model write 1-0.25 instead of (100-25)/100, without penalty.
                for base in [10.0, 100.0, 1000.0]:
                    if base in ref_nums:
                        for x in list(ref_nums):
                            val = x / base
                            if val > 0 and val != int(val):
                                ref_nums.add(val)
                model_nums = {float(m) for m in re.findall(r'\d+\.?\d*', expr_clean)}
                if not model_nums.issubset(ref_nums):
                    if verbose:
                        print(f"Model used numbers not in reference: {model_nums - ref_nums}")
                    return False

            return True

        except Exception as e:
            if verbose:
                print("Unexpected error...", e, expr_clean, ref_clean)
            return pd.NA

    except (SympifyError, SyntaxError, TypeError, ValueError, AttributeError) as e:
        if verbose:
            print(f"Error evaluating expression '<{reference_expr}>': {e}")
        return pd.NA


def save_metrics(acc_dict: dict, model_short_name: str, save_path: str = "evaluation_summary.csv") -> pd.DataFrame:
    """
    Save accuracies into a table:
      - index: model_short_name
      - columns: acc metrics (keys of acc_dict)
    If model exists, overwrite its row; otherwise append.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    from filelock import FileLock
    lock_path = save_path + ".lock"

    with FileLock(lock_path):
        # new row
        new_row = pd.DataFrame([acc_dict], index=[model_short_name])
        new_row.index.name = "model"

        # load existing or create new
        if os.path.exists(save_path):
            try:
                df = pd.read_csv(save_path, index_col=0)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()

        # ensure columns exist (union)
        df = df.reindex(columns=sorted(set(df.columns) | set(new_row.columns)))

        # write/overwrite row
        df.loc[model_short_name, new_row.columns] = new_row.iloc[0]

        # save
        df.to_csv(save_path)
    return df


if __name__ == "__main__":
        gsm_symbolic_padded_lt2 = "gsm_symbolic_padded_to_p1_len_delta_abs_lt2"

        parser = argparse.ArgumentParser(description="Run model evaluation with configurable parameters.")
        
        parser.add_argument("--dataset_name", type=str, default="gsm8k",
                           choices=[
                               "gsm8k",
                               "svamp",
                               "gsm_symbolic",
                               "gsm_symbolic_padded_to_p1_len_delta_abs_lt2",
                               "gsm_symbolic_to_p1",
                               "gsm_p1",
                               "gsm_p2",
                               "gsm_noop",
                               "gsm_noop_clean",
                               "gsm_filler",
                           ], help="Dataset to evaluate on")
        parser.add_argument(
            "--model_name",
            type=str,
            choices=[
                "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                "Qwen/Qwen2.5-Math-72B-Instruct",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.3-70B-Instruct",
            ],
            required=True,
            help="Choose the HuggingFace model id."
        )
        parser.add_argument(
            "--evaluation_summary_fname",
            type=str,
            default="evaluation_summary.csv",
            help="evaluation summary filename"
        )
        parser.add_argument(
            "--set_id",
            type=int,
            default=None,
            help="If set, evaluate results for data/gsm_symbolic/split_datasets/test_gsm_symbolic_set_{set_id}.csv"
        )
        parser.add_argument(
            "--tasks",
            nargs="+",
            choices=["original", "symbolic_abstraction", "numerical_abstraction", "arithmetic_computation"],
            default=None,
            help="Which tasks to evaluate. Evaluates all by default."
        )
        parser.add_argument(
            "--eight_shot",
            action="store_true",
            help="Enable 8-shot evaluation handling."
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=None,
            help="If set, evaluate results from a layer-truncated run (first N layers). Must match --num_layers used in run_inference.py."
        )

        # Parse arguments
        args = parser.parse_args()
        dataset = args.dataset_name
        model = args.model_name.split("/")[-1]
        if args.num_layers is not None:
            model += f"_layer{args.num_layers}"

        if args.set_id is not None:
            if args.dataset_name == "gsm_p1":
                dataset_tag = f"gsm_p1_set_{args.set_id}"
            elif args.dataset_name == "gsm_p2":
                dataset_tag = f"gsm_p2_set_{args.set_id}"
            elif args.dataset_name == "gsm_noop":
                dataset_tag = f"gsm_noop_set_{args.set_id}"
            elif args.dataset_name == "gsm_noop_clean":
                dataset_tag = f"gsm_noop_clean_set_{args.set_id}"
            elif args.dataset_name == "gsm_filler":
                dataset_tag = f"gsm_filler_set_{args.set_id}"
            elif args.dataset_name == gsm_symbolic_padded_lt2:
                dataset_tag = f"{gsm_symbolic_padded_lt2}_set_{args.set_id}"
            elif args.dataset_name == "gsm_symbolic_to_p1":
                dataset_tag = f"gsm_symbolic_to_p1_set_{args.set_id}"
            else:
                dataset_tag = f"gsm_symbolic_set_{args.set_id}"
            evaluation_summary_fname = f"{args.dataset_name}_split_{model}_{args.evaluation_summary_fname}"
            model_key = f"{model}_set_{args.set_id}"
        else:
            dataset_tag = dataset
            evaluation_summary_fname = f"{dataset}_{model}_{args.evaluation_summary_fname}"
            model_key = model

        target_dir = os.path.join(RESULT_DIR, "8shot") if getattr(args, "eight_shot", False) else RESULT_DIR
        if dataset_tag.startswith("gsm_symbolic"):
            target_dir = os.path.join(target_dir, "gsm_symbolic")
        elif dataset_tag.startswith("gsm_p1"):
            target_dir = os.path.join(target_dir, "gsm_p1")
        elif dataset_tag.startswith("gsm_p2"):
            target_dir = os.path.join(target_dir, "gsm_p2")
        elif dataset_tag.startswith("gsm_noop_clean"):
            target_dir = os.path.join(target_dir, "gsm_noop_clean")
        elif dataset_tag.startswith("gsm_noop"):
            target_dir = os.path.join(target_dir, "gsm_noop")
        elif dataset_tag.startswith("gsm_filler"):
            target_dir = os.path.join(target_dir, "gsm_filler")
            
        out_path = f"{target_dir}/{dataset_tag}_results_{model}.csv"

        print(f"=====================================================================")
        print(f"Starting evaluation job...")
        print(f"Model  : {model}")
        print(f"Dataset: {dataset_tag}")
        print(f"=====================================================================")

        # read inference response df
        if not os.path.exists(out_path):
            print(f"Inference file not found, skipping: {out_path}")
            exit(0)
        df = pd.read_csv(out_path)

        print(f"Dataset size: {len(df)}")

        # Refresh ground-truth columns from the canonical dataset on disk so
        # label fixes propagate without re-running inference.
        df = refresh_ground_truth(df, canonical_dataset_path(args.dataset_name, args.set_id))
        
        # ------------------------------------------------------------------------------------------------------
        # Post processing model generation and extract final answers
        print("Stage 1/2: Post-processing model generation to extract final answers...")
        answer_column = "answer"

        tasks_to_eval = args.tasks if args.tasks is not None else TASK_CONFIG.keys()

        for task in tasks_to_eval:
            for instr in ["direct", "cot"]:
                gen_col = f"{task}_{instr}"
                ans_col = f"{task}_{instr}_answer"
                if gen_col not in df.columns:
                    continue
                df[ans_col] = df[gen_col].apply(postprocess_generation)
        df.to_csv(out_path, index=False)

        
        # ------------------------------------------------------------------------------------------------------
        # Evaluation
        print("Stage 2/2: Evaluating extracted answers against references...")
        EVAL_SPECS = {
            "original": {
                "answer_col": "answer",
                "fn": lambda ref, pred, **_: evaluate_final_answer(pred, ref, verbose=False),
            },
            "numerical_abstraction": {
                "answer_col": "numerical_abstraction_answer",
                "fn": lambda ref, pred, **_: evaluate_numerical_abstraction(pred, ref, verbose=False, strict=True),
            },
            "numerical_abstraction_value": {
                "answer_col": "numerical_abstraction_answer",
                "inference_task": "numerical_abstraction",  # reuse the same inference columns
                "fn": lambda ref, pred, **_: evaluate_numerical_abstraction(pred, ref, verbose=False, strict=False),
            },
            "arithmetic_computation": {
                "answer_col": "answer",
                "fn": lambda ref, pred, **_: evaluate_final_answer(pred, ref, verbose=False),
            },
            "symbolic_abstraction": {
                "answer_col": "symbolic_abstraction_answer",
                "fn": lambda ref, pred, r=None, bind=None, ans=None, **_: evaluate_symbolic_abstraction(ref, pred, instance_id=r, symbol_binding=bind, numerical_answer=ans),
            },
        }
        for task, spec in EVAL_SPECS.items():
            # numerical_abstraction_value runs whenever numerical_abstraction does
            effective_task = spec.get("inference_task", task)
            if effective_task not in tasks_to_eval and task not in tasks_to_eval:
                continue
            ref_col = spec["answer_col"]

            for instr in ["direct", "cot"]:
                pred_col = f"{effective_task}_{instr}_answer"
                out_col = f"{task}_{instr}_correctness"

                if pred_col in df.columns and ref_col in df.columns:
                    print(f"Evaluating task: {task}, Instruction: {instr}...")
                    df[out_col] = df.progress_apply(lambda r: spec["fn"](r[ref_col], r[pred_col], r=f"{r.get('id', r.name)},instance={r.get('instance', '')}", bind=r.get('symbol_binding', None), ans=r.get('answer', None), q=r.get('question', '')), axis=1)

        print("Evaluation completes. Saving metrics...")
        df.to_csv(out_path, index=False)

        acc_dict = {}
        for task in ["original", "numerical_abstraction", "numerical_abstraction_value", "symbolic_abstraction", "arithmetic_computation"]:
            for instr in ["direct", "cot"]:
                col = f"{task}_{instr}_correctness"
                if col in df.columns:
                    acc_dict[f"{task}_{instr}_acc"] = df[col].mean()

        summary_dir = os.path.join(RESULT_DIR, "8shot") if getattr(args, "eight_shot", False) else RESULT_DIR
        summary_path = os.path.join(summary_dir, evaluation_summary_fname)
        acc_table = save_metrics(acc_dict, model_key, summary_path)
