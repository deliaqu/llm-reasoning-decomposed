"""Shared data loading helpers for interpretability experiments."""

import glob
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULT_DIR, HOME_DIR


def count_ops(expr: str) -> int:
    """Count arithmetic operators in a numerical expression."""
    return len(re.findall(r'[+\-*/]', str(expr)))


def count_calc_steps(solution: str) -> int:
    """Count explicit arithmetic computations in a GSM8K solution.

    Each <<expr=result>> marker represents one arithmetic step performed by
    the solver. This is a tighter label than sentence count (n_steps), which
    also includes verbal/algebraic steps without explicit calculation.
    """
    return len(re.findall(r'<<', str(solution)))


def noop_template_ids() -> set:
    """Return the set of GSM8K row indices used as NoOp templates."""
    import os as _os
    sys.path.append(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    from paths import DATA_DIR
    path = _os.path.join(DATA_DIR, "gsm_symbolic", "test_gsm_noop.csv")
    df = pd.read_csv(path, usecols=["original_id"])
    return set(df["original_id"].unique())


def load_gsm8k(model_short: str, exclude_noop_templates: bool = True, correct_only: bool = False) -> pd.DataFrame:
    path = os.path.join(RESULT_DIR, f"gsm8k_results_{model_short}.csv")
    df = pd.read_csv(path)
    df["n_ops"] = df["numerical_abstraction_answer"].apply(count_ops)
    # Override pre-computed n_steps (sentence count) with <<>> arithmetic step count
    df["n_steps"] = df["solution"].apply(count_calc_steps)
    # Exclude noop templates first, while index still matches original GSM8K row positions
    if exclude_noop_templates:
        excluded = noop_template_ids()
        before = len(df)
        df = df[~df.index.isin(excluded)].reset_index(drop=True)
        print(f"Excluded {before - len(df)} GSM8K problems used as NoOp templates "
              f"({len(df)} remaining)")
    # Drop problems with 0 or 1 calc steps — algebraic solutions with unreliable labels
    bad = df["n_steps"] <= 1
    if bad.any():
        print(f"Dropping {bad.sum()} GSM8K rows with calc_steps <= 1 (unreliable labels)")
        df = df[~bad].reset_index(drop=True)
    if correct_only:
        before = len(df)
        df = df[df["original_cot_correctness"].astype(float) == 1].reset_index(drop=True)
        print(f"Keeping only correct GSM8K instances ({len(df)}/{before})")
    return df


def load_gsm_symbolic(model_short: str) -> pd.DataFrame:
    files = sorted(glob.glob(
        os.path.join(RESULT_DIR, "gsm_symbolic", f"gsm_symbolic_set_*_results_{model_short}.csv")
    ))
    if not files:
        raise FileNotFoundError(f"No GSM-Symbolic result files found for {model_short}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_gsm_noop(model_short: str, drop_bad_labels: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(
        os.path.join(RESULT_DIR, "gsm_noop", f"gsm_noop_set_*_results_{model_short}.csv")
    ))
    if not files:
        raise FileNotFoundError(f"No GSM-NoOp result files found for {model_short}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # Join <<>>-based calc step count from GSM8K via original_id (= GSM8K row index)
    gsm8k_path = os.path.join(RESULT_DIR, f"gsm8k_results_{model_short}.csv")
    if os.path.exists(gsm8k_path):
        gsm8k = pd.read_csv(gsm8k_path, usecols=["solution"])
        gsm8k["n_steps"] = gsm8k["solution"].apply(count_calc_steps)
        gsm8k_steps = gsm8k[["n_steps"]].reset_index().rename(columns={"index": "original_id"})
        df = df.merge(gsm8k_steps, on="original_id", how="left")
    if drop_bad_labels:
        # Drop NoOp instances from templates whose solutions have 0 or 1 <<>> steps —
        # these are algebraic problems where the label doesn't reflect actual complexity.
        bad = (df["n_steps"] <= 1) | df["n_steps"].isna()
        if bad.any():
            print(f"Dropping {bad.sum()} NoOp rows with calc_steps <= 1 or NaN (unreliable labels)")
            df = df[~bad].reset_index(drop=True)
    return df


def probe_dir(model_short: str, base: str = None) -> str:
    base = base or os.path.join(HOME_DIR, "results", "hidden_states")
    os.makedirs(base, exist_ok=True)
    return base


def hidden_states_path(dataset: str, model_short: str, base_dir: str) -> str:
    return os.path.join(base_dir, f"{dataset}_hidden_{model_short}.npy")


def sym_noop_idx_path(model_short: str, base_dir: str) -> str:
    """Path to the noop row indices that were matched when extracting gsm_symbolic."""
    return os.path.join(base_dir, f"gsm_symbolic_noop_idx_{model_short}.npy")
