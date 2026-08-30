"""
Presence linear probe at the answer-position residual.

Unified successor to the (now-archived) entity_presence_probe and
op_presence_probe scripts: every probe target — surface-word presence in the
prompt or formula-operator presence in the template — sits in one pipeline
with one statistical contract, one output directory, one figure.

Categories — all probed under the same CV pipeline + bootstrap-CI contract:

  Per-prompt word-presence (binary, K selected target probes per category):
    entity_name      — proper-name role fillers
    entity_other     — common-noun role fillers (colors, sports, foods, ...)
    op_cue           — arithmetic-cue words ("total", "more", "twice", ...)
    function_word    — generic narrative function words (negative control)
    number           — numeric-token presence

  Per-template formula structure (binary, fixed targets):
    op_presence      — does the formula use +, −, ×, ÷?

Surface-token categories (entity_name, entity_other, number) carry positives
that vary within a paired-data pair (the swap modifies a single role's value),
so they're probed natively under paired_template_disjoint CV. Template-static
categories (op_cue, function_word, op_presence) have identical labels for both
halves of every pair, so paired CV is degenerate; per-target auto-fallback to
template_disjoint CV on the label==1 originals subset handles them.

Output layout (replaces results/entity_presence_probe/ and results/op_presence_probe/):
  results/presence_probe/<model>/<dataset>/<cv_mode>/<mode>/<correctness>/
    presence_probe.npz             # per-category probe accuracy arrays
    presence_probe_meta.json       # word lists + run config
    presence_probe.png             # aggregated 6-category figure
  where <correctness> ∈ {all, correct, wrong}.

Usage:
    python run_presence_probe.py --model_id meta-llama/Llama-3.3-70B-Instruct
    python run_presence_probe.py --plot_only
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP
from run_input_recovery import NUMBER_RE, load_aligned_questions
from run_op_multiset_probe import op_multiset
from run_template_similarity import (
    _correctness_col, _coerce_bool, cache_path, load_gsm_dataset, meta_path,
)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from paths import RESULT_DIR as _DE_RESULT_DIR

OUT_DIR = Path(LOGIT_LENS_RESULT_DIR).parent / "presence_probe"
# Per-template formula-op probes (op_presence category).
OPS = ["+", "-", "*", "/"]
OP_NAMES = {"+": "plus", "-": "minus", "*": "times", "/": "divide"}
TEMPLATE_DIR = Path(__file__).parent / "data/gsm_symbolic/templates/symbolic"
PAIRED_ALIGN_KEYS = [
    "original_id", "instance", "pair_id", "label", "value",
]

# Keep the lexical category definitions local so this probe does not depend on
# internal constant names from run_input_recovery.py.
OP_CUES = {
    "total", "totals", "totaling", "totaled", "altogether", "combined",
    "sum", "summed", "together", "all", "entire", "whole",
    "more", "less", "fewer", "greater", "smaller", "bigger", "larger",
    "longer", "shorter", "than", "difference",
    "twice", "thrice", "double", "triple", "quadruple", "half", "halved",
    "doubled", "tripled", "times", "percent", "percentage",
    "ratio", "rate", "average", "mean",
    "gives", "gave", "given", "received", "receives", "took", "takes",
    "buys", "bought", "sells", "sold", "earns", "earned",
    "spends", "spent", "paid", "owes", "owed", "wins", "won",
    "loses", "lost", "ate", "eats", "eaten", "used", "uses",
    "added", "subtracted", "removed",
    "remaining", "left", "leftover", "rest",
    "each", "every", "per",
    "plus", "minus", "multiplied", "divided",
    "calculate", "find", "compute",
}

FUNCTION = {
    "the", "a", "an", "some", "any", "all", "both", "either", "neither",
    "and", "or", "but", "if", "then", "so", "because", "as", "though",
    "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "into", "out", "up", "down", "over", "under", "through", "between",
    "while", "during", "after", "before",
    "is", "are", "was", "were", "be", "been", "being",
    "has", "have", "had", "having",
    "do", "does", "did", "done", "doing",
    "can", "could", "will", "would", "shall", "should",
    "may", "might", "must", "ought",
    "this", "that", "these", "those",
    "it", "its", "he", "she", "his", "her", "him", "they", "them", "their",
    "we", "us", "you", "your", "yours", "i", "my", "me", "mine",
    "what", "how", "much", "many", "when", "where", "who", "whom", "which", "why",
    "there", "here", "now", "no", "not", "yes",
    "also", "only", "just", "even", "still", "yet",
    "wants", "wanted", "want", "needs", "need", "needed",
    "knows", "know", "knew", "thinks", "think", "thought",
    "decides", "decided", "decide",
    "goes", "went", "going", "go",
    "comes", "came", "coming", "come",
    "same", "different", "such", "other", "another",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve",
}

NUMERIC_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "half", "halves", "quarter",
    "quarters", "third", "thirds", "fourth", "fourths", "fifth", "fifths",
    "sixth", "sixths", "seventh", "sevenths", "eighth", "eighths", "ninth",
    "ninths", "tenth", "tenths", "dozen",
}

NUMERIC_CONNECTORS = {"and", "a", "an", "of"}
TEXT_NUMBER_RE = re.compile(
    r"""
    \b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|
       thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|
       thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|
       billion|half|halves|quarter|quarters|third|thirds|fourth|fourths|fifth|
       fifths|sixth|sixths|seventh|sevenths|eighth|eighths|ninth|ninths|tenth|
       tenths|dozen)
       (?:[-\s]+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|
       eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|
       nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|
       thousand|million|billion|half|halves|quarter|quarters|third|thirds|
       fourth|fourths|fifth|fifths|sixth|sixths|seventh|sevenths|eighths|
       eighth|ninth|ninths|tenth|tenths|dozen|and|a|an|of))*\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
WORD_RE = re.compile(r"\b[A-Za-z]+\b")


# ---------------------------------------------------------------------------
# Template-derived entity vocabulary
# ---------------------------------------------------------------------------

_INIT_ROLE_RE  = re.compile(r"^\s*-\s*([^=]+?)=", re.MULTILINE)
_MARKUP_RE     = re.compile(r"\{([^{}]+?),\s*([^{}]+?)\}")
_SAMPLE_LIST_RE = re.compile(r"sample\(\[([^\]]+)\]")
_QUOTED_RE     = re.compile(r"['\"]([^'\"]+)['\"]")
_UNIT_ROLE_RE = re.compile(r"(^|_)(cur|currency|unit|units?|symbol)($|_)")
_QUANTITY_WORDS = {
    "cent", "cents", "dollar", "dollars",
    "second", "seconds", "minute", "minutes", "hour", "hours", "day", "days",
    "week", "weeks", "month", "months", "year", "years",
    "inch", "inches", "foot", "feet", "yard", "yards", "mile", "miles",
    "meter", "meters", "kilometer", "kilometers", "gram", "grams",
    "kilogram", "kilograms", "pound", "pounds", "ounce", "ounces", "oz",
    "liter", "liters", "milliliter", "milliliters", "ml", "cc",
    "percent", "percentage",
}


def _init_section(annotated: str) -> str:
    if "#init:" not in annotated:
        return ""
    return (annotated.split("#init:")[1]
            .split("#conditions")[0]
            .split("#answer")[0])


def _parse_init_assignments(annotated: str):
    """Return (numeric_roles, role_rhs) parsed from the #init: section.

    A role declared with `$` prefix is numeric (operand); without `$` it is
    a candidate categorical surface role.  Roles can be declared jointly:
    `obj1, obj2 = sample([...])`.
    """
    numeric, role_rhs = set(), {}
    init = _init_section(annotated)
    for line in init.split("\n"):
        line = line.strip().lstrip("- ")
        if not line or "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        for raw_role in lhs.split(","):
            r = raw_role.strip()
            if not r:
                continue
            if r.startswith("$"):
                numeric.add(r[1:])
            else:
                role_rhs[r] = rhs.strip()
    return numeric, role_rhs


def _words_from_value(value: str) -> list[str]:
    return [
        w.lower()
        for w in re.findall(r"[A-Za-z]+", value)
        if len(w) > 1
    ]


def _looks_numeric_or_quantity(value: str) -> bool:
    words = _words_from_value(value)
    if not words:
        return True
    allowed = NUMERIC_WORDS | NUMERIC_CONNECTORS | _QUANTITY_WORDS
    return all(w in allowed for w in words)


def _is_entity_role(role: str, rhs: str, example_values: list[str]) -> bool:
    """Whether a non-$ template role should count as an entity surface role."""
    role_l = role.lower()
    if _UNIT_ROLE_RE.search(role_l):
        return False

    sample_values = _QUOTED_RE.findall(rhs)
    values = sample_values or example_values
    if values and all(_looks_numeric_or_quantity(v) for v in values):
        return False

    return True


def _is_name_role(role: str) -> bool:
    """Heuristic split: roles whose name contains 'name' carry proper-name
    surface tokens (`name`, `name1`..`name5`, `school_name`, `park_name`,
    `comet_name`, `fee1_name`, ...). Everything else marked as an entity role
    is treated as a common-noun entity (objects, colors, foods, sports, days,
    relations, places, ...)."""
    return "name" in role.lower()


def _literal_pattern(text: str) -> str:
    """Escape literal template text while normalizing whitespace.

    Whitespace runs become `\\s+`. Non-whitespace tokens are escaped literally.
    Critical to preserve `\\s+` even between adjacent markups whose only
    separator is a single space — otherwise adjacent `.+?` capture groups have
    no separator and bleed into each other.
    """
    if not text:
        return ""
    out = []
    for tok in re.split(r"(\s+)", text):
        if not tok:
            continue
        if tok.isspace():
            out.append(r"\s+")
        else:
            out.append(re.escape(tok))
    return "".join(out)


def _compile_question_pattern(question_part: str):
    # Strip leading/trailing whitespace from the template before pattern
    # building — rendered instance questions are .strip()-ed, so requiring
    # `\s+` for template-edge whitespace (e.g., trailing `\n\n`) would cause
    # spurious mismatches against every prompt. Mirrors the fix in
    # build_pair_inference_data.build_extraction_regex.
    question_part = question_part.strip()
    parts = []
    role_examples = {}
    seen_roles = set()
    last = 0

    for m in _MARKUP_RE.finditer(question_part):
        parts.append(_literal_pattern(question_part[last:m.start()]))
        role = m.group(1).strip()
        value = m.group(2).strip()
        role_examples.setdefault(role, []).append(value)
        if role in seen_roles:
            parts.append(r".+?")
        else:
            parts.append(fr"(?P<{role}>.+?)")
            seen_roles.add(role)
        last = m.end()

    parts.append(_literal_pattern(question_part[last:]))
    pattern = r"^\s*" + "".join(parts) + r"\s*$"
    return re.compile(pattern, flags=re.IGNORECASE | re.DOTALL), role_examples


def parse_template_entity_specs(template_dir: Path = TEMPLATE_DIR) -> dict:
    """Return per-template extraction specs for marked entity-like roles."""
    specs = {}
    for json_file in sorted(template_dir.glob("*.json")):
        with open(json_file) as f:
            t = json.load(f)
        annotated = t["question_annotated"]
        orig_id = int(t["id_orig"])
        question_part = annotated.split("#init:")[0]
        numeric_roles, role_rhs = _parse_init_assignments(annotated)
        pattern, role_examples = _compile_question_pattern(question_part)

        entity_roles = {
            role
            for role, examples in role_examples.items()
            if role not in numeric_roles
            and _is_entity_role(role, role_rhs.get(role, ""), examples)
        }
        specs[orig_id] = {
            "pattern": pattern,
            "entity_roles": entity_roles,
            "role_examples": role_examples,
        }
    return specs


def parse_template_entities(template_dir: Path = TEMPLATE_DIR) -> dict:
    """Return {original_id: set of entity-surface words} for each template.

    Combines two sources:
      1. Inline `{role, value}` markups in the question_annotated (gives the
         example surface form for each role).
      2. Inline `sample([...])` lists in the #init: section (gives the full
         vocabulary for that role).

    External list references (e.g. `sample(names_male)`) are not resolved —
    those entity values are recovered empirically from the inline markups
    that appear when the template is instantiated, so they show up in the
    instance questions even if not in this lookup.  We therefore *also* fall
    back at runtime to "any non-op-cue / non-function / non-number word in
    the question whose template's entity vocabulary indicates an entity
    role" — see `build_entity_vocab_from_instances`.
    """
    specs = parse_template_entity_specs(template_dir)
    out = {}
    for orig_id, spec in specs.items():
        words = set()
        entity_roles = spec["entity_roles"]
        for role, values in spec["role_examples"].items():
            if role in entity_roles:
                for value in values:
                    words.update(_words_from_value(value))
        out[orig_id] = words
    return out


def build_entity_vocab_from_instances(
    questions: list[str],
    template_ids: np.ndarray,
    template_entity_vocab: dict,
) -> tuple:
    """Return (per_prompt_entity_words, per_prompt_name_words,
    per_prompt_other_words, prompt_freq, template_count, word_subcat).

    Entity words are extracted from the concrete question by matching it
    against the corresponding `question_annotated` template and reading only
    marked entity-like roles.  This avoids the old "varies within template"
    fallback, which incorrectly treated sampled durations/units like
    "45 minutes" as entities.

    The name/other split uses `_is_name_role` on the source template role.
    `word_subcat` is a {word: "name"|"other"} map built from the most-common
    sub-role assignment seen across all prompts (used downstream for selection
    stratification and per-category training).
    """
    specs = parse_template_entity_specs()
    per_prompt_entities: list[set] = []
    per_prompt_name: list[set] = []
    per_prompt_other: list[set] = []
    subcat_counts: dict = {}
    unmatched = 0

    def _filter(words: set) -> set:
        return {
            w for w in words
            if w not in OP_CUES
            and w not in FUNCTION
            and w not in NUMERIC_WORDS
            and w not in _QUANTITY_WORDS
            and not NUMBER_RE.fullmatch(w)
        }

    for question, t in zip(questions, template_ids):
        spec = specs.get(int(t))
        ents: set = set()
        name_ents: set = set()
        other_ents: set = set()
        if spec is not None:
            matched = spec["pattern"].match(question)
            if matched is not None:
                for role in spec["entity_roles"]:
                    value = matched.groupdict().get(role)
                    if not value:
                        continue
                    role_words = _words_from_value(value)
                    bucket = name_ents if _is_name_role(role) else other_ents
                    for w in role_words:
                        bucket.add(w)
                        subcat = "name" if _is_name_role(role) else "other"
                        subcat_counts.setdefault(w, Counter())[subcat] += 1
                    ents.update(role_words)
            else:
                unmatched += 1

        if not ents:
            # Fallback: any non-op/non-function/non-number question word that
            # appears in this template's entity vocab. Subcat unknown here →
            # default to "other" so we don't inflate the name bucket.
            question_words = {m.group(0).lower() for m in WORD_RE.finditer(question)}
            ents = question_words & template_entity_vocab.get(int(t), set())
            other_ents = ents.copy()
            for w in ents:
                subcat_counts.setdefault(w, Counter())["other"] += 1

        ents = _filter(ents)
        name_ents = _filter(name_ents)
        other_ents = _filter(other_ents)
        per_prompt_entities.append(ents)
        per_prompt_name.append(name_ents)
        per_prompt_other.append(other_ents)

    if unmatched:
        print(f"  warning: template entity regex did not match {unmatched} prompts; used fallback vocab")

    # Frequency stats for target selection
    prompt_freq: Counter = Counter()
    seen_per_template: dict = {}
    for ents, t in zip(per_prompt_entities, template_ids):
        for w in ents:
            prompt_freq[w] += 1
            seen_per_template.setdefault(w, set()).add(int(t))
    template_count = {w: len(ts) for w, ts in seen_per_template.items()}

    word_subcat = {w: cnts.most_common(1)[0][0] for w, cnts in subcat_counts.items()}

    return (per_prompt_entities, per_prompt_name, per_prompt_other,
            prompt_freq, template_count, word_subcat)


# ---------------------------------------------------------------------------
# Word extraction (word-level, not token-level)
# ---------------------------------------------------------------------------

def extract_words_per_prompt(questions: list[str]) -> list[dict]:
    """For each question, return {category: set(words)} — word-level categorization."""
    out = []

    def span_overlaps(s: int, e: int, spans) -> bool:
        return any(max(s, ss) < min(e, ee) for ss, ee in spans)

    for q in questions:
        cats = {"entity": set(), "entity_name": set(), "entity_other": set(),
                "op_cue": set(), "number": set(), "function_word": set(),
                "answer": set()}
        number_spans = []
        for m in NUMBER_RE.finditer(q):
            cats["number"].add(m.group(0).lower())
            number_spans.append((m.start(), m.end()))

        digit_number_spans = list(number_spans)
        for m in TEXT_NUMBER_RE.finditer(q):
            words = [
                wm.group(0).lower()
                for wm in WORD_RE.finditer(m.group(0))
                if wm.group(0).lower() not in NUMERIC_CONNECTORS
            ]
            if not words or not all(w in NUMERIC_WORDS for w in words):
                continue
            if span_overlaps(m.start(), m.end(), digit_number_spans):
                continue
            cats["number"].add(" ".join(words))
            for wm in WORD_RE.finditer(m.group(0)):
                word = wm.group(0).lower()
                if word in NUMERIC_CONNECTORS:
                    continue
                number_spans.append((
                    m.start() + wm.start(),
                    m.start() + wm.end(),
                ))

        for m in WORD_RE.finditer(q):
            if span_overlaps(m.start(), m.end(), number_spans):
                continue
            word = m.group(0).lower()
            if word in OP_CUES:
                cats["op_cue"].add(word)
            elif word in FUNCTION:
                cats["function_word"].add(word)
            elif len(word) > 2:
                # Length filter cuts down on punctuation and abbreviations.
                cats["entity"].add(word)
        out.append(cats)
    return out


def _count_variable_templates(labels: np.ndarray, template_ids: np.ndarray) -> int:
    """Templates where the word's presence varies across instances:
    0 < positives_in_template < total_instances_in_template. These are the
    only templates usable for within-template CV."""
    cnt = 0
    for t in np.unique(template_ids):
        m = template_ids == t
        s = int(labels[m].sum())
        if 0 < s < int(m.sum()):
            cnt += 1
    return cnt


def per_template_op_presence(model_id: str) -> dict:
    """{template_id -> {op: bool}} parsed from gsm_symbolic results CSV.

    op-presence labels are template-level (one set of booleans per template),
    derived from `symbolic_abstraction_answer`. Every instance of the same
    template inherits the same labels. Used to build the op_presence category
    in the unified probe pipeline.
    """
    import glob as _glob
    file_suffix = model_id.split("/")[-1]
    # Try vLLM-backend path first (used for Llama), then fall back to
    # transformers_direct (used for gemma / qwen* per the project convention).
    files = sorted(_glob.glob(os.path.join(
        _DE_RESULT_DIR, "gsm_symbolic",
        f"gsm_symbolic_set_*_results_{file_suffix}.csv",
    )))
    if not files:
        files = sorted(_glob.glob(os.path.join(
            _DE_RESULT_DIR, "transformers_direct", "gsm_symbolic",
            f"gsm_symbolic_set_*_results_{file_suffix}_transformers.csv",
        )))
    if not files:
        raise FileNotFoundError(
            f"per_template_op_presence: no gsm_symbolic results files for "
            f"{file_suffix} under {Path(_DE_RESULT_DIR) / 'gsm_symbolic'} "
            f"or transformers_direct/gsm_symbolic"
        )
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    template_df = df.drop_duplicates("original_id")[
        ["original_id", "symbolic_abstraction_answer"]
    ]
    out: dict = {}
    for row in template_df.itertuples():
        ms = op_multiset(row.symbolic_abstraction_answer)
        out[int(row.original_id)] = {op: (op in ms) for op in OPS}
    return out


def _count_informative_pairs(labels: np.ndarray, pair_ids: np.ndarray) -> int:
    """Pairs with exactly two rows and exactly one target-word-positive row."""
    pair_pos = {}
    for i, pid in enumerate(pair_ids):
        pair_pos.setdefault(pid, []).append(int(labels[i]))
    return sum(1 for ys in pair_pos.values() if len(ys) == 2 and sum(ys) == 1)


def _count_informative_pair_templates(
    labels: np.ndarray, pair_ids: np.ndarray, template_ids: np.ndarray,
) -> int:
    """Number of distinct templates that contribute at least one informative
    pair for the target word. Used to gate eligibility for
    `paired_template_disjoint` CV (needs ≥n_folds distinct templates to form
    template-out-of-distribution test folds)."""
    pair_rows: dict = {}
    for i, pid in enumerate(pair_ids):
        pair_rows.setdefault(pid, []).append(i)
    templates = set()
    for pid, rows in pair_rows.items():
        if len(rows) == 2 and int(labels[rows].sum()) == 1:
            templates.add(int(template_ids[rows[0]]))
    return len(templates)


def select_balanced_words(
    per_prompt: list[dict],
    template_ids: np.ndarray,
    category: str,
    k: int,
    min_count: int = 50,
    max_ratio: float = 0.7,
    min_pos_templates: int = 3,
    min_neg_templates: int = 3,
    pos_template_range: tuple[int, int] | None = None,
    pos_count_range: tuple[int, int] | None = None,
    cv_mode: str = "template_disjoint",
    pair_ids: np.ndarray | None = None,
    word_to_group: dict | None = None,
    originals_mask: np.ndarray | None = None,
) -> list[tuple]:
    """Pick K words from `category` with usable CV splits.

    cv_mode="template_disjoint" (default): requires ≥min_pos_templates positive
    and ≥min_neg_templates negative templates; sorts by template-balance then
    prompt-balance. This biases toward words spread across many positive
    templates, which in GSM-Symbolic happens to favor proper-name entities
    (since many templates instantiate from a names list, names span more
    templates than e.g. animal/object entities).

    cv_mode="within_template": requires ≥min_pos_templates *variable* templates
    (templates where the word appears in some but not all instances). Sorts by
    number of variable templates (more = better CV stability), then by total
    count. This admits non-name entities (animals, objects, days, kin terms)
    that appear in fewer templates but with within-template variance.

    cv_mode="paired": requires ≥min_pos_templates informative matched pairs,
    where exactly one row in the pair contains the word. This is the paired-CV
    eligibility condition; template-count constraints are intentionally ignored
    because templates are not the held-out units in this mode.

    cv_mode="paired_template_disjoint": requires ≥min_pos_templates informative
    pairs AND ≥min_pos_templates *distinct* templates contributing informative
    pairs (so template-out-of-distribution test folds can be formed).

    `pos_template_range` and `pos_count_range` (both inclusive) constrain the
    candidate words' template-count and prompt-count to a target band — used to
    keep control categories distribution-matched against the entity selection.

    Returns list of (word, prompt_count) tuples.
    """
    n = len(per_prompt)
    counter = Counter()
    for d in per_prompt:
        for w in d[category]:
            counter[w] += 1

    candidates = []
    for word, count in counter.items():
        if not (min_count <= count <= int(max_ratio * n)):
            continue
        if pos_count_range is not None and not (
            pos_count_range[0] <= count <= pos_count_range[1]
        ):
            continue

        labels = np.array(
            [1 if word in d[category] else 0 for d in per_prompt],
            dtype=np.int32,
        )
        pos_templates = set(template_ids[labels == 1].tolist())
        neg_templates = set(template_ids[labels == 0].tolist()) - pos_templates

        # Distribution-matching to the entity selection (function-word and
        # number controls pass pos_template_range=ent_template_range). The
        # check is hoisted out of the cv_mode branches so it applies under
        # every mode — previously it was only enforced in the
        # template_disjoint else-branch, which silently let controls drift
        # in paired_template_disjoint runs.
        if pos_template_range is not None and not (
            pos_template_range[0] <= len(pos_templates) <= pos_template_range[1]
        ):
            continue

        if cv_mode == "paired":
            if pair_ids is None:
                raise ValueError("cv_mode='paired' requires pair_ids")
            n_inform = _count_informative_pairs(labels, pair_ids)
            if n_inform < min_pos_templates:
                continue
            prompt_balance = abs(count - n / 2)
            candidates.append((word, count, len(pos_templates),
                               len(neg_templates), -n_inform, prompt_balance))
        elif cv_mode == "paired_template_disjoint":
            if pair_ids is None:
                raise ValueError(
                    "cv_mode='paired_template_disjoint' requires pair_ids")
            n_inform = _count_informative_pairs(labels, pair_ids)
            n_inform_templates = _count_informative_pair_templates(
                labels, pair_ids, template_ids)
            paired_ok = (n_inform >= min_pos_templates
                         and n_inform_templates >= min_pos_templates)
            if paired_ok:
                prompt_balance = abs(count - n / 2)
                # Rank by template-count (more templates → more CV stability),
                # then by informative-pair count.
                candidates.append((word, count, len(pos_templates),
                                   len(neg_templates),
                                   -n_inform_templates, -n_inform))
            elif originals_mask is not None:
                # Template-static target (e.g., function words, common
                # op-cues) — paired-CV is degenerate because both halves of
                # every pair share the label. Evaluate eligibility on the
                # label==1 originals subset with template_disjoint criteria;
                # at training time the same fallback will fire.
                sub_labels = labels[originals_mask]
                sub_templates = template_ids[originals_mask]
                sub_pos = set(sub_templates[sub_labels == 1].tolist())
                sub_neg = (set(sub_templates[sub_labels == 0].tolist())
                           - sub_pos)
                if (len(sub_pos) < min_pos_templates
                        or len(sub_neg) < min_neg_templates):
                    continue
                # Rank these candidates AFTER the paired-eligible ones by
                # giving them a sentinel -1 in the template-count slot. Among
                # themselves, rank by template-balance then prompt-balance,
                # mirroring template_disjoint selection.
                tbal = abs(len(sub_pos) - len(sub_neg))
                pbal = abs(int(sub_labels.sum()) - len(sub_labels) / 2)
                candidates.append((word, int(sub_labels.sum()),
                                   len(sub_pos), len(sub_neg),
                                   -1, tbal + pbal / max(1, len(sub_labels))))
            else:
                continue
        elif cv_mode == "within_template":
            n_var = _count_variable_templates(labels, template_ids)
            if n_var < min_pos_templates:
                continue
            # Sort key: more variable templates first (more stable CV),
            # then larger total count.
            candidates.append((word, count, len(pos_templates),
                               len(neg_templates), -n_var, -count))
        else:
            if (len(pos_templates) < min_pos_templates
                    or len(neg_templates) < min_neg_templates):
                continue
            template_balance = abs(len(pos_templates) - len(neg_templates))
            prompt_balance = abs(count - n / 2)
            candidates.append((word, count, len(pos_templates),
                               len(neg_templates), template_balance,
                               prompt_balance))

    candidates.sort(key=lambda item: (item[4], item[5]))

    if word_to_group is None:
        return [(w, c) for w, c, *_ in candidates[:k]]

    # Group-stratified round-robin: cycle through groups, picking the next
    # best (word, count) from each in turn, until k words are selected.
    # Used in paired CV so K_ENTITY isn't dominated by one role (e.g., names
    # win the unstratified ranking because they span the most templates).
    by_group: dict = {}
    for w, c, *_ in candidates:
        g = word_to_group.get(w, "_unknown")
        by_group.setdefault(g, []).append((w, c))
    groups = sorted(by_group.keys())
    selected: list = []
    while len(selected) < k:
        progressed = False
        for g in groups:
            if len(selected) >= k:
                break
            if by_group[g]:
                selected.append(by_group[g].pop(0))
                progressed = True
        if not progressed:
            break
    return selected


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def fit_one_layer(X_tr, y_tr, X_te, y_te):
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe.score(X_te, y_te)


def load_probe_aligned_questions(
    model_id: str,
    meta_csv: Path,
    dataset_name: str,
    mode: str,
) -> pd.DataFrame:
    """Load source questions in exactly the cached hidden-state row order.

    The legacy helper from run_input_recovery is correct for plain
    gsm_symbolic, where (original_id, instance) uniquely identifies a row.
    gsm_symbolic_pairs has many rows with the same base key, so paired data
    must align on pair-specific columns and retain pair metadata.
    """
    if dataset_name != "gsm_symbolic_pairs":
        return load_aligned_questions(model_id, meta_csv)

    meta = pd.read_csv(meta_csv)
    df = load_gsm_dataset(
        model_id, dataset_name=dataset_name, correct_only=False, mode=mode,
    )
    align_keys = [k for k in PAIRED_ALIGN_KEYS if k in meta.columns and k in df.columns]
    missing = [k for k in PAIRED_ALIGN_KEYS if k not in align_keys]
    if missing:
        raise ValueError(
            "Cannot align gsm_symbolic_pairs cache for entity_presence_probe: "
            f"missing required paired key columns {missing}. Rebuild the "
            "template-similarity cache so its meta CSV includes pair_id, "
            "label, and value."
        )
    if meta.duplicated(align_keys).any():
        raise ValueError(
            f"Cached paired meta has duplicate alignment keys: {align_keys}"
        )
    if df.duplicated(align_keys).any():
        raise ValueError(
            f"Paired result CSVs have duplicate alignment keys: {align_keys}"
        )

    meta_order = meta[align_keys].copy()
    df_aligned = meta_order.merge(df, on=align_keys, how="left")
    if len(df_aligned) != len(meta):
        raise AssertionError("Alignment mismatch")
    if df_aligned["question"].isna().any():
        bad = df_aligned[df_aligned["question"].isna()][align_keys].head(3)
        raise AssertionError(
            "Some paired questions missing after merge; sample keys: "
            f"{bad.to_dict(orient='records')}"
        )

    # Preserve cache meta columns (especially correctness columns and pair
    # annotations) as the source of truth for downstream filtering.
    for col in meta.columns:
        if col in align_keys:
            continue
        if col in df_aligned.columns:
            df_aligned[col] = meta[col].values
        else:
            df_aligned.insert(len(df_aligned.columns), col, meta[col].values)
    return df_aligned.reset_index(drop=True)


def subsample_to_match_distribution(
    labels: np.ndarray,
    template_ids: np.ndarray,
    target_n_templates: int,
    target_n_positives: int,
    seed: int,
) -> np.ndarray:
    """Return a copy of `labels` with positives subsampled so the positive set
    spans `target_n_templates` distinct templates and contains
    `target_n_positives` total prompts (best effort).

    Drops whole templates first to hit the template-count target, then drops
    individual positive prompts within retained templates to hit the
    prompt-count target. Negatives are untouched.

    Used to match op-cue probes to the entity per-word distribution: op-cues
    naturally appear in many more templates than proper-name entities, so
    without subsampling the op-cue probe would benefit from greater train
    template diversity, confounding any accuracy difference.
    """
    rng = np.random.RandomState(seed)
    new_labels = labels.copy()
    pos_idx = np.where(labels == 1)[0]
    if len(pos_idx) == 0:
        return new_labels

    pos_templates = sorted(set(template_ids[pos_idx].tolist()))
    if len(pos_templates) > target_n_templates:
        keep = set(rng.choice(pos_templates, target_n_templates, replace=False).tolist())
        drop_mask = np.array([t not in keep for t in template_ids[pos_idx]])
        new_labels[pos_idx[drop_mask]] = 0
        pos_idx = np.where(new_labels == 1)[0]

    if len(pos_idx) > target_n_positives:
        drop = rng.choice(pos_idx, len(pos_idx) - target_n_positives, replace=False)
        new_labels[drop] = 0

    return new_labels


def within_template_splits(
    labels: np.ndarray,
    template_ids: np.ndarray,
    n_folds: int,
    cap_positives: int,
    seed: int,
    min_fold_pos: int = 10,
):
    """Build CV splits where train and test both draw from the same set of
    *variable templates* (templates where the word appears in some but not all
    instances), restricted to those templates' instances only. Templates where
    the word never appears or always appears contribute nothing.

    This isolates "is the entity encoded?" from "is the template encoded?":
    the probe cannot win by detecting "this prompt is from a template that
    sometimes has the word" because both train and test prompts come from
    exactly that pool. Within that pool, positives and negatives are pooled
    across templates and split into n_folds chunks (random, class-stratified
    only via separate pos/neg chunking).

    Why pool rather than per-template stratified KFold: many entity words have
    very few positives per template (e.g. names sampled 1× from a list of 30
    inside a single template instance), so per-template KFold demanding
    ≥n_folds positives per template would skip nearly every template. Pooling
    keeps the probe trainable while preserving template-shared sampling.

    Train and test sets are class-balanced; train is capped at cap_positives.

    Returns: list of (train_idx, test_idx) numpy arrays, length n_folds.
    Returns [] if there aren't enough variable templates or instances.
    """
    rng = np.random.RandomState(seed)
    pos_mask = labels == 1

    variable_templates = []
    for t in np.unique(template_ids):
        m = template_ids == t
        s = int(labels[m].sum())
        if 0 < s < int(m.sum()):
            variable_templates.append(int(t))

    if len(variable_templates) < n_folds:
        return []

    var_mask = np.isin(template_ids, variable_templates)
    pos_pool = np.where(var_mask & pos_mask)[0]
    neg_pool = np.where(var_mask & ~pos_mask)[0]

    if len(pos_pool) < n_folds or len(neg_pool) < n_folds:
        return []

    pos_perm = rng.permutation(pos_pool)
    neg_perm = rng.permutation(neg_pool)
    pos_chunks = np.array_split(pos_perm, n_folds)
    neg_chunks = np.array_split(neg_perm, n_folds)

    splits = []
    for fold in range(n_folds):
        train_pos = np.concatenate(
            [pos_chunks[i] for i in range(n_folds) if i != fold])
        train_neg = np.concatenate(
            [neg_chunks[i] for i in range(n_folds) if i != fold])
        test_pos = pos_chunks[fold]
        test_neg = neg_chunks[fold]

        n_tr = min(len(train_pos), len(train_neg), cap_positives)
        if n_tr < min_fold_pos:
            return []
        n_te = min(len(test_pos), len(test_neg))
        if n_te < max(min_fold_pos // 2, 2):
            return []

        train_idx = np.concatenate([
            rng.choice(train_pos, n_tr, replace=False),
            rng.choice(train_neg, n_tr, replace=False),
        ])
        test_idx = np.concatenate([
            rng.choice(test_pos, n_te, replace=False),
            rng.choice(test_neg, n_te, replace=False),
        ])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        splits.append((train_idx, test_idx))

    return splits


def paired_splits(
    labels: np.ndarray,
    pair_ids: np.ndarray,
    n_folds: int,
    cap_positives: int,
    seed: int,
):
    """Build CV splits keyed on pair_id. Each pair contributes both halves
    (the original-entity row and the swap-entity row) to the SAME fold; held-out
    pairs are tested as units. This exploits the matched-pair structure of
    `gsm_symbolic_pairs` — the probe is asked "given a residual, was the
    target word present?" on prompt pairs that differ only in the target word
    itself, so a probe that picks up template / surface-context features will
    score 50% (since both halves of a pair share those).

    Only "informative" pairs are used: pairs where exactly one half is positive
    for the target word (the other being the swap with a different value).
    Pairs where neither half contains the word, or both do (impossible by
    construction since target_value ≠ swap_value), are excluded.

    Each fold's train/test set has exactly N pairs × 2 rows = 2N rows, with
    perfect 1:1 class balance by construction.

    cap_positives caps the per-fold training set in pairs (so train rows ≤
    2 × cap_positives). Test set is uncapped.

    Returns: list of (train_idx, test_idx) numpy arrays, length n_folds.
    Returns [] if fewer than n_folds informative pairs exist.
    """
    rng = np.random.RandomState(seed)

    # Group rows by pair_id; identify informative pairs.
    pair_rows: dict[object, list[int]] = {}
    for i, pid in enumerate(pair_ids):
        pair_rows.setdefault(pid, []).append(i)

    informative = []
    for pid, rows in pair_rows.items():
        ys = labels[rows]
        # Exactly one positive and one negative half.
        if len(rows) == 2 and ys.sum() == 1:
            informative.append(pid)

    if len(informative) < n_folds:
        return []

    perm_pids = rng.permutation(np.array(informative, dtype=object))
    fold_pids = np.array_split(perm_pids, n_folds)

    splits = []
    for fold in range(n_folds):
        test_pids = set(fold_pids[fold].tolist())
        train_pids = [pid for f in range(n_folds) if f != fold
                      for pid in fold_pids[f].tolist()]

        if cap_positives is not None and len(train_pids) > cap_positives:
            train_pids = list(rng.choice(train_pids, cap_positives, replace=False))

        train_idx = np.array(
            [i for pid in train_pids for i in pair_rows[pid]], dtype=np.int64)
        test_idx = np.array(
            [i for pid in test_pids for i in pair_rows[pid]], dtype=np.int64)

        # Need ≥2 pairs (4 rows) in train and ≥1 pair (2 rows) in test.
        # Paired CV is naturally class-balanced 1:1 so absolute minima are low.
        if len(train_idx) < 4 or len(test_idx) < 2:
            return []

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        splits.append((train_idx, test_idx))

    return splits


def paired_template_disjoint_splits(
    labels: np.ndarray,
    pair_ids: np.ndarray,
    template_ids: np.ndarray,
    n_folds: int,
    cap_positives: int,
    seed: int,
):
    """Paired CV with template-out-of-distribution test folds.

    Combines two controls:
      1. Pair-aware: each informative pair contributes its original and swap
         rows to the same fold (matched-pair negatives, as in `paired_splits`).
      2. Template-disjoint: templates appearing in a test fold are absent from
         the corresponding train fold. The probe cannot rely on
         template-conditional residual patterns learned during training.

    An informative pair has exactly one positive half (the pair_id selects
    rows that share a template, since both halves come from the same
    instantiated template). We bucket pairs by their template_id, split
    templates into K disjoint chunks, then for each fold test on all pairs
    whose template is in the test chunk.

    Returns: list of (train_idx, test_idx) arrays, length n_folds. Returns []
    if fewer than n_folds distinct templates carry informative pairs, or if
    any fold ends up with too few train/test pairs.
    """
    rng = np.random.RandomState(seed)

    pair_rows: dict[object, list[int]] = {}
    for i, pid in enumerate(pair_ids):
        pair_rows.setdefault(pid, []).append(i)

    # Informative pairs + their template_id (both halves of a pair share a
    # template, so we read the template from the first row).
    pair_to_template: dict[object, int] = {}
    informative: list = []
    for pid, rows in pair_rows.items():
        if len(rows) != 2:
            continue
        if int(labels[rows].sum()) != 1:
            continue
        informative.append(pid)
        pair_to_template[pid] = int(template_ids[rows[0]])

    templates_with_pairs = sorted({pair_to_template[pid] for pid in informative})
    if len(templates_with_pairs) < n_folds:
        return []

    perm_templates = rng.permutation(np.array(templates_with_pairs, dtype=np.int64))
    template_chunks = np.array_split(perm_templates, n_folds)

    splits = []
    for fold in range(n_folds):
        test_templates = set(int(t) for t in template_chunks[fold])
        train_templates = set(int(t) for t in templates_with_pairs) - test_templates

        test_pids = [pid for pid in informative
                     if pair_to_template[pid] in test_templates]
        train_pids = [pid for pid in informative
                      if pair_to_template[pid] in train_templates]

        if cap_positives is not None and len(train_pids) > cap_positives:
            train_pids = list(rng.choice(
                np.array(train_pids, dtype=object), cap_positives, replace=False))

        train_idx = np.array(
            [i for pid in train_pids for i in pair_rows[pid]], dtype=np.int64)
        test_idx = np.array(
            [i for pid in test_pids for i in pair_rows[pid]], dtype=np.int64)

        # Need ≥2 pairs (4 rows) train, ≥1 pair (2 rows) test (1:1 by
        # construction, so absolute minima can be small).
        if len(train_idx) < 4 or len(test_idx) < 2:
            return []

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        splits.append((train_idx, test_idx))

    return splits


def template_disjoint_splits(
    labels: np.ndarray,
    template_ids: np.ndarray,
    n_folds: int,
    cap_positives: int,
    seed: int,
    min_fold_pos: int = 10,
):
    """Yield (train_idx, test_idx) pairs where the templates appearing in
    `test_idx` are *disjoint* from those in `train_idx`, so the probe cannot
    win by memorising template-cluster signatures.

    Within each fold:
      - Templates are split into K disjoint chunks separately for "uses
        target word" and "doesn't use", so test always has both classes.
      - Training set is balanced (equal pos/neg samples, capped at
        cap_positives).
      - Test set is also balanced for interpretable accuracy.

    Returns: list of (train_idx, test_idx) numpy arrays, length n_folds.
    Returns [] if there aren't enough positive or negative templates to form
    n_folds non-empty test groups.
    """
    rng = np.random.RandomState(seed)
    pos_mask = labels == 1
    pos_templates = np.array(sorted(set(template_ids[pos_mask].tolist())))
    neg_templates = np.array(sorted(
        set(template_ids[~pos_mask].tolist()) - set(pos_templates.tolist())
    ))

    if len(pos_templates) < n_folds or len(neg_templates) < n_folds:
        return []  # cannot form template-disjoint folds

    pos_perm = pos_templates[rng.permutation(len(pos_templates))]
    neg_perm = neg_templates[rng.permutation(len(neg_templates))]
    pos_chunks = np.array_split(pos_perm, n_folds)
    neg_chunks = np.array_split(neg_perm, n_folds)

    splits = []
    for fold in range(n_folds):
        test_t = np.concatenate([pos_chunks[fold], neg_chunks[fold]])
        train_t = np.concatenate([
            np.concatenate([pos_chunks[i] for i in range(n_folds) if i != fold]),
            np.concatenate([neg_chunks[i] for i in range(n_folds) if i != fold]),
        ])
        in_train = np.isin(template_ids, train_t)
        in_test  = np.isin(template_ids, test_t)

        # Balance training set
        train_pos = np.where(in_train & pos_mask)[0]
        train_neg = np.where(in_train & ~pos_mask)[0]
        n_tr = min(len(train_pos), len(train_neg), cap_positives)
        if n_tr < min_fold_pos:
            return []  # not enough training examples

        # Balance test set
        test_pos = np.where(in_test & pos_mask)[0]
        test_neg = np.where(in_test & ~pos_mask)[0]
        n_te = min(len(test_pos), len(test_neg))
        if n_te < max(min_fold_pos // 2, 2):
            return []

        train_idx = np.concatenate([
            rng.choice(train_pos, n_tr, replace=False),
            rng.choice(train_neg, n_tr, replace=False),
        ])
        test_idx = np.concatenate([
            rng.choice(test_pos, n_te, replace=False),
            rng.choice(test_neg, n_te, replace=False),
        ])
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)
        splits.append((train_idx, test_idx))

    return splits


def train_probes_for_word(
    hidden: np.ndarray,
    word: str,
    labels: np.ndarray,
    template_ids: np.ndarray,
    cap_positives: int,
    n_folds: int,
    seed: int,
    n_jobs: int,
    cv_mode: str = "template_disjoint",
    pair_ids: np.ndarray | None = None,
    shuffled_labels: bool = False,
    min_fold_pos: int = 10,
) -> np.ndarray:
    """Per-layer mean accuracy (n_layers,) for one target word.

    cv_mode="template_disjoint": train/test on disjoint templates — the probe
    cannot exploit template-cluster shortcuts but is conflated with cross-template
    generalization.

    cv_mode="within_template": train/test on different instances of the *same*
    variable templates — directly tests whether the entity choice is encoded,
    controlling for template structure.

    cv_mode="paired": train/test split on pair_id (each pair = matched
    pos+neg differing only in the swapped entity). Requires pair_ids array.
    The strongest control for surface-context confounds since the two halves
    of a pair share everything except the target entity word.

    cv_mode="paired_template_disjoint": paired CV plus template-out-of-distribution
    test folds — templates in each test fold are disjoint from train. Tests
    whether the probe finds a *template-invariant* direction for the target
    entity, with matched-pair negatives. Requires pair_ids and template_ids.
    """
    n_layers = hidden.shape[1]

    if cv_mode == "paired":
        if pair_ids is None:
            return np.full(n_layers, np.nan)
        splits = paired_splits(
            labels, pair_ids, n_folds, cap_positives, seed,
        )
    elif cv_mode == "paired_template_disjoint":
        if pair_ids is None:
            return np.full(n_layers, np.nan)
        splits = paired_template_disjoint_splits(
            labels, pair_ids, template_ids, n_folds, cap_positives, seed,
        )
    elif cv_mode == "within_template":
        splits = within_template_splits(
            labels, template_ids, n_folds, cap_positives, seed,
            min_fold_pos=min_fold_pos,
        )
    else:
        splits = template_disjoint_splits(
            labels, template_ids, n_folds, cap_positives, seed,
            min_fold_pos=min_fold_pos,
        )
    if not splits:
        return np.full(n_layers, np.nan)

    # Optionally permute labels per fold to fit a shuffled-label baseline:
    # same train/test partitions as the real probe (eligibility decided on
    # real labels), only the labels seen by the LR are randomized. Used to
    # estimate the empirical chance floor for the figure.
    if shuffled_labels:
        rng_shuf = np.random.RandomState(seed + 7919)
        shuffled_splits = []
        for tr, te in splits:
            tr_lbl = labels[tr].copy(); rng_shuf.shuffle(tr_lbl)
            te_lbl = labels[te].copy(); rng_shuf.shuffle(te_lbl)
            shuffled_splits.append((tr, te, tr_lbl, te_lbl))
    else:
        shuffled_splits = None

    def _layer_acc(L: int) -> float:
        X = hidden[:, L, :].astype(np.float32)
        if shuffled_splits is not None:
            scores = [fit_one_layer(X[tr], tr_lbl, X[te], te_lbl)
                      for tr, te, tr_lbl, te_lbl in shuffled_splits]
        else:
            scores = [fit_one_layer(X[tr], labels[tr], X[te], labels[te])
                      for tr, te in splits]
        return float(np.mean(scores))

    accs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_layer_acc)(L) for L in range(n_layers)
    )
    return np.array(accs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    # Publication palette: surface tokens in warm hues (will-be-stripped),
    # structural / late signals in cool hues (will-be-built), narrative
    # fillers in graphite (the floor).
    "entity_name":      "#c2185b",   # deep magenta (proper names)
    "entity_other":     "#e64a19",   # burnt orange (common-noun entities)
    "number":           "#1565c0",   # deep blue (operand surface tokens)
    "op_presence":      "#6a1b9a",   # deep purple (formula structure)
    "op_cue":           "#2e7d32",   # forest green (arithmetic-cue words)
    "function_word":    "#616161",   # graphite (narrative floor)
    # Combined when --combine_narrative_floor is set.
    "narrative_filler": "#616161",   # graphite
    # Combined when --combine_entity is set.
    "entity":           "#c2185b",   # deep magenta (matches entity_name)
}
CATEGORY_LABELS = {
    "entity_name":      "Entity (proper names)",
    "entity_other":     "Entity (common nouns)",
    "number":           "Number tokens",
    "op_presence":      "Op-presence",
    "op_cue":           "Op-cue words",
    "function_word":    "Function words",
    "narrative_filler": "Narrative fillers",
    "entity":           "Entity tokens",
}
# Abstraction band shown as a gradient that fades in from L0 and reaches
# peak intensity at L22, encoding the gradual nature of the transition
# without committing to a hard left edge. Right edge at L22 matches
# template_similarity's "Template-structure emergence" marker; empirically
# the steepest single-step entity drop sits at L21→L22 in both gsm_symbolic
# and gsm_symbolic_pairs (cot/all and cot/correct), with the curve
# plateauing immediately after.
ABSTRACTION_BAND_START = 0
ABSTRACTION_BAND_END = 22
ABSTRACTION_BAND_COLOR = "#fdae61"
ABSTRACTION_BAND_MAX_ALPHA = 0.30


# Pretty-print maps for figure titles (consistent with run_template_similarity).
DATASET_PRETTY = {
    "gsm_symbolic":       "GSM-Symbolic",
    "gsm_symbolic_pairs": "GSM-Symbolic-Pairs",
}
MODE_PRETTY = {
    "direct":               "Direct",
    "cot":                  "CoT",
    "cot_pre_reasoning":    "CoT (pre-reasoning)",
    "question_last_token":  "Question last token",
    "question_span_mean":   "Question span mean",
}
CORRECTNESS_PRETTY = {
    "all":     "all instances",
    "correct": "correct only",
    "wrong":   "wrong only",
}


# (Legacy single-layer markers — no longer drawn. The abstraction stage is
# shown as the shaded ABSTRACTION_BAND defined above. Kept here as
# documentation of the prior layer-by-layer landmarks for cross-reference
# with template_similarity / op_multiset / DLA analyses.)
STAGE_BOUNDARIES = []


def cv_label(cv_mode: str) -> str:
    return {
        "template_disjoint": "template-disjoint CV",
        "within_template": "within-template CV",
        "paired": "paired CV",
        "paired_template_disjoint": "paired + template-disjoint CV",
    }.get(cv_mode, cv_mode)


def plot_results(
    entity_name_accs: np.ndarray,
    entity_other_accs: np.ndarray,
    op_cue_accs: np.ndarray,
    function_accs: np.ndarray,
    number_accs: np.ndarray,
    op_presence_accs: np.ndarray,
    entity_name_words: list,
    entity_other_words: list,
    op_cue_words: list,
    function_words: list,
    number_words: list,
    model_short: str,
    out_dir: Path,
    mode: str = "direct",
    cv_mode: str = "template_disjoint",
    skip_categories: tuple = (),
    combine_narrative_floor: bool = False,
    combine_entity: bool = False,
    dataset_name: str = "gsm_symbolic",
    correctness: str = "all",
    output_filename: str = "main.png",
    shuffled_accs_by_cat: dict | None = None,
    show_effect_size_brackets: bool = True,
):
    """Publication-style aggregated figure.

    Style mirrors run_template_similarity.plot_results: compact 12×4.8 axes,
    minimal suptitle, four stage-boundary vertical guides with rotated layer
    annotations, 95% bootstrap CI bands per category. Categories listed in
    `skip_categories` are hidden from the plot (used to draw a clean paired-
    entity-only figure where mixed-CV controls would be methodologically
    incoherent — see the gsm_symbolic vs gsm_symbolic_pairs split).

    When `combine_narrative_floor` is True, function_word and op_cue are
    merged into a single `narrative_filler` curve (concatenated per-word probe
    arrays → larger K, tighter bootstrap CI). The two are semantically
    distinct (syntactic glue vs arithmetic-cue content words) but both behave
    as floor controls — they sit in the .55–.65 plateau across all conditions
    and don't show any stage-aligned dynamics.

    When `combine_entity` is True, entity_name and entity_other are merged
    into a single `entity` curve. Use for the paired/entity-only figure where
    the "Entity tokens" message is what matters more than the name/common-noun
    distinction.
    """
    # Optionally fuse entity_name + entity_other into one combined entity
    # curve. Same bootstrap-over-K-target-probes contract.
    if combine_entity:
        if entity_name_accs.size and entity_other_accs.size:
            entity_accs = np.concatenate(
                [entity_name_accs, entity_other_accs], axis=1)
            entity_words = list(entity_name_words) + list(entity_other_words)
        elif entity_name_accs.size:
            entity_accs = entity_name_accs
            entity_words = list(entity_name_words)
        elif entity_other_accs.size:
            entity_accs = entity_other_accs
            entity_words = list(entity_other_words)
        else:
            entity_accs = entity_name_accs                  # empty placeholder
            entity_words = []
    else:
        entity_accs = np.full((0, 0), np.nan, dtype=np.float32)
        entity_words = []

    # Optionally fuse function_word + op_cue into one narrative-floor curve
    # by concatenating their per-word probe arrays. The bootstrap unit
    # (one word probe per sample) is preserved.
    if combine_narrative_floor:
        if function_accs.size and op_cue_accs.size:
            narrative_accs = np.concatenate([function_accs, op_cue_accs], axis=1)
            narrative_words = list(function_words) + list(op_cue_words)
        elif function_accs.size:
            narrative_accs = function_accs
            narrative_words = list(function_words)
        elif op_cue_accs.size:
            narrative_accs = op_cue_accs
            narrative_words = list(op_cue_words)
        else:
            narrative_accs = function_accs                  # empty placeholder
            narrative_words = []
    else:
        narrative_accs = np.full((0, 0), np.nan, dtype=np.float32)
        narrative_words = []

    # Plot order chosen so warm/surface curves render under cool/structural —
    # both readable when bands overlap. The narrative_filler / entity entries
    # only appear under their respective combine_* flags; otherwise we list
    # the constituents individually.
    if combine_entity:
        entity_block = [("entity", entity_accs, entity_words)]
    else:
        entity_block = [
            ("entity_name",   entity_name_accs,  entity_name_words),
            ("entity_other",  entity_other_accs, entity_other_words),
        ]
    if combine_narrative_floor:
        narrative_block = [("narrative_filler", narrative_accs, narrative_words)]
    else:
        narrative_block = [
            ("op_cue",        op_cue_accs,       op_cue_words),
            ("function_word", function_accs,     function_words),
        ]
    categories = (
        entity_block
        + [("number",      number_accs,       number_words)]
        + [("op_presence", op_presence_accs,  [])]
        + narrative_block
    )
    skip = set(skip_categories)
    categories = [(n, a, w) for n, a, w in categories
                  if n not in skip and a.size and a.shape[1] > 0]
    if not categories:
        raise ValueError("plot_results: nothing to plot (all categories empty "
                         "or skipped)")
    n_layers = categories[0][1].shape[0]
    layers = np.arange(n_layers)

    def bootstrap_ci(arr, n_boot=1000, ci=95.0, seed=0):
        """Per-layer mean and bootstrap CI over per-word probe accuracies."""
        n_layers, K = arr.shape
        if K == 0:
            nan = np.full(n_layers, np.nan)
            return nan, nan, nan
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, K, size=(n_boot, K))
        boot = np.nanmean(arr[:, idx], axis=2)
        mean = np.nanmean(arr, axis=1)
        lo = np.nanpercentile(boot, (100 - ci) / 2, axis=1)
        hi = np.nanpercentile(boot, 100 - (100 - ci) / 2, axis=1)
        return mean, lo, hi

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Aggregated probe accuracy per category ───────────────────────────────
    # No suptitle — figure metadata (model / dataset / mode / correctness)
    # belongs in the paper caption, not on the plot itself. The output path
    # carries the full context for browsing.
    fig, ax = plt.subplots(figsize=(12, 4.8))

    # Compute per-curve effect-size endpoints for option-D bracketed deltas.
    # L_start = first layer with reliable signal (skip the very-early embedding
    # layers where probes haven't activated). L_end = abstraction-band right
    # edge (L22).
    effect_size_start = 5
    effect_size_end = ABSTRACTION_BAND_END
    curve_means = {}
    for name, accs, _ in categories:
        mean, lo, hi = bootstrap_ci(accs, n_boot=1000, ci=95.0, seed=0)
        color = CATEGORY_COLORS[name]
        ax.fill_between(layers, lo, hi, color=color, alpha=0.18)
        ax.plot(layers, mean, color=color, lw=1.8,
                label=CATEGORY_LABELS[name])
        curve_means[name] = mean

    # Abstraction band: amber gradient that fades in from L0 and reaches
    # peak alpha at L22. The gradient encodes that the transition is gradual
    # — no hard left edge — while L22 (right edge, marked with the dashed
    # vertical line) is the empirical plateau onset and matches
    # template_similarity's "Template-structure emergence" boundary.
    band_hi = ABSTRACTION_BAND_END
    band_lo = ABSTRACTION_BAND_START
    stage_ticks: list[int] = []  # boundary layer ticks to add to x-axis
    if band_hi < n_layers:
        from matplotlib.colors import LinearSegmentedColormap, to_rgba
        base_rgb = to_rgba(ABSTRACTION_BAND_COLOR)[:3]
        cmap = LinearSegmentedColormap.from_list(
            "abstraction_band",
            [(*base_rgb, 0.0), (*base_rgb, ABSTRACTION_BAND_MAX_ALPHA)],
            N=128,
        )
        gradient = np.linspace(0.0, 1.0, 128).reshape(1, -1)
        ax.imshow(
            gradient,
            extent=(band_lo, band_hi, 0.0, 1.0),
            transform=ax.get_xaxis_transform(),
            aspect="auto",
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            interpolation="bilinear",
            zorder=0,
        )
        ax.axvline(band_hi, color="#333333", lw=0.9, alpha=0.55, ls="--")
        # Anchor the stage label in the upper region (axis-fraction ~0.95)
        # so it doesn't overlap with the probe curves; va="top" + rotation=90
        # means the label hangs down from y=0.95.
        ax.text(
            band_hi - 0.5, 0.95, "Abstraction",
            transform=ax.get_xaxis_transform(),
            ha="right", va="top", fontsize=7.5, color="#333333",
        )
        stage_ticks.append(band_hi)
    # Legacy per-layer markers (currently empty).
    for boundary, stage_label in STAGE_BOUNDARIES:
        if boundary >= n_layers:
            continue
        ax.axvline(boundary, color="#333333", lw=0.9, alpha=0.55, ls="--")
        ax.text(
            boundary, 0.95, stage_label,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", rotation=90, fontsize=7.5, color="#333333",
        )
        stage_ticks.append(boundary)

    # Per-curve endpoint annotations: mark the highest value reached in the
    # pre-L22 abstraction band and the value at L22. Replaces the older
    # L5→L22 dashed bracket + Δ annotation: peak + endpoint are easier to
    # read off than an integrated delta and don't presuppose L5 as a
    # baseline.
    # Skipped when there are >4 curves on the plot (e.g. supp) — labels
    # at the same x-position pile up and become unreadable.
    if (show_effect_size_brackets
            and effect_size_end < n_layers
            and len(curve_means) <= 4):
        for name, mean in curve_means.items():
            color = CATEGORY_COLORS[name]
            # Pre-L22 peak: argmax over L0..L22 (inclusive).
            pre_band = np.asarray(mean[: effect_size_end + 1])
            peak_layer = int(pre_band.argmax())
            peak_val = float(pre_band[peak_layer])
            l22_val = float(mean[effect_size_end])
            # Markers at peak and L22.
            ax.plot([peak_layer], [peak_val],
                    color=color, marker="^", markersize=6,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5)
            ax.plot([effect_size_end], [l22_val],
                    color=color, marker="o", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    zorder=5)
            # Pick label vertical offset based on local curve direction so
            # labels lean away from the line. For the L22 marker we look at
            # the next few layers (post-L22 trajectory); for the peak marker
            # we just push up (peak is by definition the local maximum so
            # the curve is below the marker on both sides).
            post_end = min(effect_size_end + 5, n_layers - 1)
            post_slope = float(mean[post_end]) - l22_val
            l22_dy = 10 if post_slope < 0 else -10
            l22_va = "bottom" if l22_dy > 0 else "top"
            label_bbox = dict(boxstyle="round,pad=0.15", fc="white",
                              ec="none", alpha=0.85)
            # L22 value label, vertical offset away from the line.
            ax.annotate(
                f"{l22_val:.2f}",
                xy=(effect_size_end, l22_val),
                xytext=(6, l22_dy), textcoords="offset points",
                color=color, fontsize=8.5, fontweight="bold",
                ha="left", va=l22_va, zorder=6, bbox=label_bbox,
            )
            # Peak value label, placed above the marker with white bbox.
            ax.annotate(
                f"{peak_val:.2f}",
                xy=(peak_layer, peak_val),
                xytext=(0, 9), textcoords="offset points",
                color=color, fontsize=8.0, fontweight="bold",
                ha="center", va="bottom", zorder=6, bbox=label_bbox,
            )

    # Optional empirical chance baseline from shuffled-label refits, pooled
    # across all populated per-prompt categories. Replaces the theoretical
    # 0.5 dotted line with a real probe-under-null measurement, which
    # accounts for any subtle imbalance / overfit in the eval setup.
    pooled_shuffled = None
    if shuffled_accs_by_cat is not None:
        cols = [a for a in shuffled_accs_by_cat.values()
                if a is not None and a.size and a.shape[1] > 0]
        if cols:
            pooled_shuffled = np.concatenate(cols, axis=1)
    if pooled_shuffled is not None:
        sh_mean, sh_lo, sh_hi = bootstrap_ci(pooled_shuffled)
        ax.fill_between(layers, sh_lo, sh_hi, color="#888888", alpha=0.15)
        ax.plot(layers, sh_mean, color="#888888", lw=1.2, ls=":",
                label="Shuffled labels (chance)")
        # Inline "chance" annotation on the curve itself so the figure is
        # readable without consulting the legend.
        ax.text(
            n_layers - 1, float(sh_mean[-1]), "chance",
            ha="right", va="bottom", fontsize=8, color="#666666",
        )
    else:
        ax.axhline(0.5, color="#888888", lw=0.7, ls=":", alpha=0.7,
                   label="Chance (0.5)")
        ax.text(
            n_layers - 1, 0.5, "chance",
            ha="right", va="bottom", fontsize=8, color="#666666",
        )
    ax.set_xlim(0, n_layers - 1)
    # Custom x-axis ticks: regular interval marks plus the stage-boundary
    # layers. The stage layers (e.g. 22) are bolded so the reader can read
    # them off the x-axis without needing an inline annotation. Drops
    # default-interval ticks that sit within 4 layers of a stage tick to
    # avoid overlap.
    interval = 20
    default_ticks = list(range(0, n_layers, interval))
    if (n_layers - 1) not in default_ticks:
        default_ticks.append(n_layers - 1)
    ticks = sorted(set(default_ticks + stage_ticks))
    # Drop default ticks that are too close to a stage tick.
    cleaned = [t for t in ticks
               if t in stage_ticks
               or all(abs(t - s) >= 4 for s in stage_ticks)]
    ax.set_xticks(cleaned)
    for tlabel in ax.get_xticklabels():
        try:
            v = int(tlabel.get_text())
        except (TypeError, ValueError):
            continue
        if v in stage_ticks:
            tlabel.set_fontweight("bold")
            tlabel.set_color("#333333")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe accuracy")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    main_legend = ax.legend(
        title=None,
        fontsize=9.5,
        title_fontsize=9.5,
        loc="upper right",
        frameon=True,
    )
    # Secondary legend explaining the triangle (pre-L22 peak) and circle
    # (L22 value) markers used on each curve.
    if (show_effect_size_brackets
            and effect_size_end < n_layers
            and len(curve_means) <= 4):
        from matplotlib.lines import Line2D
        marker_handles = [
            Line2D([0], [0], marker="^", color="#444444", lw=0,
                   markersize=6, markeredgecolor="white",
                   markeredgewidth=0.8, label="Pre-L22 peak"),
            Line2D([0], [0], marker="o", color="#444444", lw=0,
                   markersize=5, markeredgecolor="white",
                   markeredgewidth=0.8, label=f"L{effect_size_end} value"),
        ]
        marker_legend = ax.legend(
            handles=marker_handles,
            fontsize=8.5,
            loc="lower right",
            frameon=True,
        )
        ax.add_artist(main_legend)

    plt.tight_layout()
    agg_path = out_dir / output_filename
    plt.savefig(agg_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Aggregated plot saved to {agg_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(model_id: str, k_entity: int, k_op_cue: int, k_function: int,
        k_number: int, cap_positives: int,
        n_folds: int, n_jobs: int, seed: int, mode: str = "direct",
        correct_only: bool = False, wrong_only: bool = False,
        cv_mode: str = "template_disjoint",
        dataset_name: str = "gsm_symbolic",
        k_op_presence: int = 4,
        skip_categories: tuple = (),
        combine_narrative_floor: bool = False,
        combine_entity: bool = False,
        with_shuffled_baseline: bool = False,
        min_entity_count: int = 0,
        unpaired_cv: bool = False,
        min_fold_pos: int = 10,
        k_answer: int = 0):
    if correct_only and wrong_only:
        raise ValueError("--correct_only and --wrong_only are mutually exclusive")
    if cv_mode not in ("template_disjoint", "within_template"):
        raise ValueError(f"unknown cv_mode {cv_mode!r}")
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    correctness = "correct" if correct_only else ("wrong" if wrong_only else "all")
    # Public cv_mode is the user-facing label that drives the output path; the
    # *effective* cv_mode used by the inner pipeline is the paired-aware
    # variant when the dataset is paired (paired CV is meaningful only with
    # pair_id-aware negatives + template-OOD folds together). For
    # within_template the effective mode is identical.
    user_cv_mode = cv_mode
    cv_mode = (
        "paired_template_disjoint"
        if (user_cv_mode == "template_disjoint"
            and dataset_name == "gsm_symbolic_pairs"
            and not unpaired_cv)
        else user_cv_mode
    )
    # When the user forces unpaired CV on the paired dataset, tag the output
    # path so it doesn't overwrite the canonical paired run.
    if unpaired_cv and dataset_name == "gsm_symbolic_pairs":
        user_cv_mode = "template_disjoint_unpaired"
        print(f"\n--unpaired_cv: using template_disjoint CV on pairs data "
              f"(unpaired negatives, 6x more data than plain gsm_symbolic). "
              f"Output goes to {user_cv_mode}/.")
    # Output layout: <model>/<dataset>/<cv_mode>/<mode>/<correctness>/  using
    # the user-facing cv_mode (the dataset axis already distinguishes paired
    # vs unpaired template_disjoint runs).
    out_dir = OUT_DIR / model_short / dataset_name / user_cv_mode / mode / correctness
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cache ────────────────────────────────────────────────────────────
    # The all-instances cache is the single source of truth. The correctness
    # filter is applied in-memory below using the meta's correctness column,
    # which is determined by mode (see run_template_similarity._correctness_col).
    hs_path  = cache_path(model_short, correct_only=False,
                          dataset_name=dataset_name, mode=mode)
    meta_csv = meta_path(model_short, correct_only=False,
                         dataset_name=dataset_name, mode=mode)
    if not hs_path.exists():
        raise FileNotFoundError(f"Cached hidden states not found at {hs_path}")
    print(f"Loading hidden states: {hs_path}")
    hidden = np.load(hs_path)
    print(f"  shape: {hidden.shape}")

    df = load_probe_aligned_questions(model_id, meta_csv, dataset_name, mode)
    print(f"Aligned {len(df)} questions")
    if hidden.shape[0] != len(df):
        raise ValueError(
            f"Hidden-state/meta row mismatch: {hidden.shape[0]} hidden rows "
            f"vs {len(df)} aligned metadata rows. Rebuild the cache."
        )

    if correct_only or wrong_only:
        flag_name = "--correct_only" if correct_only else "--wrong_only"
        col = _correctness_col(mode)
        if col not in df.columns:
            raise ValueError(
                f"{flag_name} set but column {col!r} not in aligned meta. "
                f"Available correctness columns: "
                f"{[c for c in df.columns if 'correctness' in c]}"
            )
        correct_mask = _coerce_bool(df[col]).fillna(False).to_numpy(dtype=bool)

        # Pair-level filtering when the dataset is paired: keep a pair iff
        # BOTH halves satisfy the row-level filter. Avoids orphan halves that
        # would break paired-CV downstream and is the most defensible
        # correctness condition for paired data ("the model handled both the
        # original AND the swapped prompt correctly").
        if "pair_id" in df.columns:
            df_tmp = df.assign(_correct=correct_mask)
            target = True if correct_only else False
            pair_pass = (
                df_tmp.groupby("pair_id")["_correct"]
                .agg(lambda s: len(s) == 2 and (s == target).all())
            )
            keep_pair_ids = set(pair_pass[pair_pass].index)
            mask = df["pair_id"].isin(keep_pair_ids).to_numpy()
            kind = "complete pairs with both halves correct" if correct_only else (
                "complete pairs with both halves wrong"
            )
            print(f"  paired data detected — applying pair-level filter "
                  f"({kind}): {pair_pass.sum()}/{len(pair_pass)} pairs kept")
        else:
            mask = correct_mask if correct_only else ~correct_mask

        before = len(df)
        hidden = hidden[mask]
        df = df[mask].reset_index(drop=True)
        print(f"Applied {flag_name} filter (column={col}): "
              f"{before} -> {len(df)} instances")

    template_ids = df["original_id"].values.astype(np.int64)
    print(f"  {len(np.unique(template_ids))} unique templates")
    pair_ids_arr = df["pair_id"].values if "pair_id" in df.columns else None
    if pair_ids_arr is not None:
        print(f"  {len(set(pair_ids_arr))} unique pair_ids "
              f"(paired-CV available via --cv_mode paired)")
    elif cv_mode in ("paired", "paired_template_disjoint"):
        raise ValueError(
            f"--cv_mode {cv_mode} requires `pair_id` in the meta CSV but it's "
            "not present. Did you build the cache with a non-paired dataset?"
        )

    # Precompute an "originals" mask for paired_template_disjoint fallback:
    # the label==1 canonical original half of each (template, instance), with
    # duplicates across swap-role variants removed. Used to subset data when
    # a target word turns out to be template-static (e.g., function words,
    # op-cues) — paired-CV is degenerate for those because the same label
    # value appears in both halves of every pair, so we fall back to plain
    # template_disjoint on this deduped subset.
    originals_mask = None
    if (cv_mode == "paired_template_disjoint"
            and "label" in df.columns and "instance" in df.columns):
        m = (df["label"] == 1).to_numpy()
        # Within label==1 rows, dedup by (original_id, instance).
        keep_idx = (df.loc[m]
                    .drop_duplicates(subset=["original_id", "instance"],
                                     keep="first")
                    .index.to_numpy())
        originals_mask = np.zeros(len(df), dtype=bool)
        originals_mask[keep_idx] = True
        print(f"  originals fallback subset: {int(originals_mask.sum())} "
              "label==1 originals (used automatically for template-static "
              "probe targets that have no informative pairs)")

    # ── Word-level categorization (op-cue / function / number) ───────────────
    print("Extracting per-prompt word sets…")
    per_prompt = extract_words_per_prompt(df["question"].tolist())

    # ── Final-answer category (one value per row, derived from gold) ────────
    # Used to probe where the computed answer becomes linearly decodable from
    # the residual — the Stage 4 (computation) signature, parallel to the
    # number-token probes' Stage 3 signature on operand tokens.
    if "answer" in df.columns:
        gold_answers = df["answer"].tolist()
        for d, a in zip(per_prompt, gold_answers):
            if a is None:
                continue
            try:
                af = float(a)
                if not np.isfinite(af):
                    continue
                # Integer-valued answers are typed as "<int>"; non-integer
                # floats stay as the float repr. The probe target is the
                # full answer-value string, treated as a single "word".
                tok = str(int(af)) if af == int(af) else str(af)
            except (TypeError, ValueError):
                continue
            d["answer"].add(tok)
        n_with_answer = sum(1 for d in per_prompt if d["answer"])
        n_unique_answers = len({list(d["answer"])[0] for d in per_prompt if d["answer"]})
        print(f"  parsed gold answer for {n_with_answer}/{len(per_prompt)} rows; "
              f"{n_unique_answers} unique answer values")
    else:
        print("  no 'answer' column in dataset; answer-probe category will be empty")

    # ── Template-derived entity vocabulary ───────────────────────────────────
    print(f"Parsing entity vocabulary from {TEMPLATE_DIR.name}…")
    template_entity_vocab = parse_template_entities()
    universal_vocab = set().union(*template_entity_vocab.values())
    print(f"  parsed {len(template_entity_vocab)} templates; "
          f"{len(universal_vocab)} unique entity-role surface words")

    print("Overwriting heuristic entities with template-marked ones…")
    (entity_per_prompt, name_per_prompt, other_per_prompt,
     _, _, word_subcat) = build_entity_vocab_from_instances(
        df["question"].tolist(), template_ids, template_entity_vocab,
    )
    for d, ents, names, others in zip(
        per_prompt, entity_per_prompt, name_per_prompt, other_per_prompt
    ):
        d["entity"] = ents
        d["entity_name"] = names
        d["entity_other"] = others
    n_name_vocab = sum(1 for v in word_subcat.values() if v == "name")
    n_other_vocab = sum(1 for v in word_subcat.values() if v == "other")
    print(f"  entity sub-vocabulary: {n_name_vocab} name-role words, "
          f"{n_other_vocab} common-noun-role words")

    # For paired data, build a word -> primary role mapping so entity
    # selection can be stratified across roles (otherwise proper names
    # dominate the unstratified ranking and squeeze out animals/days/colors).
    word_to_role = None
    if "role" in df.columns:
        from collections import Counter as _C
        word_role_counts: dict = {}
        for i, d in enumerate(per_prompt):
            r = df["role"].iloc[i]
            for w in d["entity"]:
                word_role_counts.setdefault(w, _C())[r] += 1
        # Primary role = most-common role this word appears under.
        word_to_role = {w: cnt.most_common(1)[0][0]
                        for w, cnt in word_role_counts.items()}
        print(f"  built word->role map for {len(word_to_role)} entity words "
              f"across {len(set(word_to_role.values()))} roles "
              f"(role-stratified entity selection enabled)")

    # ── Pick balanced target words ───────────────────────────────────────────
    # Entity goes first; we then constrain every other category to match the
    # entity selection's template-count and prompt-count ranges so the
    # template-disjoint CV difficulty is equalized across categories.
    # min_count scales with dataset size so that `--correct_only` / `--wrong_only`
    # (which shrink the prompt pool) don't silently exclude all entities. The
    # floor scales with n_folds rather than a fixed 20, because the truly
    # binding constraint downstream is template_disjoint_splits requiring
    # ≥n_folds positive templates and ≥10 train positives per fold — anything
    # tighter than that just hides candidates that the CV would have handled
    # gracefully on its own.
    # Scaled to match the per-fold train floor: each fold needs ≥min_fold_pos
    # positives in its train pool. Default min_fold_pos=10 → chunk_floor=4
    # (preserves the legacy `n_folds * 4` floor); lowering min_fold_pos lets
    # rarer words become candidates so the relaxed per-fold gate can take effect.
    chunk_floor = max(min_fold_pos * 2 // 5, 2)
    adaptive_min_count = max(n_folds * chunk_floor, int(len(per_prompt) * 0.01))
    if min_entity_count > 0:
        adaptive_min_count = max(adaptive_min_count, min_entity_count)
        print(f"\n--min_entity_count={min_entity_count} applied; "
              f"effective min_count for word selection = {adaptive_min_count}")
    # Split the K-budget evenly between name and common-noun entity buckets.
    k_entity_name = k_entity // 2
    k_entity_other = k_entity - k_entity_name
    entity_name_words = select_balanced_words(
        per_prompt, template_ids, "entity_name", k=k_entity_name,
        min_count=adaptive_min_count,
        min_pos_templates=n_folds, min_neg_templates=n_folds,
        cv_mode=cv_mode,
        pair_ids=pair_ids_arr,
        word_to_group=word_to_role,
        originals_mask=originals_mask,
    )
    entity_other_words = select_balanced_words(
        per_prompt, template_ids, "entity_other", k=k_entity_other,
        min_count=adaptive_min_count,
        min_pos_templates=n_folds, min_neg_templates=n_folds,
        cv_mode=cv_mode,
        pair_ids=pair_ids_arr,
        word_to_group=word_to_role,
        originals_mask=originals_mask,
    )
    entity_words = entity_name_words + entity_other_words  # union, for band/cap calcs
    if not entity_words and k_entity > 0:
        raise RuntimeError(
            f"Entity selection returned 0 words (n_prompts={len(per_prompt)}, "
            f"min_count={adaptive_min_count}, n_folds={n_folds}). "
            "Without entities, distribution-matched control selection is "
            "undefined. Either relax filters, reduce --n_folds, or run on the "
            "unfiltered cache."
        )
    if not entity_words:
        print("k_entity=0: skipping entity probes and entity-derived control "
              "band-matching. Function/number/op-cue controls will use the "
              "default selection criteria.")

    # Recover the entity selection's template-count and prompt-count bands.
    def _word_template_count(word, category):
        return len({int(t) for t, d in zip(template_ids, per_prompt) if word in d[category]})

    entity_templates = [_word_template_count(w, "entity") for w, _ in entity_words]
    entity_counts = [c for _, c in entity_words]
    if entity_templates:
        # Constrain controls to the entity template-count band (±5 slack).
        # Prompt-count is NOT constrained because op_cue / function / number
        # words have naturally higher prompt counts; uniform `cap_positives`
        # equalizes per-fold training-set size separately.
        ent_template_range = (
            max(n_folds, min(entity_templates) - 5),
            max(entity_templates) + 5,
        )
    else:
        ent_template_range = None
    # Kept for backward compat; no longer used.
    ent_count_range = None
    print(f"\nEntity-derived selection band: template-count {ent_template_range}")

    # Op-cue is selected WITHOUT the entity template-range constraint —
    # op-cues like "each", "per", "than" appear in nearly every template, so
    # constraining to entity's narrow template-count band excludes everything.
    # Distribution matching is instead done per-word by subsampling positives
    # below (see op_cue_label_overrides).
    op_cue_words = select_balanced_words(
        per_prompt, template_ids, "op_cue", k=k_op_cue,
        min_count=adaptive_min_count,
        min_pos_templates=n_folds, min_neg_templates=n_folds,
        cv_mode=cv_mode,
        pair_ids=pair_ids_arr,
        originals_mask=originals_mask,
    )
    print(f"\nEntity-NAME targets ({len(entity_name_words)}):")
    for w, c in entity_name_words:
        print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")
    print(f"\nEntity-OTHER targets ({len(entity_other_words)}):")
    for w, c in entity_other_words:
        print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")
    print(f"\nOp-cue targets ({len(op_cue_words)}):")
    for w, c in op_cue_words:
        print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")

    # ── Pick balanced control words (distribution-matched to entity) ─────────
    function_words = select_balanced_words(
        per_prompt, template_ids, "function_word", k=k_function,
        min_count=adaptive_min_count,
        min_pos_templates=n_folds, min_neg_templates=n_folds,
        pos_template_range=ent_template_range,
        pos_count_range=ent_count_range,
        cv_mode=cv_mode,
        pair_ids=pair_ids_arr,
        originals_mask=originals_mask,
    )
    number_words = select_balanced_words(
        per_prompt, template_ids, "number", k=k_number,
        min_count=adaptive_min_count,
        min_pos_templates=n_folds, min_neg_templates=n_folds,
        pos_template_range=ent_template_range,
        pos_count_range=ent_count_range,
        cv_mode=cv_mode,
        pair_ids=pair_ids_arr,
        originals_mask=originals_mask,
    )
    if k_answer > 0:
        answer_words = select_balanced_words(
            per_prompt, template_ids, "answer", k=k_answer,
            min_count=adaptive_min_count,
            min_pos_templates=n_folds, min_neg_templates=n_folds,
            pos_template_range=ent_template_range,
            pos_count_range=ent_count_range,
            cv_mode=cv_mode,
            pair_ids=pair_ids_arr,
            originals_mask=originals_mask,
        )
    else:
        answer_words = []
    print(f"\nFunction-word controls ({len(function_words)}):")
    for w, c in function_words:
        print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")
    print(f"\nNumber-token controls ({len(number_words)}):")
    for w, c in number_words:
        print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")
    if answer_words:
        print(f"\nAnswer-value targets ({len(answer_words)}):")
        for w, c in answer_words:
            print(f"  '{w}': {c} prompts ({100*c/len(per_prompt):.1f}%)")

    # ── Equalize per-fold training-set size across categories ────────────────
    # cap_positives caps train_pos and train_neg per fold. The smallest entity
    # has only ~entity_min_count positive prompts; with template-disjoint CV,
    # ~(n_folds-1)/n_folds of those land in any training fold. We cap every
    # category at that number so all probes see the same training-set size and
    # accuracy differences attribute to encoding, not data volume.
    if entity_counts and cv_mode == "template_disjoint":
        # Floor at 10: matches the minimum train-set size required by
        # `template_disjoint_splits`, so the equalization doesn't make every
        # probe fail when entity supply is small.
        equalized_cap = max(10, int(min(entity_counts) * (n_folds - 1) / n_folds))
        if equalized_cap < cap_positives:
            print(f"\nEqualizing cap_positives: {cap_positives} -> {equalized_cap} "
                  f"(derived from smallest entity = {min(entity_counts)} prompts, "
                  f"n_folds={n_folds})")
            cap_positives = equalized_cap
    elif cv_mode == "within_template":
        # Within-template CV pools train data across many templates (one
        # train fold from each variable template), so the per-fold train
        # supply scales with #variable_templates × per-template instances,
        # not with the smallest word's total count. Don't auto-cap — leave
        # the user-provided cap_positives in place.
        print(f"\ncv_mode=within_template: keeping cap_positives={cap_positives} "
              f"(no entity-derived equalization).")
    elif cv_mode in ("paired", "paired_template_disjoint"):
        # Paired CV: train set has one pos + one neg per informative pair,
        # so size = #informative_pairs × 2. cap_positives caps the number of
        # train PAIRS per fold (not raw positives), since pos and neg are
        # 1:1 by construction. Don't equalize from entity_counts (different
        # math); leave user-provided value as the per-fold pair cap.
        print(f"\ncv_mode={cv_mode}: keeping cap_positives={cap_positives} "
              f"(interpreted as per-fold train-pair cap; no equalization).")

    # ── Train probes ─────────────────────────────────────────────────────────
    n_layers = hidden.shape[1]

    def _train_category(words, category, label_overrides=None, shuffled=False):
        desc = category if not shuffled else f"{category} (shuffled)"
        accs = np.full((n_layers, len(words)), np.nan, dtype=np.float32)
        for ti, (word, _) in enumerate(tqdm(words, desc=desc)):
            if label_overrides is not None and word in label_overrides:
                labels = label_overrides[word]
            else:
                labels = np.array(
                    [1 if word in d[category] else 0 for d in per_prompt],
                    dtype=np.int32,
                )
            n_pos_t = len(set(template_ids[labels == 1].tolist()))
            n_neg_t = len(set(template_ids[labels == 0].tolist())
                          - set(template_ids[labels == 1].tolist()))
            n_pos = int((labels == 1).sum())
            use_fallback = False     # set True for template-static targets
            if cv_mode in ("paired", "paired_template_disjoint"):
                # Count informative pairs for this word: pairs with exactly
                # one half positive (the other being the swap).
                if pair_ids_arr is None:
                    print(f"  '{word}': skipped (cv_mode={cv_mode} requires pair_id "
                          f"in meta CSV; not present)")
                    continue
                pair_pos = {}
                pair_template = {}
                for i, pid in enumerate(pair_ids_arr):
                    pair_pos.setdefault(pid, []).append(int(labels[i]))
                    pair_template.setdefault(pid, int(template_ids[i]))
                inform_pids = [pid for pid, ys in pair_pos.items()
                                if len(ys) == 2 and sum(ys) == 1]
                n_inform = len(inform_pids)
                if n_inform < n_folds:
                    # Template-static target (function words / op-cues that
                    # appear in every pair's both halves). Under
                    # paired_template_disjoint, fall back to plain
                    # template_disjoint on the label==1 originals subset so
                    # these targets can still be probed under a consistent
                    # template-OOD eval. paired (without _td) still skips —
                    # the fallback only applies when the user opted into the
                    # template-disjoint contract.
                    if (cv_mode == "paired_template_disjoint"
                            and originals_mask is not None):
                        use_fallback = True
                        print(f"  '{word}': informative_pairs={n_inform}<{n_folds} "
                              "(template-static target); falling back to "
                              "template_disjoint on label==1 originals "
                              f"(K={int(originals_mask.sum())})")
                    else:
                        print(f"  '{word}': skipped (informative_pairs="
                              f"{n_inform}, need ≥{n_folds})")
                        continue
                elif cv_mode == "paired_template_disjoint":
                    n_inform_t = len({pair_template[pid] for pid in inform_pids})
                    if n_inform_t < n_folds:
                        if originals_mask is not None:
                            use_fallback = True
                            print(f"  '{word}': informative_pair_templates="
                                  f"{n_inform_t}<{n_folds}; falling back to "
                                  "template_disjoint on label==1 originals")
                        else:
                            print(f"  '{word}': skipped "
                                  f"(informative_pair_templates={n_inform_t}, "
                                  f"need ≥{n_folds})")
                            continue
                    else:
                        print(f"  '{word}': pos={n_pos}, informative_pairs="
                              f"{n_inform}, templates={n_inform_t}")
                else:
                    print(f"  '{word}': pos={n_pos}, informative_pairs={n_inform}")
            elif cv_mode == "within_template":
                n_var = _count_variable_templates(labels, template_ids)
                if n_var < n_folds:
                    print(f"  '{word}': skipped (variable_templates={n_var}, "
                          f"need ≥{n_folds})")
                    continue
                print(f"  '{word}': pos={n_pos}, variable_templates={n_var}")
            else:
                if n_pos_t < n_folds or n_neg_t < n_folds:
                    print(f"  '{word}': skipped (pos_templates={n_pos_t}, "
                          f"neg_templates={n_neg_t}, need ≥{n_folds} of each)")
                    continue
                print(f"  '{word}': pos={n_pos}, pos_templates={n_pos_t}")

            if use_fallback:
                accs[:, ti] = train_probes_for_word(
                    hidden[originals_mask], word,
                    labels[originals_mask], template_ids[originals_mask],
                    cap_positives, n_folds, seed, n_jobs,
                    cv_mode="template_disjoint",
                    pair_ids=None,
                    shuffled_labels=shuffled,
                    min_fold_pos=min_fold_pos,
                )
            else:
                accs[:, ti] = train_probes_for_word(
                    hidden, word, labels, template_ids,
                    cap_positives, n_folds, seed, n_jobs,
                    cv_mode=cv_mode,
                    pair_ids=pair_ids_arr,
                    shuffled_labels=shuffled,
                    min_fold_pos=min_fold_pos,
                )
        return accs

    # Build per-word subsampled labels for op-cues so each op-cue word's
    # positive distribution matches a randomly-paired entity word's
    # (template_count, prompt_count). Uses sampling-with-replacement over the
    # entity pool when there are more op-cues than entities.
    op_cue_label_overrides = {}
    if (op_cue_words and entity_words
            and cv_mode not in ("paired", "paired_template_disjoint")):
        rng = np.random.RandomState(seed)
        ent_pairs = list(zip(entity_templates, entity_counts))
        print("\nSubsampling op-cue positives to match entity distribution:")
        for i, (word, count) in enumerate(op_cue_words):
            target_n_templates, target_n_positives = ent_pairs[i % len(ent_pairs)]
            full_labels = np.array(
                [1 if word in d["op_cue"] else 0 for d in per_prompt],
                dtype=np.int32,
            )
            full_pos = int(full_labels.sum())
            full_pos_t = len(set(template_ids[full_labels == 1].tolist()))
            sub_labels = subsample_to_match_distribution(
                full_labels, template_ids,
                target_n_templates=target_n_templates,
                target_n_positives=target_n_positives,
                seed=seed + i,
            )
            sub_pos = int(sub_labels.sum())
            sub_pos_t = len(set(template_ids[sub_labels == 1].tolist()))
            print(f"  '{word}': {full_pos} prompts / {full_pos_t} templates "
                  f"-> {sub_pos} / {sub_pos_t} "
                  f"(target {target_n_positives} / {target_n_templates})")
            op_cue_label_overrides[word] = sub_labels
    elif op_cue_words and cv_mode in ("paired", "paired_template_disjoint"):
        print(f"\ncv_mode={cv_mode}: not subsampling op-cue positives; paired CV "
              "requires labels to stay matched within pair_id.")

    cv_desc = cv_label(cv_mode)
    print(f"\nTraining entity-NAME probes ({cv_desc})…")
    entity_name_accs = _train_category(entity_name_words, "entity_name")
    print(f"\nTraining entity-OTHER probes ({cv_desc})…")
    entity_other_accs = _train_category(entity_other_words, "entity_other")
    # Only call op-cues "distribution-matched" when subsampling actually fired;
    # paired modes skip it (labels must stay aligned within pair_id) so the
    # label would be misleading there.
    op_cue_match_note = (
        ", distribution-matched"
        if op_cue_label_overrides
        else " (no entity-matching: paired-CV requires labels to stay "
             "aligned within pair_id)"
    )
    print(f"\nTraining op-cue probes ({cv_desc}{op_cue_match_note})…")
    op_cue_accs = _train_category(op_cue_words, "op_cue",
                                   label_overrides=op_cue_label_overrides)
    print(f"\nTraining function-word probes ({cv_desc})…")
    function_accs = _train_category(function_words, "function_word")
    print(f"\nTraining number-token probes ({cv_desc})…")
    number_accs = _train_category(number_words, "number")
    if answer_words:
        print(f"\nTraining answer-value probes ({cv_desc})…")
        answer_accs = _train_category(answer_words, "answer")
    else:
        answer_accs = np.full((n_layers, 0), np.nan, dtype=np.float32)

    # ── Shuffled-label chance baseline ──────────────────────────────────────
    # Optional: refit each non-empty per-prompt category with permuted labels
    # to estimate the empirical chance floor under the exact CV setup. Used
    # downstream as a "Shuffled labels" reference curve in the figure (pooled
    # across categories). Skipped if disabled — adds ~equal compute to the
    # main probe pass.
    if with_shuffled_baseline:
        print(f"\nFitting shuffled-label baseline ({cv_desc})…")
        entity_name_shuffled  = _train_category(entity_name_words,  "entity_name",  shuffled=True)
        entity_other_shuffled = _train_category(entity_other_words, "entity_other", shuffled=True)
        op_cue_shuffled       = _train_category(op_cue_words,       "op_cue",       label_overrides=op_cue_label_overrides, shuffled=True)
        function_shuffled     = _train_category(function_words,     "function_word", shuffled=True)
        number_shuffled       = _train_category(number_words,       "number",       shuffled=True)
        if answer_words:
            answer_shuffled   = _train_category(answer_words,        "answer",       shuffled=True)
        else:
            answer_shuffled   = np.full((n_layers, 0), np.nan, dtype=np.float32)
    else:
        empty = np.full((n_layers, 0), np.nan, dtype=np.float32)
        entity_name_shuffled = entity_other_shuffled = op_cue_shuffled = \
            function_shuffled = number_shuffled = answer_shuffled = empty

    # ── op_presence (per-template formula structure: +, −, ×, ÷) ─────────────
    # Template-level labels: one boolean per (template, op). Under paired CV
    # both halves of a pair share the label → degenerate; we subset to the
    # label==1 originals (`originals_mask`) and use plain template_disjoint
    # CV — same mechanism as the function-word / op-cue fallback elsewhere.
    if k_op_presence > 0:
        ops_to_fit = OPS[:k_op_presence]
        print(f"\nTraining op_presence probes (template_disjoint on "
              f"label==1 originals) — {len(ops_to_fit)} operators: {ops_to_fit}")
        try:
            template_to_ops = per_template_op_presence(model_id)
        except FileNotFoundError as exc:
            print(f"  op_presence: skipped — {exc}")
            template_to_ops = None

        if template_to_ops is None:
            op_presence_accs = np.full((n_layers, 0), np.nan, dtype=np.float32)
            op_presence_words: list = []
        else:
            if originals_mask is not None:
                op_hidden = hidden[originals_mask]
                op_templates = template_ids[originals_mask]
            else:
                op_hidden = hidden
                op_templates = template_ids
            op_presence_accs = np.full((n_layers, len(ops_to_fit)),
                                       np.nan, dtype=np.float32)
            op_presence_words = []
            for i, op in enumerate(ops_to_fit):
                op_labels = np.array(
                    [int(template_to_ops.get(int(t), {}).get(op, False))
                     for t in op_templates],
                    dtype=np.int32,
                )
                n_pos = int(op_labels.sum())
                # Labels here are template-level (every row of a template
                # shares a single op-presence boolean), so positive/negative
                # template sets are disjoint by construction. Take the full
                # set-difference (vs subtracting `pos_t` as a count) to be
                # safe even if that invariant is ever violated.
                pos_templates_set = set(op_templates[op_labels == 1].tolist())
                neg_templates_set = (set(op_templates[op_labels == 0].tolist())
                                     - pos_templates_set)
                pos_t = len(pos_templates_set)
                neg_t = len(neg_templates_set)
                if pos_t < n_folds or neg_t < n_folds:
                    print(f"  op '{op}': skipped (pos_templates={pos_t}, "
                          f"neg_templates={neg_t}, need ≥{n_folds} of each)")
                    continue
                print(f"  op '{op}': pos={n_pos}, pos_templates={pos_t}, "
                      f"neg_templates={neg_t}")
                op_presence_accs[:, i] = train_probes_for_word(
                    op_hidden, f"has_{OP_NAMES[op]}",
                    op_labels, op_templates,
                    cap_positives, n_folds, seed, n_jobs,
                    cv_mode="template_disjoint",
                    pair_ids=None,
                    min_fold_pos=min_fold_pos,
                )
                op_presence_words.append((f"has_{op}", n_pos))
    else:
        op_presence_accs = np.full((n_layers, 0), np.nan, dtype=np.float32)
        op_presence_words = []

    # ── Save ──────────────────────────────────────────────────────────────────
    # Partial-run merge: if the user re-ran with only a subset of categories
    # populated (e.g., K_ENTITY=0 K_NUMBER=0 K_FUNCTION=30 to add function-word
    # probes without recomputing entities), preserve the prior arrays + word
    # lists for the categories that weren't fit this round.
    new_arrays = {
        "entity_name_accs":  entity_name_accs,
        "entity_other_accs": entity_other_accs,
        "op_cue_accs":       op_cue_accs,
        "function_accs":     function_accs,
        "number_accs":       number_accs,
        "answer_accs":       answer_accs,
        "op_presence_accs":  op_presence_accs,
        # Shuffled-label baseline (empty when --with_shuffled_baseline not set).
        "entity_name_accs_shuffled":  entity_name_shuffled,
        "entity_other_accs_shuffled": entity_other_shuffled,
        "op_cue_accs_shuffled":       op_cue_shuffled,
        "function_accs_shuffled":     function_shuffled,
        "number_accs_shuffled":       number_shuffled,
        "answer_accs_shuffled":       answer_shuffled,
    }
    new_words = {
        "entity_name_words":   entity_name_words,
        "entity_other_words":  entity_other_words,
        "op_cue_words":        op_cue_words,
        "function_words":      function_words,
        "number_words":        number_words,
        "answer_words":        answer_words,
        "op_presence_words":   op_presence_words,
    }
    npz_path = out_dir / "presence_probe.npz"
    meta_json = out_dir / "presence_probe_meta.json"
    if npz_path.exists() and meta_json.exists():
        try:
            old_arrays = dict(np.load(npz_path))
            with open(meta_json) as f:
                old_meta = json.load(f)
            # The saved cv_mode is the user-facing/path value (e.g. the
            # renamed "template_disjoint_unpaired" when --unpaired_cv is set
            # on pairs data); compare against that, not the inner cv_mode
            # used by the training pipeline (which has been coerced to e.g.
            # "paired_template_disjoint" or kept as "template_disjoint").
            old_cv_mode = old_meta.get("cv_mode", user_cv_mode)
            old_mode = old_meta.get("mode", mode)
            if old_cv_mode != user_cv_mode or old_mode != mode:
                print(f"Existing {npz_path.name} has mode={old_mode}/"
                      f"cv_mode={old_cv_mode}, this run has mode={mode}/"
                      f"cv_mode={user_cv_mode}. Refusing to merge — overwriting "
                      "entirely instead.")
            else:
                preserved = []
                for key, arr in new_arrays.items():
                    old = old_arrays.get(key)
                    if (arr.shape[1] == 0 and old is not None
                            and old.shape[0] == arr.shape[0]
                            and old.shape[1] > 0):
                        new_arrays[key] = old
                        word_key = key.replace("_accs", "_words")
                        if word_key in old_meta:
                            new_words[word_key] = [
                                (d["word"], d["count"])
                                for d in old_meta[word_key]
                            ]
                        preserved.append(f"{key}({old.shape[1]})")
                if preserved:
                    print(f"Partial run: preserved {', '.join(preserved)} from "
                          f"existing {npz_path.name}; this run only refits the "
                          "categories you supplied K>0 for.")
        except (KeyError, ValueError, OSError) as e:
            print(f"Could not merge with existing {npz_path.name} ({e}); "
                  "overwriting.")
    # Reassign the local names so the summary / plot at the bottom of run()
    # reflect the merged arrays.
    entity_name_accs   = new_arrays["entity_name_accs"]
    entity_other_accs  = new_arrays["entity_other_accs"]
    op_cue_accs        = new_arrays["op_cue_accs"]
    function_accs      = new_arrays["function_accs"]
    number_accs        = new_arrays["number_accs"]
    answer_accs        = new_arrays["answer_accs"]
    op_presence_accs   = new_arrays["op_presence_accs"]
    entity_name_words  = new_words["entity_name_words"]
    entity_other_words = new_words["entity_other_words"]
    op_cue_words       = new_words["op_cue_words"]
    function_words     = new_words["function_words"]
    number_words       = new_words["number_words"]
    answer_words       = new_words["answer_words"]
    op_presence_words  = new_words["op_presence_words"]
    # Pull shuffled arrays out of new_arrays (preserved through partial-merge
    # if they were on disk from a prior run).
    entity_name_shuffled  = new_arrays["entity_name_accs_shuffled"]
    entity_other_shuffled = new_arrays["entity_other_accs_shuffled"]
    op_cue_shuffled       = new_arrays["op_cue_accs_shuffled"]
    function_shuffled     = new_arrays["function_accs_shuffled"]
    number_shuffled       = new_arrays["number_accs_shuffled"]
    answer_shuffled       = new_arrays["answer_accs_shuffled"]
    np.savez(
        npz_path,
        entity_name_accs=entity_name_accs,
        entity_other_accs=entity_other_accs,
        op_cue_accs=op_cue_accs,
        function_accs=function_accs,
        number_accs=number_accs,
        answer_accs=answer_accs,
        op_presence_accs=op_presence_accs,
        entity_name_accs_above_chance=entity_name_accs - 0.5,
        entity_other_accs_above_chance=entity_other_accs - 0.5,
        op_cue_accs_above_chance=op_cue_accs - 0.5,
        function_accs_above_chance=function_accs - 0.5,
        number_accs_above_chance=number_accs - 0.5,
        answer_accs_above_chance=answer_accs - 0.5,
        op_presence_accs_above_chance=op_presence_accs - 0.5,
        entity_name_accs_shuffled=entity_name_shuffled,
        entity_other_accs_shuffled=entity_other_shuffled,
        op_cue_accs_shuffled=op_cue_shuffled,
        function_accs_shuffled=function_shuffled,
        number_accs_shuffled=number_shuffled,
        answer_accs_shuffled=answer_shuffled,
    )
    with open(meta_json, "w") as f:
        readout = {
            "within_template": "within-template stratified-KFold linear probe accuracy",
            "paired":          "paired-fold linear probe accuracy "
                               "(matched pos/neg pair_id holdout)",
            "paired_template_disjoint":
                "paired + template-disjoint linear probe accuracy "
                "(matched pos/neg pair_id holdout, templates held out; "
                "template-static targets such as function words automatically "
                "fall back to template_disjoint on the label==1 originals subset)",
        }.get(cv_mode, "template-disjoint linear probe accuracy")
        json.dump({
            "readout_name":       readout,
            "chance_level":       0.5,
            "entity_name_words":   [{"word": w, "count": c} for w, c in entity_name_words],
            "entity_other_words":  [{"word": w, "count": c} for w, c in entity_other_words],
            "op_cue_words":        [{"word": w, "count": c} for w, c in op_cue_words],
            "function_words":      [{"word": w, "count": c} for w, c in function_words],
            "number_words":        [{"word": w, "count": c} for w, c in number_words],
            "answer_words":        [{"word": w, "count": c} for w, c in answer_words],
            "op_presence_words":   [{"word": w, "count": c} for w, c in op_presence_words],
            "n_prompts":          len(per_prompt),
            "n_layers":           n_layers,
            "cap_positives":      cap_positives,
            "n_folds":            n_folds,
            "min_fold_pos":       min_fold_pos,
            "mode":               mode,
            "cv_mode":            user_cv_mode,
            "cv_mode_effective":  cv_mode,
        }, f, indent=2)
    print(f"\nSaved to {out_dir}")

    # ── Summary ───────────────────────────────────────────────────────────────
    def _layer_mean(arr):
        if arr.shape[1] == 0:
            return np.full(arr.shape[0], np.nan)
        return np.nanmean(arr, axis=1)

    name_mean    = _layer_mean(entity_name_accs)
    other_mean   = _layer_mean(entity_other_accs)
    op_mean      = _layer_mean(op_cue_accs)
    fn_mean      = _layer_mean(function_accs)
    num_mean     = _layer_mean(number_accs)
    op_pres_mean = _layer_mean(op_presence_accs)
    print("\nPer-layer mean probe accuracy:")
    print(f"{'L':>4}  {'ent-name':>9}  {'ent-other':>10}  {'op_cue':>8}  "
          f"{'function':>10}  {'number':>8}  {'op_pres':>8}")
    for L in [0, 5, 10, 15, 20, 21, 24, 25, 30, 35, 38, 39, 40, 50, 70, 79, 80]:
        if L >= n_layers:
            continue
        print(f"{L:>4}  {name_mean[L]:>9.4f}  {other_mean[L]:>10.4f}  "
              f"{op_mean[L]:>8.4f}  {fn_mean[L]:>10.4f}  "
              f"{num_mean[L]:>8.4f}  {op_pres_mean[L]:>8.4f}")

    shuffled_by_cat = {
        "entity_name":   entity_name_shuffled,
        "entity_other":  entity_other_shuffled,
        "op_cue":        op_cue_shuffled,
        "function_word": function_shuffled,
        "number":        number_shuffled,
    }
    plot_results(
        entity_name_accs, entity_other_accs,
        op_cue_accs, function_accs, number_accs,
        op_presence_accs,
        entity_name_words, entity_other_words,
        op_cue_words, function_words, number_words,
        model_short, out_dir, mode=mode, cv_mode=user_cv_mode,
        skip_categories=skip_categories,
        combine_narrative_floor=combine_narrative_floor,
        combine_entity=combine_entity,
        dataset_name=dataset_name,
        correctness=correctness,
        shuffled_accs_by_cat=shuffled_by_cat,
    )


def plot_only(model_id: str, mode: str = "direct",
              correct_only: bool = False, wrong_only: bool = False,
              cv_mode: str = "template_disjoint",
              dataset_name: str = "gsm_symbolic",
              skip_categories: tuple = (),
              combine_narrative_floor: bool = False,
              combine_entity: bool = False):
    if correct_only and wrong_only:
        raise ValueError("--correct_only and --wrong_only are mutually exclusive")
    model_short = MODEL_NAME_MAP.get(model_id, model_id.split("/")[-1])
    correctness = "correct" if correct_only else ("wrong" if wrong_only else "all")
    out_dir = OUT_DIR / model_short / dataset_name / cv_mode / mode / correctness
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path  = out_dir / "presence_probe.npz"
    meta_json = out_dir / "presence_probe_meta.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"No results at {npz_path}. Run without --plot_only first.")
    arr = np.load(npz_path)
    with open(meta_json) as f:
        meta = json.load(f)
    n_layers = arr["entity_name_accs"].shape[0]
    plot_results(
        arr["entity_name_accs"], arr["entity_other_accs"],
        arr["op_cue_accs"], arr["function_accs"], arr["number_accs"],
        arr["op_presence_accs"]
            if "op_presence_accs" in arr.files
            else np.full((n_layers, 0), np.nan, dtype=np.float32),
        [(d["word"], d["count"]) for d in meta["entity_name_words"]],
        [(d["word"], d["count"]) for d in meta["entity_other_words"]],
        [(d["word"], d["count"]) for d in meta["op_cue_words"]],
        [(d["word"], d["count"]) for d in meta["function_words"]],
        [(d["word"], d["count"]) for d in meta["number_words"]],
        model_short, out_dir, mode=mode, cv_mode=cv_mode,
        skip_categories=skip_categories,
        combine_narrative_floor=combine_narrative_floor,
        combine_entity=combine_entity,
        dataset_name=dataset_name,
        correctness=correctness,
        shuffled_accs_by_cat={
            "entity_name":   arr["entity_name_accs_shuffled"]   if "entity_name_accs_shuffled"   in arr.files else None,
            "entity_other":  arr["entity_other_accs_shuffled"]  if "entity_other_accs_shuffled"  in arr.files else None,
            "op_cue":        arr["op_cue_accs_shuffled"]        if "op_cue_accs_shuffled"        in arr.files else None,
            "function_word": arr["function_accs_shuffled"]      if "function_accs_shuffled"      in arr.files else None,
            "number":        arr["number_accs_shuffled"]        if "number_accs_shuffled"        in arr.files else None,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--k_entity", type=int, default=10)
    parser.add_argument("--k_op_cue", type=int, default=10)
    parser.add_argument("--k_function", type=int, default=8,
                        help="Function-word controls (low-information floor).")
    parser.add_argument("--k_number", type=int, default=8,
                        help="Number-token controls (token-aligned ceiling).")
    parser.add_argument("--k_answer", type=int, default=0,
                        help="Final-answer probes: per-row gold answer is the "
                             "target. The probe asks 'is this row's answer X?' "
                             "for each of the top-K most-common answer values. "
                             "Accuracy rises where the computed answer becomes "
                             "linearly decodable from the residual — the Stage "
                             "4 (computation) signature. 0 = skip.")
    parser.add_argument("--k_op_presence", type=int, default=4,
                        help=f"Number of formula-op probes (one per operator "
                             f"in {OPS}). Max 4; set 0 to skip the category.")
    parser.add_argument("--cap_positives", type=int, default=300,
                        help="Max positive (and negative) examples per probe.")
    parser.add_argument("--min_entity_count", type=int, default=0,
                        help="Override floor on per-word prompt count when "
                             "selecting entity words. Useful for unpaired "
                             "datasets where the auto-equalization caps "
                             "cap_positives down to the rarest entity's "
                             "count. Set higher (e.g. 150) to drop rare "
                             "entities and let cap_positives take effect.")
    parser.add_argument("--unpaired_cv", action="store_true",
                        help="On --dataset gsm_symbolic_pairs, use plain "
                             "template_disjoint CV instead of the auto-coerced "
                             "paired_template_disjoint. Lets the unpaired "
                             "probe inherit the pairs dataset's ~6x larger "
                             "per-word prompt counts. Output is tagged "
                             "'template_disjoint_unpaired' to avoid clobbering "
                             "the canonical paired results.")
    parser.add_argument("--n_folds", type=int, default=3)
    parser.add_argument("--min_fold_pos", type=int, default=10,
                        help="Per-fold floor on positive examples in the train "
                             "pool (test floor derived as max(min_fold_pos//2, "
                             "2)). Default 10 matches the legacy CV gate; lower "
                             "to 4 or 5 to admit small slices (e.g. cot wrong on "
                             "noop_clean has ~280 prompts in 23 templates and "
                             "needs a relaxed gate to keep any entity probe). "
                             "Also scales the candidate-selection floor via "
                             "adaptive_min_count, so words at the new fold-pos "
                             "threshold actually pass selection.")
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["direct", "cot", "cot_pre_reasoning", "question_last_token",
                 "question_span_mean"],
        default="direct",
        help="direct: residual at the answer-prefix token of the direct prompt. "
             "cot: residual at the answer-prefix token following the cached "
             "chain-of-thought reasoning. cot_pre_reasoning: cot-framed prompt "
             "but no reasoning yet, residual at the assistant-header end. "
             "question_last_token: residual at the last token of the question "
             "inside the chat-formatted prompt. question_span_mean: residual "
             "mean-pooled over all question tokens inside the chat-formatted "
             "prompt. "
             "Selects which template_similarity hidden-state cache to read.",
    )
    parser.add_argument(
        "--correct_only", action="store_true",
        help="Filter to only instances the model got right under the selected "
             "mode's correctness signal. Results are written to a separate "
             "`<mode>_correct/` output directory.",
    )
    parser.add_argument(
        "--wrong_only", action="store_true",
        help="Filter to only instances the model got wrong under the selected "
             "mode's correctness signal. Mutually exclusive with --correct_only. "
             "Results are written to a separate `<mode>_wrong/` output directory.",
    )
    parser.add_argument(
        "--dataset",
        default="gsm_symbolic",
        help="Which dataset's hidden-state cache to read. The cache root may "
             "be redirected per-dataset (see CACHE_ROOT_OVERRIDES in "
             "run_template_similarity.py); for large paired datasets, use a "
             "large external cache directory.",
    )
    parser.add_argument(
        "--skip_categories", nargs="*", default=[],
        choices=list(CATEGORY_COLORS.keys()),
        help="Categories to hide from the aggregated plot. Useful for drawing "
             "a methodologically pristine paired figure that excludes "
             "controls that fall back to template_disjoint on originals "
             "(function_word, op_cue, op_presence). Has no effect on training "
             "— probes are still fit and written to the npz.",
    )
    parser.add_argument(
        "--combine_narrative_floor", action="store_true",
        help="Merge function_word and op_cue into a single 'narrative_filler' "
             "curve in the plot. Reduces visual clutter and tightens the "
             "bootstrap CI (K ≈ 50). Plot-time only — the npz still stores "
             "both categories separately for supplementary analysis.",
    )
    parser.add_argument(
        "--combine_entity", action="store_true",
        help="Merge entity_name and entity_other into a single 'entity' "
             "curve in the plot. Useful for the paired/entity-only figure "
             "where the headline message is 'surface entity tokens lose "
             "decodability' rather than the name vs common-noun split. "
             "Plot-time only — npz keeps the two subcategories separate.",
    )
    parser.add_argument(
        "--with_shuffled_baseline", action="store_true",
        help="After fitting the real probes, refit each per-prompt category "
             "with permuted labels (same train/test partitions, shuffled "
             "labels) to estimate the empirical chance floor. Adds roughly "
             "equal compute to the main pass. Results are saved as separate "
             "*_accs_shuffled arrays in the npz and pooled into a single "
             "'Shuffled labels (chance)' curve on the figure (replaces the "
             "theoretical dotted line at 0.5).",
    )
    parser.add_argument(
        "--cv_mode",
        choices=["template_disjoint", "within_template"],
        default="template_disjoint",
        help="template_disjoint (default): templates in test folds are disjoint "
             "from training. On paired datasets (gsm_symbolic_pairs) this "
             "automatically uses matched-pair negatives in addition (formerly "
             "`paired_template_disjoint`). Template-static probe targets "
             "(function words, op-cues, op-presence) fall back to plain "
             "template_disjoint on the label==1 originals subset since "
             "paired-CV is degenerate for them. within_template: train and "
             "test on different instances of the *same* variable templates; "
             "isolates entity encoding from template structure and admits "
             "non-name entities (animals, objects, days, kin terms).",
    )
    args = parser.parse_args()

    skip_categories = tuple(args.skip_categories)

    def _do_run():
        run(args.model_id, args.k_entity, args.k_op_cue,
            args.k_function, args.k_number,
            args.cap_positives, args.n_folds, args.n_jobs, args.seed,
            mode=args.mode, correct_only=args.correct_only,
            wrong_only=args.wrong_only, cv_mode=args.cv_mode,
            dataset_name=args.dataset, k_op_presence=args.k_op_presence,
            skip_categories=skip_categories,
            combine_narrative_floor=args.combine_narrative_floor,
            combine_entity=args.combine_entity,
            with_shuffled_baseline=args.with_shuffled_baseline,
            min_entity_count=args.min_entity_count,
            unpaired_cv=args.unpaired_cv,
            min_fold_pos=args.min_fold_pos,
            k_answer=args.k_answer)

    if args.plot_only:
        try:
            plot_only(args.model_id, mode=args.mode,
                      correct_only=args.correct_only, wrong_only=args.wrong_only,
                      cv_mode=args.cv_mode, dataset_name=args.dataset,
                      skip_categories=skip_categories,
                      combine_narrative_floor=args.combine_narrative_floor,
                      combine_entity=args.combine_entity)
        except FileNotFoundError as e:
            print(f"--plot_only requested but no results found ({e}).")
            print("Falling through to a full run.")
            _do_run()
        except KeyError as e:
            print(f"--plot_only requested but cached npz is stale ({e}).")
            print("Cache schema changed; falling through to a full run.")
            _do_run()
    else:
        _do_run()


if __name__ == "__main__":
    main()
