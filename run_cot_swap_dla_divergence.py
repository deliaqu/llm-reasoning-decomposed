"""
Divergence-anchored DLA: per-head Direct Logit Attribution onto the FIRST
token of the wrong-plan engagement clause in noop's natural CoT.

Where the existing run_cot_swap_dla.py asks "which heads write toward `####`
at the cot_end readout?", this asks "which heads push the model to start
writing about the distractor's numerical content, at the moment the wrong
plan is committed?"

Anchor logic per pair (hybrid rule):
  1. From the noop_question, extract numbers present in noop but not in the
     filler-DF "source" question. These are the distractor-specific numbers.
  2. Find first character position c* in `target_own_cot_prefix` where any
     distractor number first appears.
  3. Backtrack to two candidate boundaries before c*:
       - sentence boundary (./!/?/\n) → c_sent
       - clause boundary  (./!/?/\n/,/;/:) → c_clause
  4. Token-align via offset_mapping (containing-token semantics).
  5. Let gap = p_clause - p_sent.
       gap <= 10 → anchor = p_sent  (full-sentence engagement: discourse
                                     marker like "After that," is part of
                                     the wrong-plan unit)
       gap >  10 → anchor = p_clause (embedded clause inside a larger
                                      sentence with substantial prior content)
     p_sent=0 cases naturally fall into the "use P_clause" bucket because
     their gap is always >10 (the first CoT sentence contains substantial
     correct setup before the wrong-plan clause).
  6. Target token = noop_CoT_tokens[anchor] — the actual token noop's CoT
     emitted at that position (e.g. " After", " taking", " we", ...).

Forward pass:
  prompt = chat_template(noop_question + COT_INSTRUCTION) + noop_CoT[:anchor]
  last_pos = end of prompt = the slot the model used to predict target_token
  V at the noop-clause positions (in the question) is captured at each
  layer; per-head DLA is computed exactly the same way as run_cot_swap_dla.py
  (attention-weighted OV from clause positions to readout, projected onto
  the per-pair target token via lm_head).

Output schema mirrors run_cot_swap_dla.py per (model, contrast, layer), with
the target-side filename (`noop` for NoOp contrasts, `p1` for P1):
  <OUT_DIR>/<model>/<contrast>_divergence_dla/l{L}/{side}.json
  with fields {attn_mean, ov_mean, dla_mean, target, side, n}.
The `target` field is a sentinel string ("first-token-of-wrong-plan-clause")
since the target varies per pair; per-pair targets live in the raw npz.

Usage:
  python run_cot_swap_dla_divergence.py --layers 22 23 24 ... 36
"""

import argparse
import difflib
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import COT_INSTRUCTION, LOGIT_LENS_RESULT_DIR, MODEL_NAME_MAP
from run_attention_analysis import get_changed_positions
from run_noop_dla import OUT_DIR
from utils.noop_utils import load_cot_swap_as_patching_pairs, load_model

CHECKPOINT_EVERY = 25
CHECKPOINT_VERSION = 2
ANCHOR_CONTEXT_MIN_OVERLAP = 2

# Path to cot_boundary baselines (noop natural generation under
# transformers re-generation). Used by --filter_flipped to skip pairs
# where noop's natural completion happens to produce gold — these pairs
# slipped past the noop_clean_wrong filter due to decoding-pipeline
# differences between the original inference and our analysis pipeline.
FLIPPED_BASELINE_PATH = (
    Path(__file__).resolve().parent / "results" / "noop_patching"
    / "llama-3.3-70b-instruct" / "filler_df_vs_noop_clean_tfm"
    / "cot_boundary" / "question_span" / "baselines_correct_to_wrong.jsonl"
)


def load_flipped_pair_ids():
    """Return set of (original_id, instance) keys for pairs where noop's
    natural re-generation produces the gold answer (despite being labeled
    noop_clean_wrong). Affects ~2.7% of cot_boundary baseline pairs;
    ~0.9% of our anchored pairs."""
    flipped = set()
    if not FLIPPED_BASELINE_PATH.exists():
        print(f"warning: flipped-baseline file missing at {FLIPPED_BASELINE_PATH}")
        return flipped
    with open(FLIPPED_BASELINE_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get("unpatched_target_correct"):
                flipped.add((int(r["original_id"]), int(r["instance_w"])))
    return flipped

# Sentence/clause boundary detectors.
# Periods are matched only when followed by whitespace or end-of-string — this
# prevents decimal points inside numbers like "1.10" from being treated as
# clause boundaries (which would cause the backtracker to land at the digit
# itself instead of at a meaningful syntactic position before it).
SENT_RE = re.compile(r"[!?\n]|\.(?=\s|$)")
CLAUSE_RE = re.compile(r"[!?\n,;:]|\.(?=\s|$)")
ANCHOR_GAP_THRESHOLD = 10   # gap <= 10 → P_sent; gap > 10 → P_clause
ANCHOR_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "there", "their", "they",
    "them", "then", "than", "from", "have", "has", "had", "his", "her",
    "she", "he", "it", "its", "are", "was", "were", "will", "would",
    "could", "should", "each", "per", "how", "many", "much", "what",
    "when", "where", "which", "also", "does", "did", "done", "due",
}


# ---------- anchor + target token utilities ----------------------------------

def extract_nums(text):
    """Extract numeric tokens (drop commas/dollars) as a set."""
    cleaned = text.replace(",", "").replace("$", "")
    return set(re.findall(r"\d*\.?\d+", cleaned)) - {""}


def _standalone_percent_pattern(n):
    """Match N% as its own numeric mention, not inside 91% or 0.10%."""
    return re.compile(r"(?<![A-Za-z0-9.])" + re.escape(n) + r"%(?![A-Za-z0-9.])")


def _content_terms(text):
    return {
        w for w in re.findall(r"[A-Za-z]+", text.lower())
        if len(w) > 2 and w not in ANCHOR_STOPWORDS
    }


def _target_changed_text(src_q, tgt_q):
    """Return the target-side changed sentence/span between source and target."""
    sm = difflib.SequenceMatcher(None, src_q, tgt_q, autojunk=False)
    spans = [(j1, j2) for tag, _i1, _i2, j1, j2 in sm.get_opcodes()
             if tag != "equal"]
    if not spans:
        return ""
    lo = min(j1 for j1, _j2 in spans)
    hi = max(j2 for _j1, j2 in spans)
    while lo > 0 and tgt_q[lo - 1] not in ".?!\n":
        lo -= 1
    while lo < len(tgt_q) and tgt_q[lo] in " \t\n":
        lo += 1
    while hi < len(tgt_q) and tgt_q[hi] not in ".?!\n":
        hi += 1
    return tgt_q[lo:hi].strip()


def _target_changed_opcode_spans(src_q, tgt_q):
    """Target-side non-equal character spans from a char-level diff."""
    sm = difflib.SequenceMatcher(None, src_q, tgt_q, autojunk=False)
    return [
        (tag, j1, j2)
        for tag, _i1, _i2, j1, j2 in sm.get_opcodes()
        if tag != "equal" and j1 < j2
    ]


def _number_mentions(text, nums):
    """Return (start, end, number) mentions for `nums` in `text`."""
    mentions = []
    for n in nums:
        pct = _standalone_percent_pattern(n).search(text)
        if pct is not None:
            mentions.append((pct.start(), pct.end(), n))
        for m in iter_number_matches(text, n):
            if _is_bullet_enumerator(text, m):
                continue
            mentions.append((m.start(), m.end(), n))
    # Deduplicate cases where N% also yielded a bare N match at the same start.
    out = {}
    for s, e, n in mentions:
        key = (s, n)
        if key not in out or e > out[key][1]:
            out[key] = (s, e, n)
    return sorted(out.values())


def _number_match_patterns(n):
    patterns = [r"\b" + re.escape(n) + r"\b"]
    if n.isdigit() and len(n) > 3:
        comma_form = f"{int(n):,}"
        if comma_form != n:
            patterns.append(
                r"(?<![A-Za-z0-9.])" + re.escape(comma_form) + r"(?![A-Za-z0-9.])"
            )
    return patterns


def iter_number_matches(text, n):
    for pat in _number_match_patterns(n):
        yield from re.finditer(pat, text)


def _match_has_adjacent(text, match, chars):
    return (
        (match.start() > 0 and text[match.start() - 1] in chars)
        or (match.end() < len(text) and text[match.end()] in chars)
    )


def _match_is_decimal_part(text, match):
    if match.start() > 0 and text[match.start() - 1] == ".":
        return match.start() > 1 and text[match.start() - 2].isdigit()
    if match.end() < len(text) and text[match.end()] == ".":
        return match.end() + 1 < len(text) and text[match.end() + 1].isdigit()
    return False


def _nums_allowed_with_adjacent(target_context, nums, chars):
    allowed = set()
    for n in nums:
        for m in iter_number_matches(target_context, n):
            if _match_has_adjacent(target_context, m, chars):
                allowed.add(n)
                break
    return allowed


def _right_expand_to_boundary(text, pos):
    """Expand to the next sentence boundary, preserving the boundary char."""
    stops = [j for j in (text.find(ch, pos) for ch in ".?!\n") if j >= 0]
    return (min(stops) + 1) if stops else len(text)


def _left_expand_to_sentence(text, pos):
    lo = max(
        text.rfind(".", 0, pos),
        text.rfind("?", 0, pos),
        text.rfind("!", 0, pos),
        text.rfind("\n", 0, pos),
    )
    lo = 0 if lo < 0 else lo + 1
    while lo < len(text) and text[lo] in " \t\n":
        lo += 1
    return lo


def _trim_span(text, lo, hi):
    while lo > 0 and lo < hi and text[lo - 1].isalpha() and text[lo].isalpha():
        lo -= 1
    while lo < hi and text[lo] in " \t\n,;:":
        lo += 1
    while hi > lo and text[hi - 1] in " \t\n":
        hi -= 1
    return lo, hi


def p1_changed_question_span(row):
    """Return the target-question char span for the P1-only changed content.

    The padded-symbolic source often differs from P1 in two places: a padding
    sentence is deleted and the real P1 operation is inserted/replaced. The
    generic longest-prefix/suffix token diff therefore spans too much. This
    helper selects target-side diff spans that contain P1-only numeric mentions
    and then expands them just enough to recover the full local clause/sentence.
    """
    src_q = row.get("source_question", "")
    tgt_q = row.get("target_question", "")
    if not isinstance(src_q, str) or not isinstance(tgt_q, str):
        return None
    nums = extract_nums(tgt_q) - extract_nums(src_q)
    if not nums:
        return None

    mentions = _number_mentions(tgt_q, nums)
    if not mentions:
        return None

    changed = _target_changed_opcode_spans(src_q, tgt_q)
    selected = []
    for tag, j1, j2 in changed:
        if any(ms < j2 and me > j1 for ms, me, _n in mentions):
            selected.append((tag, j1, j2))

    if selected:
        lo = min(j1 for _tag, j1, _j2 in selected)
        hi = max(j2 for _tag, _j1, j2 in selected)
        if any(tag != "insert" for tag, _j1, _j2 in selected):
            lo = _left_expand_to_sentence(tgt_q, lo)
        hi = _right_expand_to_boundary(tgt_q, hi)
        return _trim_span(tgt_q, lo, hi)

    # Fallback: use the earliest target-only numeric mention and keep the
    # containing sentence. This is conservative and should be rare.
    lo = _left_expand_to_sentence(tgt_q, mentions[0][0])
    hi = _right_expand_to_boundary(tgt_q, mentions[0][1])
    return _trim_span(tgt_q, lo, hi)


def _sentence_window(text, char_pos):
    lo = max(
        text.rfind(".", 0, char_pos),
        text.rfind("?", 0, char_pos),
        text.rfind("!", 0, char_pos),
        text.rfind("\n", 0, char_pos),
    )
    lo = 0 if lo < 0 else lo + 1
    hi = len(text)
    for marker in ".?!\n":
        j = text.find(marker, char_pos)
        if j >= 0:
            hi = min(hi, j)
    return text[lo:hi]


def _candidate_context_window(text, char_pos, radius=220):
    """Context for lexical alignment, including prior heading/bullet context.

    We intentionally do not include future text outside the current sentence:
    otherwise an earlier arithmetic coincidence can borrow lexical overlap from
    a later distractor heading.
    """
    lo = max(0, char_pos - radius)
    return _sentence_window(text, char_pos) + " " + text[lo:char_pos]


def _is_numbered_list_boundary(text, match):
    """True when a period boundary is the dot in a numbered-list marker."""
    if match.group(0) != ".":
        return False
    line_start = text.rfind("\n", 0, match.start()) + 1
    before_dot = text[line_start:match.start()].strip()
    if not before_dot.isdigit():
        return False
    after = match.end()
    return after < len(text) and text[after] in " \t"


def backtrack(boundary_re, text, char_pos):
    """Walk back to last boundary whose post-whitespace position is STRICTLY
    before char_pos. Searches the full text (not a slice) so the regex's
    lookahead `\\s|$` correctly sees the character that comes AFTER each
    period — otherwise periods inside numbers like '1.10' get matched because
    the slice end satisfies `$`.

    Iterates boundaries in reverse: the very last boundary may have only
    whitespace between it and char_pos, which would make the anchor land AT
    char_pos itself (the bullet-enumerator failure mode where '\\n2.' is the
    last boundary before '10' in '\\n2. 10%'). In that case we keep walking
    back until we find a boundary with at least one non-whitespace character
    between it and char_pos.
    """
    bounds = [m for m in boundary_re.finditer(text) if m.end() <= char_pos]
    for m in reversed(bounds):
        j = m.end()
        while j < len(text) and text[j] in " \t\n":
            j += 1
        if j < char_pos:
            return j
        if j == char_pos and _is_numbered_list_boundary(text, m):
            # Anchor just after the list marker ("2.| 10%...") instead of
            # backing up to the marker token itself.
            return m.end()
    return 0


def char_to_token_containing(offset_mapping, char_pos):
    """Find the token index whose char span contains char_pos
    (offset_start <= char_pos < offset_end). Handles leading-whitespace
    tokens correctly."""
    for i, (s, e) in enumerate(offset_mapping):
        if s <= char_pos < e:
            return i
    return None


def _is_bullet_enumerator(text, match):
    """True if `match` (a re.Match) sits at the start of a line and is
    immediately followed by '.' + whitespace (i.e., looks like '4. ')."""
    line_start = text.rfind("\n", 0, match.start()) + 1
    if text[line_start:match.start()].strip() != "":
        return False
    after = match.end()
    if after >= len(text) or text[after] != ".":
        return False
    if after + 1 < len(text) and text[after + 1] not in " \t":
        return False
    return True


def find_distractor_engagement_char(noop_cot, distractor_nums, target_context=""):
    """Find the first char position in `noop_cot` where a distractor number
    appears as a real engagement (not a coincidental match).

    Per distractor number N, in order of specificity:
      1. `\\bN%` (the "X%" pattern — most reliable for inflation distractors).
         Catches "15%" in "15% inflation" but skips "1.15" multiplier (where
         the "15" isn't followed by %).
      2. `\\bN\\b` (word-boundary, excluding bullet enumerators like "4. ").
         The \\b prevents matching "2" inside "2250" (derived value); the
         bullet skip prevents matching "4" inside an enumerated-list "4." entry.

    Across distractor numbers, pick the EARLIEST char position.
    If target_context is provided, non-percent numeric matches are first
    filtered to occurrences whose containing CoT sentence shares content words
    with the inserted target-side clause. This avoids small-number coincidences
    like matching the `2` in `1320 * 2` before the later `2 hours` distractor
    discussion.
    Returns None if no acceptable match found.
    """
    candidates = []
    target_terms = _content_terms(target_context)
    fraction_ok = _nums_allowed_with_adjacent(target_context, distractor_nums, "/")
    decimal_ok = _nums_allowed_with_adjacent(target_context, distractor_nums, ".")
    for n in distractor_nums:
        # Step 1: try N% (explicit percent reference)
        pct_m = _standalone_percent_pattern(n).search(noop_cot)
        if pct_m is not None:
            candidates.append(pct_m.start())
            continue
        # Step 2: word-boundary N, skipping bullet enumerators
        num_candidates = []
        for m in iter_number_matches(noop_cot, n):
            if _is_bullet_enumerator(noop_cot, m):
                continue
            if _match_has_adjacent(noop_cot, m, "/") and n not in fraction_ok:
                continue
            if _match_is_decimal_part(noop_cot, m) and n not in decimal_ok:
                continue
            overlap = 0
            if target_terms:
                overlap = len(
                    target_terms & _content_terms(_candidate_context_window(noop_cot, m.start()))
                )
            num_candidates.append((m.start(), overlap))
        if target_terms:
            aligned = [
                (pos, overlap) for pos, overlap in num_candidates
                if overlap >= ANCHOR_CONTEXT_MIN_OVERLAP
            ]
            if aligned:
                candidates.append(min(pos for pos, _overlap in aligned))
                continue
        if num_candidates:
            candidates.append(num_candidates[0][0])
    return min(candidates) if candidates else None


def find_anchor(row, tokenizer, num_strategy="set_diff"):
    """Return (anchor_token_idx_in_noop_cot, target_token_id, bucket_label) or
    None if the pair has no detectable distractor engagement.

    bucket_label ∈ {"discourse-marker", "embedded-clause", "first-sentence",
                    "no-discourse-marker"} for diagnostics.

    num_strategy:
      "set_diff" (default, NoOp): distractor_nums = numbers(tgt) - numbers(src).
        Works when the distractor introduces fresh numbers (typical for NoOp).
      "p1_set_diff": same number rule, but lexical context comes from the
        P1-only changed question span rather than the broad source/target diff.
      "changed_span" (P1): distractor_nums = numbers(target-side changed span).
        Needed when the inserted clause REUSES base-problem numbers (e.g., P1's
        "I add 10 liters of water" reusing the "10" from "spill 10 liters" —
        set-diff would be empty in that case). Includes the question delta
        too, but `target_context` lexical-overlap filtering picks the correct
        occurrence in the CoT.
    """
    src_q = row.get("source_question", "")
    tgt_q = row.get("target_question", "")
    noop_cot = row.get("target_own_cot_prefix", "")
    if not all(isinstance(x, str) and x for x in [src_q, tgt_q, noop_cot]):
        return None
    target_context = _target_changed_text(src_q, tgt_q)
    if num_strategy == "set_diff":
        distractor_nums = extract_nums(tgt_q) - extract_nums(src_q)
    elif num_strategy == "p1_set_diff":
        distractor_nums = extract_nums(tgt_q) - extract_nums(src_q)
        span = p1_changed_question_span(row)
        if span is not None:
            target_context = tgt_q[span[0]:span[1]]
    elif num_strategy == "changed_span":
        distractor_nums = extract_nums(_target_changed_text(src_q, tgt_q))
    else:
        raise ValueError(f"Unknown num_strategy={num_strategy!r}")
    if not distractor_nums:
        return None

    c_star = find_distractor_engagement_char(noop_cot, distractor_nums, target_context)
    if c_star is None:
        return None

    c_sent = backtrack(SENT_RE, noop_cot, c_star)
    c_clause = backtrack(CLAUSE_RE, noop_cot, c_star)

    enc = tokenizer(noop_cot, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    ids = enc["input_ids"]

    pt_sent = char_to_token_containing(offsets, c_sent) if c_sent < len(noop_cot) else 0
    pt_clause = char_to_token_containing(offsets, c_clause) if c_clause < len(noop_cot) else 0
    if pt_sent is None or pt_clause is None:
        return None

    gap = pt_clause - pt_sent
    if gap <= ANCHOR_GAP_THRESHOLD:
        anchor = pt_sent
        bucket = "discourse-marker" if gap > 0 else "no-discourse-marker"
    else:
        anchor = pt_clause
        bucket = "first-sentence" if pt_sent == 0 else "embedded-clause"

    if anchor < 0 or anchor >= len(ids):
        return None
    target_token = ids[anchor]
    return anchor, target_token, bucket


# ---------- prompt construction ----------------------------------------------

def build_divergence_prompt(noop_question, noop_cot_decoded_prefix, tokenizer):
    """chat_template(noop_question + "\n" + COT_INSTRUCTION) + CoT prefix.

    `noop_cot_decoded_prefix` is the first `anchor` tokens of noop's CoT,
    decoded back to a string (so it round-trips cleanly through tokenization).
    """
    msgs = [{"role": "user", "content": f"{noop_question}\n{COT_INSTRUCTION}"}]
    base = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    return base + noop_cot_decoded_prefix


def p1_changed_token_positions(row, prompt, tokenizer):
    """Token positions in `prompt` overlapping the P1-only target-question span."""
    span = p1_changed_question_span(row)
    if span is None:
        return []
    lo, hi = span
    question = row["target_question"]
    q_start = prompt.find(question)
    if q_start < 0:
        return []
    abs_lo = q_start + lo
    abs_hi = q_start + hi
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for i, (s, e) in enumerate(enc["offset_mapping"]):
        if e > s and e > abs_lo and s < abs_hi:
            positions.append(i)
    return positions


# ---------- W_O per-head ------------------------------------------------------

def w_o_per_head(model, layer_idx, n_heads, head_dim, d_model):
    """Return W_O reshaped to (n_heads, d_model, head_dim) on CPU."""
    blk = model.model.layers[layer_idx]
    W_O = blk.self_attn.o_proj.weight.data
    return W_O.detach().cpu().to(torch.float32).view(d_model, n_heads, head_dim).permute(1, 0, 2).contiguous()


# ---------- output paths ------------------------------------------------------

def _suffix(restrict_percent, filter_flipped=False):
    parts = []
    if restrict_percent:
        parts.append("_pct")
    if filter_flipped:
        parts.append("_nofl")
    return "".join(parts)


def divergence_layer_dir(model_name, contrast, layer, restrict_percent=False, filter_flipped=False):
    return OUT_DIR / model_name / f"{contrast}_divergence_dla{_suffix(restrict_percent, filter_flipped)}" / f"l{layer}"


def checkpoint_path(model_name, contrast, restrict_percent=False, filter_flipped=False):
    return OUT_DIR / model_name / f"{contrast}_divergence_dla{_suffix(restrict_percent, filter_flipped)}" / "_inflight_checkpoint.pkl"


def load_checkpoint(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"warning: could not load checkpoint at {path}: {e!r}")
        return None


def _checkpoint_row_count(accum):
    """Return the row count implied by layer accumulators, or None if inconsistent."""
    lengths = []
    for layer_buf in accum.values():
        layer_lengths = {len(v) for v in layer_buf.values()}
        if len(layer_lengths) != 1:
            return None
        lengths.extend(layer_lengths)
    counts = set(lengths)
    if len(counts) != 1:
        return None
    return counts.pop()


def _checkpoint_is_consistent(accum, completed_rows, row_ids):
    """Check that checkpoint arrays and row metadata agree.

    `completed_rows` also includes skipped rows, so it can be larger than the
    number of accumulated DLA rows. The strict equality is between layer buffers
    and row_ids.
    """
    row_count = _checkpoint_row_count(accum)
    if row_count is None:
        return False, "inconsistent layer buffer lengths"
    if row_count != len(row_ids):
        return False, (
            f"row metadata mismatch (buffers={row_count}, row_ids={len(row_ids)})"
        )
    if len(completed_rows) < row_count:
        return False, (
            f"completed rows missing entries (buffers={row_count}, "
            f"completed={len(completed_rows)})"
        )
    completed_keys = set(completed_rows)
    row_keys = {(int(r[0]), int(r[1])) for r in row_ids}
    if not row_keys.issubset(completed_keys):
        return False, "row_ids include keys absent from completed_rows"
    return True, ""


def save_checkpoint(path, accum, completed_rows, row_ids, n_skip):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"version": CHECKPOINT_VERSION,
                     "accum": accum, "completed_rows": completed_rows,
                     "row_ids": row_ids, "n_skip": n_skip}, f)
    tmp.replace(path)


# ---------- main run ---------------------------------------------------------

def is_percent_engagement(row):
    """Return True if any distractor-only number is cited as "N%" in noop's
    natural CoT. This is the reliable inflation-style subset where the
    anchor heuristic works well (~44% of pairs)."""
    src_q = row.get("source_question", "")
    tgt_q = row.get("target_question", "")
    noop = row.get("target_own_cot_prefix", "")
    if not all(isinstance(x, str) and x for x in [src_q, tgt_q, noop]):
        return False
    distractor_nums = extract_nums(tgt_q) - extract_nums(src_q)
    return any(_standalone_percent_pattern(n).search(noop) for n in distractor_nums)


def run_divergence_dla(model, tokenizer, model_name, contrast, layers,
                       restrict_percent=False, filter_flipped=False):
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")

    cfg = model.config
    n_heads, n_kv_heads = cfg.num_attention_heads, cfg.num_key_value_heads
    # Prefer explicit cfg.head_dim — gemma-2-9b has head_dim=256 while
    # hidden_size//n_heads = 3584//16 = 224, so the derived value is wrong.
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    kv_groups = n_heads // n_kv_heads
    d_model = cfg.hidden_size

    block_idxs = [l - 1 for l in layers]
    W_O = {bi: w_o_per_head(model, bi, n_heads, head_dim, d_model) for bi in block_idxs}

    captured = {}
    hooks = []
    for bi in block_idxs:
        attn_mod = model.model.layers[bi].self_attn

        def make_hook(idx):
            def _h(_m, _i, out):
                captured[idx] = out.detach()
            return _h

        hooks.append(attn_mod.v_proj.register_forward_hook(make_hook(bi)))

    # P1-vs-padded source: no donor CoT to override (we anchor on P1's own CoT).
    # Number-extraction strategy stays at set_diff to mirror NoOp exactly: only
    # the pairs where P1's extra step introduces a NEW operand are anchored.
    # Pairs where the extra step reuses base-problem numbers (e.g., "spill 10
    # liters" → "add 10 liters of water") are skipped — changed_span would
    # recover them but anchors land in CoT setup phase, not extra-step
    # engagement, because base-problem terms saturate the lexical-overlap
    # filter. Set-diff yields ~55% of 4446 = ~2500 candidate pairs, well above
    # NoOp's 936 → 720 yield.
    is_p1_contrast = contrast == "p1_correct_vs_padded"
    side_name = "p1" if is_p1_contrast else "noop"
    target_desc = (
        "first-token-of-p1-extra-step-engagement (per-pair)"
        if is_p1_contrast
        else "first-token-of-wrong-plan-clause (per-pair)"
    )

    if is_p1_contrast:
        cot_source = None
        num_strategy = "p1_set_diff"
    else:
        cot_source = "noop_clean"
        num_strategy = "set_diff"

    df = load_cot_swap_as_patching_pairs(source=contrast, cot_source=cot_source)
    if restrict_percent:
        keep = df.apply(is_percent_engagement, axis=1)
        df = df[keep].reset_index(drop=True)
        print(f"Restricted to 'X%' pairs (inflation-style): {len(df)} pairs")
    else:
        cot_note = "none" if cot_source is None else cot_source
        print(f"Loaded {len(df)} pairs for {contrast} (cot_source={cot_note})")
    if filter_flipped:
        flipped = load_flipped_pair_ids()
        before = len(df)
        keep = df.apply(lambda r: (int(r["original_id"]), int(r["instance"])) not in flipped, axis=1)
        df = df[keep].reset_index(drop=True)
        print(f"Filtered flipped-inference pairs: dropped {before - len(df)} → {len(df)} pairs")

    accum = {L: {"attn": [], "ov": [], "dla": [], "targets": [], "buckets": [],
                 "anchor_pos_in_cot": []} for L in layers}
    completed_rows = set()
    row_ids = []
    n_skip = {"no_anchor": 0, "no_clause": 0}

    chk = load_checkpoint(checkpoint_path(model_name, contrast, restrict_percent, filter_flipped))
    if chk is not None:
        wanted = set(layers)
        if chk.get("version") != CHECKPOINT_VERSION:
            print(
                "warning: checkpoint version mismatch "
                f"(found {chk.get('version')!r}, want {CHECKPOINT_VERSION}); "
                "restarting fresh"
            )
            checkpoint_path(model_name, contrast, restrict_percent, filter_flipped).unlink(missing_ok=True)
            chk = None
        else:
            accum = chk["accum"]
            completed_rows = chk["completed_rows"]
            row_ids = chk["row_ids"]
            n_skip = chk["n_skip"]

        if chk is not None and not wanted.issubset(accum.keys()):
            print(f"warning: checkpoint missing layers {wanted - set(accum.keys())}; restarting fresh")
            accum = {L: {"attn": [], "ov": [], "dla": [], "targets": [], "buckets": [],
                         "anchor_pos_in_cot": []} for L in layers}
            completed_rows = set()
            row_ids = []
            n_skip = {"no_anchor": 0, "no_clause": 0}
        elif chk is not None:
            ok, reason = _checkpoint_is_consistent(accum, completed_rows, row_ids)
            if not ok:
                print(f"warning: checkpoint is inconsistent ({reason}); restarting fresh")
                accum = {L: {"attn": [], "ov": [], "dla": [], "targets": [], "buckets": [],
                             "anchor_pos_in_cot": []} for L in layers}
                completed_rows = set()
                row_ids = []
                n_skip = {"no_anchor": 0, "no_clause": 0}
            else:
                print(f"resuming with {len(completed_rows)} rows already done")

        if chk is not None and wanted.issubset(accum.keys()) and not completed_rows and not row_ids:
            # Keep the just-written bad checkpoint from being loaded again if this
            # run is interrupted before the first successful periodic save.
            bad_chk = checkpoint_path(model_name, contrast, restrict_percent, filter_flipped)
            bad_chk.unlink(missing_ok=True)

    try:
        model.eval()
        for row_idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=contrast)):
            key = (int(row["original_id"]), int(row["instance"]))
            if key in completed_rows:
                continue

            anchor_info = find_anchor(row, tokenizer, num_strategy=num_strategy)
            if anchor_info is None:
                n_skip["no_anchor"] += 1
                completed_rows.add(key)
                continue
            anchor_idx, target_token, bucket = anchor_info

            noop_cot = row["target_own_cot_prefix"]
            enc = tokenizer(noop_cot, add_special_tokens=False, return_offsets_mapping=True)
            cot_ids = enc["input_ids"]
            # Decode the prefix [:anchor_idx]. We DROP the anchor token itself —
            # the model will predict it from the residual at last_pos.
            if anchor_idx == 0:
                prefix_decoded = ""
            else:
                prefix_decoded = tokenizer.decode(cot_ids[:anchor_idx], skip_special_tokens=True)

            noop_q = row["target_question"]
            prompt = build_divergence_prompt(noop_q, prefix_decoded, tokenizer)
            input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

            # Identify target-side changed positions inside the question prefill.
            # For NoOp/filler contrasts there is one contiguous question edit,
            # so longest-prefix/suffix token diff is adequate. For P1-vs-padded
            # the padded sentence deletion and P1 operation insertion are often
            # two edits; use a target-side numeric span mapped through tokenizer
            # offsets so we only attribute from the real P1 operation clause.
            src_q = row["source_question"]
            if is_p1_contrast:
                noop_changed = p1_changed_token_positions(row, prompt, tokenizer)
            else:
                src_prompt = build_divergence_prompt(src_q, prefix_decoded, tokenizer)
                src_ids = tokenizer(src_prompt, add_special_tokens=False)["input_ids"]
                _, noop_changed = get_changed_positions(src_ids, input_ids)
            if not noop_changed:
                n_skip["no_clause"] += 1
                completed_rows.add(key)
                continue
            # Filter out templates with structural sentence-layout mismatch
            # between filler-DF and noop questions: in those, get_changed_positions
            # returns much more than just the distractor sentence (e.g., the
            # entire math problem). Threshold of 25 cleanly separates such cases
            # from normal distractors (median 13 tokens, p90 15). Empirically
            # only template oid=1084 hit this in May 2026 (32 of 444 pairs).
            if not is_p1_contrast and len(noop_changed) > 25:
                n_skip["clause_too_long"] = n_skip.get("clause_too_long", 0) + 1
                completed_rows.add(key)
                continue

            last_pos = len(input_ids) - 1
            input_t = torch.tensor([input_ids]).to(model.device)
            with torch.no_grad():
                out = model(input_t, output_attentions=True)

            row_record = (int(row["original_id"]), int(row["instance"]), int(anchor_idx),
                          int(target_token), bucket)

            for layer, bi in zip(layers, block_idxs):
                attn_w = out.attentions[bi][0].float()
                v_raw = captured[bi][0]
                v_at_clause = v_raw[noop_changed].float().cpu()
                v_kv = v_at_clause.view(len(noop_changed), n_kv_heads, head_dim)
                v_all = v_kv.repeat_interleave(kv_groups, dim=1)

                P = len(noop_changed)
                M_all = torch.einsum("hkd,phd->phk", W_O[bi], v_all)
                M_batch = M_all.reshape(P * n_heads, d_model)
                with torch.no_grad():
                    logits_batch = model.lm_head(
                        M_batch.to(device=attn_w.device,
                                   dtype=model.model.layers[bi].self_attn.o_proj.weight.dtype)
                    )

                attn_to_clause_pos = attn_w[:, last_pos, :][:, noop_changed].transpose(0, 1).cpu()
                attn_to_clause = attn_to_clause_pos.sum(dim=0)

                # Per-pair target token (varies per row, not the fixed #### here)
                ov_per_pos = logits_batch[:, target_token].cpu().view(P, n_heads)
                dla_all = (attn_to_clause_pos * ov_per_pos).sum(dim=0)
                ov_all = torch.where(
                    attn_to_clause.abs() > 1e-12,
                    dla_all / attn_to_clause,
                    ov_per_pos.mean(dim=0),
                )
                accum[layer]["attn"].append(attn_to_clause.numpy())
                accum[layer]["ov"].append(ov_all.numpy())
                accum[layer]["dla"].append(dla_all.numpy())
                accum[layer]["targets"].append(int(target_token))
                accum[layer]["buckets"].append(bucket)
                accum[layer]["anchor_pos_in_cot"].append(int(anchor_idx))

            row_ids.append(row_record)
            del out
            torch.cuda.empty_cache()
            completed_rows.add(key)
            if (row_idx + 1) % CHECKPOINT_EVERY == 0:
                save_checkpoint(checkpoint_path(model_name, contrast, restrict_percent, filter_flipped), accum,
                                completed_rows, row_ids, n_skip)
    finally:
        for h in hooks:
            h.remove()
        ok, reason = _checkpoint_is_consistent(accum, completed_rows, row_ids)
        if ok:
            save_checkpoint(checkpoint_path(model_name, contrast, restrict_percent, filter_flipped), accum,
                            completed_rows, row_ids, n_skip)
        else:
            print(f"warning: not saving inconsistent checkpoint after interruption ({reason})")

    print(f"\nskipped: {n_skip}")

    # Persist per (layer, target side)
    for layer in layers:
        out_dir = divergence_layer_dir(model_name, contrast, layer, restrict_percent, filter_flipped)
        out_dir.mkdir(parents=True, exist_ok=True)
        buf = accum[layer]
        if not buf["attn"]:
            print(f"  L{layer}: no rows, skipping save")
            continue
        attn_arr = np.array(buf["attn"])
        ov_arr = np.array(buf["ov"])
        dla_arr = np.array(buf["dla"])
        result = {
            "attn_mean": attn_arr.mean(0).tolist(),
            "ov_mean": ov_arr.mean(0).tolist(),
            "dla_mean": dla_arr.mean(0).tolist(),
            "target": target_desc,
            "side": side_name,
            "n": int(attn_arr.shape[0]),
        }
        (out_dir / f"{side_name}.json").write_text(json.dumps(result, indent=2))
        np.savez_compressed(
            out_dir / f"{side_name}_raw.npz",
            attn=attn_arr, ov=ov_arr, dla=dla_arr,
            targets=np.array(buf["targets"], dtype=np.int64),
            buckets=np.array(buf["buckets"]),
            anchor_pos_in_cot=np.array(buf["anchor_pos_in_cot"], dtype=np.int32),
            row_ids=np.array(row_ids, dtype=object),
            n_heads=np.int32(n_heads), n_kv_heads=np.int32(n_kv_heads),
            kv_groups=np.int32(kv_groups), head_dim=np.int32(head_dim),
        )
        print(f"  L{layer}: N={result['n']} → {out_dir / f'{side_name}.json'}")

    chk_p = checkpoint_path(model_name, contrast, restrict_percent, filter_flipped)
    if chk_p.exists():
        chk_p.unlink()
        print(f"removed checkpoint {chk_p}")


# ---------- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--contrast", default="filler_df_correct_vs_noop_clean_wrong")
    parser.add_argument("--layers", type=int, nargs="+",
                        default=list(range(22, 37)))
    parser.add_argument("--restrict_percent", action="store_true",
                        help="Restrict to pairs where the distractor number is "
                             "cited as 'X%%' in noop's natural CoT (inflation-"
                             "style subset, ~44%% of pairs). Output dir gets "
                             "`_pct` suffix.")
    parser.add_argument("--filter_flipped", action="store_true",
                        help="Drop pairs where noop's natural re-generation "
                             "produces gold (despite being labeled "
                             "noop_clean_wrong). Removes a small fraction of "
                             "pairs (~0.9%% of anchored, ~1.5%% of X%%) where "
                             "the noop_clean_wrong filter was unreliable under "
                             "transformers re-generation. Output dir gets "
                             "`_nofl` suffix.")
    args = parser.parse_args()

    model_name = MODEL_NAME_MAP[args.model_id]
    model, tokenizer = load_model(args.model_id)
    model.set_attn_implementation("eager")
    run_divergence_dla(model, tokenizer, model_name, args.contrast, args.layers,
                       restrict_percent=args.restrict_percent,
                       filter_flipped=args.filter_flipped)
