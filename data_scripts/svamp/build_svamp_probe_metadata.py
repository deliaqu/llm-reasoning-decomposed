"""Build SVAMP probe metadata: seed-disjoint CV groups + per-row probe labels.

SVAMP (Patel et al. 2021) ships no seed-group field, but its 1000 problems are
variations (Question Sensitivity / Reasoning Ability / Structural Invariance)
of ~100 seed problems. Probe CV must therefore be SEED-disjoint — the analog of
the GSM template-disjoint folds — or variations of one seed leak across folds.

Seed groups are reconstructed by union-find over normalized Bodies from
SVAMP_original.json: two rows merge when the Jaccard similarity of their
digit-stripped Body token sets >= --jaccard. Over-merging is the safe direction
(costs statistical efficiency only), under-merging breaks CV validity, so the
default threshold is deliberately low (0.4).

Per-row probe label columns (targets for the four-stage probe suite, mirroring
run_presence_probe.py categories):
  entity_name_words   S1  proper-name role fillers found in the question
  entity_other_words  S1  object/unit common nouns (post-numeral + of-phrase heuristic)
  op_cue_words        S1/ctrl  OP_CUES (imported from run_presence_probe) present
  function_words      ctrl     FUNCTION words (imported) present
  operators           S2  gold-formula operator multiset from symbolic_abstraction_answer
  op_count            S2  number of operator tokens in the gold formula
  operand_values      S3  operand numbers parsed from symbol_binding
  answer              S4  gold answer (copied through for convenience)

Outputs (both row-index-aligned to data/test_svamp.csv):
  data/svamp_probe_metadata.csv     one row per problem
  data/svamp_probe_vocab_audit.csv  per-word document frequency + seed-cluster
                                    spread, per category (probe-feasibility audit)

Usage:
  python data_scripts/svamp/build_svamp_probe_metadata.py [--jaccard 0.4]
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_wordset_from_source(module_path: Path, name: str) -> set[str]:
    """Extract a module-level set literal from source without importing it
    (run_presence_probe transitively imports torch, absent on login nodes).
    Keeps run_presence_probe.py the single source of truth for the word lists."""
    tree = ast.parse(module_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise LookupError(f"{name} not found in {module_path}")


_PROBE_SRC = PROJECT_ROOT / "interpretability" / "run_presence_probe.py"
# Reuse the exact GSM probe word lists so SVAMP probes stay comparable.
OP_CUES = _load_wordset_from_source(_PROBE_SRC, "OP_CUES")
FUNCTION = _load_wordset_from_source(_PROBE_SRC, "FUNCTION")

DATA_DIR = PROJECT_ROOT / "data"

WORD_RE = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
NUM_RE = re.compile(r"\d+(?:\.\d+)?")
# Capitalized-token candidates for proper names; titles/pronouns/etc. that are
# capitalized mid-sentence but are not names.
NAME_BLACKLIST = {
    "i", "if", "how", "what", "when", "where", "which", "who", "why",
    "there", "each", "every", "some", "all", "the", "a", "an",
    "he", "she", "they", "his", "her", "their", "it", "its",
    "mr", "mrs", "ms", "dr", "st",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "christmas", "easter", "halloween", "thanksgiving",
    "american", "chinese", "french", "english", "spanish",
}


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text)


def norm_body_tokens(body: str) -> frozenset[str]:
    """Digit-stripped lowercase token set for seed clustering."""
    body = NUM_RE.sub(" ", body)
    return frozenset(w.lower() for w in tokens(body))


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def cluster_seeds(bodies: list[str], jaccard_threshold: float):
    """Union-find over pairwise Jaccard of digit-stripped Body token sets."""
    sets = [norm_body_tokens(b) for b in bodies]
    n = len(sets)
    uf = UnionFind(n)
    sims = []  # off-diagonal similarity sample for the threshold audit
    for i in range(n):
        si = sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sets[j]
            if not sj:
                continue
            inter = len(si & sj)
            if inter == 0:
                continue
            jac = inter / len(si | sj)
            sims.append(jac)
            if jac >= jaccard_threshold:
                uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    # densify group ids in first-appearance order
    remap: dict[int, int] = {}
    groups = []
    for r in roots:
        if r not in remap:
            remap[r] = len(remap)
        groups.append(remap[r])
    return groups, sims


def sentence_initial_words(text: str) -> set[str]:
    """Words appearing sentence-initially (can't tell name vs. capitalization)."""
    initials = set()
    for sent in re.split(r"[.?!]", text):
        toks = tokens(sent)
        if toks:
            initials.add(toks[0])
    return initials


def extract_entity_names(questions: list[str]) -> list[list[str]]:
    """Proper-name role fillers: capitalized tokens with corpus-level evidence
    of at least one NON-sentence-initial occurrence (else 'The'/'If' etc. would
    flood in), minus the blacklist."""
    mid_sentence_caps: Counter[str] = Counter()
    for q in questions:
        initials = sentence_initial_words(q)
        for w in tokens(q):
            if w[0].isupper() and w not in initials:
                mid_sentence_caps[w.lower()] += 1
    name_vocab = {
        w for w, c in mid_sentence_caps.items()
        if w not in NAME_BLACKLIST and w not in FUNCTION and w not in OP_CUES
    }
    per_row = []
    for q in questions:
        row_words = {w.lower() for w in tokens(q) if w[0].isupper()}
        per_row.append(sorted(row_words & name_vocab))
    return per_row


def extract_entity_other(questions: list[str]) -> list[list[str]]:
    """Object/unit nouns: words directly following a numeral ("76 dollars",
    "3 cups"), plus of-phrase heads after such a unit ("3 cups of flour" ->
    flour). Vocabulary built corpus-wide, presence marked per row."""
    post_num_re = re.compile(r"\d+(?:\.\d+)?\s+([A-Za-z]+)(?:\s+of\s+([A-Za-z]+))?")
    vocab: Counter[str] = Counter()
    for q in questions:
        for m in post_num_re.finditer(q):
            for w in m.groups():
                if not w:
                    continue
                w = w.lower()
                if w not in FUNCTION and w not in OP_CUES and len(w) > 2:
                    vocab[w] += 1
    # names are S1-entity_name targets, not objects
    name_rows = extract_entity_names(questions)
    name_vocab = {w for row in name_rows for w in row}
    obj_vocab = {w for w in vocab if w not in name_vocab}
    per_row = []
    for q in questions:
        row_words = {w.lower() for w in tokens(q)}
        per_row.append(sorted(row_words & obj_vocab))
    return per_row


def parse_operators(expr: str) -> list[str]:
    return re.findall(r"[+\-*/]", str(expr))


def parse_operands(symbol_binding: str) -> list[float]:
    vals = []
    for m in re.finditer(r"[a-z]\s*=\s*(\d+(?:\.\d+)?)", str(symbol_binding)):
        vals.append(float(m.group(1)))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jaccard", type=float, default=0.4,
                    help="Body-token Jaccard union threshold (lower = more merging = safer CV)")
    args = ap.parse_args()

    csv = pd.read_csv(DATA_DIR / "test_svamp.csv")
    original = json.load(open(DATA_DIR / "SVAMP_original.json"))
    assert len(csv) == len(original) == 1000

    # Re-verify index alignment (JSON Body+Question == CSV question, answers equal).
    for i, entry in enumerate(original):
        joined = f"{entry['Body'].strip()} {entry['Question'].strip()}"
        assert joined == str(csv.iloc[i]["question"]).strip(), f"question mismatch at row {i}"
        assert float(entry["Answer"]) == float(csv.iloc[i]["answer"]), f"answer mismatch at row {i}"
    print("Alignment re-verified: 1000/1000 question+answer match between JSON and CSV.")

    bodies = [e["Body"] for e in original]
    questions = csv["question"].astype(str).tolist()

    # ── seed clustering ───────────────────────────────────────────────────
    groups, sims = cluster_seeds(bodies, args.jaccard)
    sizes = Counter(Counter(groups).values())
    n_clusters = len(set(groups))
    print(f"\nSeed clustering @ Jaccard>={args.jaccard}: {n_clusters} clusters")
    print("  cluster-size histogram (size: #clusters):",
          dict(sorted(sizes.items())))
    sim_series = pd.Series(sims)
    print("  pairwise-Jaccard deciles (nonzero-overlap pairs):")
    print("   ", {f"p{int(q * 100)}": round(sim_series.quantile(q), 3)
                  for q in [0.5, 0.9, 0.95, 0.99]})
    near = sim_series[(sim_series >= args.jaccard - 0.1) & (sim_series < args.jaccard + 0.1)]
    print(f"  pairs within ±0.1 of threshold: {len(near)} "
          f"(few = threshold sits in a natural gap)")

    # ── per-row probe labels ──────────────────────────────────────────────
    entity_name_rows = extract_entity_names(questions)
    entity_other_rows = extract_entity_other(questions)
    op_cue_rows = [sorted({w.lower() for w in tokens(q)} & OP_CUES) for q in questions]
    function_rows = [sorted({w.lower() for w in tokens(q)} & FUNCTION) for q in questions]
    operators_rows = [parse_operators(e) for e in csv["symbolic_abstraction_answer"]]
    operand_rows = [parse_operands(b) for b in csv["symbol_binding"]]

    meta = pd.DataFrame({
        "svamp_id": [e["ID"] for e in original],
        "svamp_type": [e["Type"] for e in original],
        "seed_group": groups,
        "entity_name_words": [json.dumps(r) for r in entity_name_rows],
        "entity_other_words": [json.dumps(r) for r in entity_other_rows],
        "op_cue_words": [json.dumps(r) for r in op_cue_rows],
        "function_words": [json.dumps(r) for r in function_rows],
        "operators": [json.dumps(r) for r in operators_rows],
        "op_count": [len(r) for r in operators_rows],
        "operand_values": [json.dumps(r) for r in operand_rows],
        "answer": csv["answer"],
    })
    out_path = DATA_DIR / "svamp_probe_metadata.csv"
    meta.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(meta)} rows, aligned to test_svamp.csv by index)")

    # ── probe-feasibility vocab audit ─────────────────────────────────────
    # A word is probe-viable only if both its positives and negatives spread
    # over many seed clusters (the CV unit).
    audit_records = []
    group_arr = pd.Series(groups)
    for category, rows in [
        ("entity_name", entity_name_rows),
        ("entity_other", entity_other_rows),
        ("op_cue", op_cue_rows),
        ("function_word", function_rows),
    ]:
        word_docs: dict[str, list[int]] = defaultdict(list)
        for i, row_words in enumerate(rows):
            for w in row_words:
                word_docs[w].append(i)
        for w, doc_ids in word_docs.items():
            pos_clusters = group_arr.iloc[doc_ids].nunique()
            audit_records.append({
                "category": category,
                "word": w,
                "doc_freq": len(doc_ids),
                "pos_clusters": pos_clusters,
                "neg_clusters": n_clusters - pos_clusters,
            })
    audit = pd.DataFrame(audit_records).sort_values(
        ["category", "doc_freq"], ascending=[True, False])
    audit_path = DATA_DIR / "svamp_probe_vocab_audit.csv"
    audit.to_csv(audit_path, index=False)
    print(f"Wrote {audit_path}")

    print("\n=== probe-viable words per category "
          "(doc_freq>=20 AND pos_clusters>=10) ===")
    for category in ["entity_name", "entity_other", "op_cue", "function_word"]:
        sub = audit[(audit["category"] == category)
                    & (audit["doc_freq"] >= 20) & (audit["pos_clusters"] >= 10)]
        head = ", ".join(f"{r.word}({r.doc_freq})" for r in sub.head(12).itertuples())
        print(f"  {category:14s}: {len(sub):3d} viable | top: {head}")

    print("\n=== operator / operand / answer label coverage ===")
    op_multiset = Counter(tuple(sorted(r)) for r in operators_rows)
    print(f"  operator multisets: {len(op_multiset)} distinct | "
          f"top: {op_multiset.most_common(6)}")
    print(f"  op_count distribution: {dict(Counter(len(r) for r in operators_rows))}")
    print(f"  operand count distribution: {dict(Counter(len(r) for r in operand_rows))}")
    int_answers = sum(float(a) == int(float(a)) for a in csv['answer'])
    print(f"  distinct answers: {csv['answer'].nunique()} | integral: {int_answers}/1000")


if __name__ == "__main__":
    main()
