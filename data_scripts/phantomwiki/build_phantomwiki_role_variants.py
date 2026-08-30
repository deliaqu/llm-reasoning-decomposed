"""Build role-annotated PhantomWiki variants for the hypothesis-free word-role probe.

Design (no stage assignment a priori — the probe trajectories decide):
  - Each of the 24 easy universes is emitted as ORIGINAL + 2 RENAMED COPIES.
    Renaming is a family-preserving bijection (first names -> shared bank,
    surnames -> shared bank), applied in ONE simultaneous pass over articles,
    questions, answers, and solution traces. Same graph, same facts, same
    difficulty — names become swappable decoration-or-operands, and the
    shared bank makes each bank name recur across universes and roles, which
    is what gives per-name probe targets cross-group support.
  - Every question row carries a mechanical ROLE annotation for each name,
    derived from the generator's solution trace (no human judgment):
        anchor        first name appears in the question text
        intermediate  on the solution chain, not in question, not the answer
        answer        the answer entity's first name (who-questions)
        bystander     in this copy's corpus, none of the above
  - 50% of rows (deterministic hash) get a FILLER INSERTION between corpus
    and question: 2 true-but-irrelevant facts about bystanders ("The hobby of
    X is H."). True facts avoid contradictions; adjacency to the question
    puts them within the answer-position's early aggregation reach — unlike
    mid-corpus content. Inserted names/values are recorded per row.

Output: data/test_phantomwiki_roles.csv (+ per-row JSON role columns).
Usage:  ./submit.sh data_scripts/phantomwiki/build_phantomwiki_role_variants.sh
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN_DIR = PROJECT_ROOT / "data" / "phantomwiki" / "generated"
OUT_CSV = PROJECT_ROOT / "data" / "test_phantomwiki_roles.csv"

SEEDS = tuple(range(4, 28))
N_RENAMED = 2          # copies per universe, in addition to the original
INSERT_FRACTION = 0.5  # of rows receiving filler insertions
N_INSERT_FACTS = 2
SEED = 0

# Shared banks. Deliberately common English names; any entry colliding with
# an original name in the 24 universes is dropped at runtime (asserted below).
FIRST_BANK = [
    "Oliver", "Amelia", "Theo", "Isla", "Felix", "Clara", "Hugo", "Nora",
    "Jasper", "Ivy", "Silas", "Maeve", "Otto", "Freya", "Casper", "Wren",
    "Milo", "Sadie", "Ezra", "Lena", "Rufus", "Petra", "Basil", "Greta",
    "Cyrus", "Nadia", "Edwin", "Celia", "Victor", "Daphne", "Hector", "Rosa",
    "Leon", "Alma", "Bruno", "Vera", "Elias", "Thea", "Marcel", "Ines",
    "Anton", "Livia", "Oscar", "Mira", "Edgar", "Selma", "Louis", "Ida",
    "Ferdinand", "Astrid", "Conrad", "Elsa", "Gustav", "Marta", "Emil", "Ruth",
    "Albert", "Hanna", "Arthur", "Nina",
    "Ansel", "Bram", "Cosmo", "Dashiell", "Eamon", "Fintan", "Gideon",
    "Horatio", "Ignatius", "Jorah", "Kellan", "Lysander", "Magnus", "Nikolai",
    "Osric", "Percival", "Quentin", "Roderick", "Soren", "Tobias", "Ulric",
    "Vaughn", "Wendell", "Xavier", "Yusuf", "Zebediah", "Anouk", "Beatrix",
    "Cordelia", "Delphine", "Esme", "Fenella", "Ginevra", "Imogen",
    "Josephine", "Katinka", "Lucinda", "Mirabel", "Odette", "Perpetua",
    "Quilla", "Rosalind", "Saskia", "Tamsin", "Ursula", "Vivienne",
    "Wilhelmina", "Xenia", "Yvette", "Zelda",
]
LAST_BANK = [
    "Ashford", "Blackwood", "Carraway", "Dunmore", "Ellsworth", "Fairbanks",
    "Galloway", "Hargrove", "Ironside", "Jennings", "Kingsley", "Lockhart",
    "Marchbanks", "Northcott", "Oakhurst", "Pemberton", "Quimby", "Ravenswood",
    "Silverton", "Thornbury", "Underhill", "Vanterpool", "Westbrook", "Yardley",
    "Aldergate", "Birchall", "Cresswell", "Dovetail", "Eastgate", "Foxwell",
]

PROMPT_HEADER = (
    "Read the following articles about a fictional universe, then answer the "
    "question using only the information in the articles.\n\n"
)


def load_universe(seed: int):
    d = GEN_DIR / f"seed_{seed}"
    arts = json.load(open(d / "articles.json"))
    qs = json.load(open(d / "questions.json"))
    return arts, qs


def people_of(arts) -> list[str]:
    return [a["title"] for a in arts]


def simultaneous_sub(text: str, mapping: dict[str, str]) -> str:
    """One-pass word-boundary substitution over all keys (no chained renames)."""
    if not mapping:
        return text
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(mapping, key=len, reverse=True))) + r")\b")
    return pat.sub(lambda m: mapping[m.group(1)], text)


def build_name_mapping(people: list[str], rng, first_bank, last_bank):
    firsts = sorted({p.split()[0] for p in people})
    lasts = sorted({p.split()[-1] for p in people})
    # KNOWN EDGE CASE: a token can be both a first name and a surname in one
    # universe (seed 7: "Jack Karr" + the "... Jack" family). The dict merge
    # below lets the surname mapping win, so that token renames to a LAST_BANK
    # name and appears as a first-name role entry in the copies. Substitution
    # stays globally consistent (audited: no leaks/ambiguity), and such tokens
    # exist in only one universe, so the probe viability gate (>=6 universes)
    # excludes them from targets. A position-aware two-pass substitution would
    # be needed to remove the artifact entirely.
    both = set(firsts) & set(lasts)
    if both:
        print(f"  note: position-ambiguous name tokens {sorted(both)} -> surname mapping wins")
    assert len(firsts) <= len(first_bank) and len(lasts) <= len(last_bank), \
        f"bank too small: need {len(firsts)} firsts / {len(lasts)} lasts"
    new_firsts = rng.choice(first_bank, size=len(firsts), replace=False)
    new_lasts = rng.choice(last_bank, size=len(lasts), replace=False)
    return {**dict(zip(firsts, new_firsts)), **dict(zip(lasts, new_lasts))}


def question_rows(arts, qs, universe: int, copy_id: int, mapping: dict, rng):
    people = people_of(arts)
    name_re = re.compile(r"\b(" + "|".join(re.escape(p) for p in people) + r")\b") if people else None

    context = "\n\n".join(simultaneous_sub(a["article"].strip(), mapping) for a in arts)
    renamed_people = [simultaneous_sub(p, mapping) for p in people]

    rows = []
    for q in qs:
        if len(q["answer"]) != 1:
            continue
        raw_q = simultaneous_sub(q["question"], mapping)
        answer = simultaneous_sub(str(q["answer"][0]), mapping)
        traces = q.get("solution_traces") or "[]"
        if isinstance(traces, str):
            traces = json.loads(traces)
        chain_people = {simultaneous_sub(v, mapping)
                        for tr in traces for v in tr.values()
                        if isinstance(v, str) and " " in v}

        anchor_names = set(name_re.findall(q["question"])) if name_re else set()
        anchor = {simultaneous_sub(a, mapping) for a in anchor_names}
        is_name_answer = answer in renamed_people
        answer_set = {answer} if is_name_answer else set()
        intermediates = chain_people - anchor - answer_set
        bystanders = set(renamed_people) - chain_people - anchor - answer_set

        # deterministic insertion decision + choice (stable across reruns)
        h = int(hashlib.sha1(f"{q['id']}|{copy_id}".encode()).hexdigest(), 16)
        inserted_names, inserted_values, insert_block = [], [], ""
        if (h % 100) < INSERT_FRACTION * 100 and len(bystanders) >= N_INSERT_FACTS:
            fact_rng = np.random.default_rng(h % (2**32))
            picks = fact_rng.choice(sorted(bystanders), size=N_INSERT_FACTS, replace=False)
            facts = []
            for person in picks:
                # pull a true attribute fact for this person from the renamed context
                m = re.search(rf"The (hobby|occupation) of {re.escape(person)} is ([^.\n]+)\.", context)
                if not m:
                    continue
                facts.append(f"The {m.group(1)} of {person} is {m.group(2)}.")
                inserted_names.append(person.split()[0])
                inserted_values.append(m.group(2))
            if facts:
                insert_block = "Note: " + " ".join(facts) + "\n\n"

        rows.append({
            "question": f"{PROMPT_HEADER}{context}\n\n{insert_block}Question: {raw_q}",
            "answer": answer,
            "pw_id": f"{q['id']}|copy{copy_id}",
            "universe": universe,
            "copy_id": copy_id,
            "raw_question": raw_q,
            "difficulty": int(q["difficulty"]),
            "qtype": int(q["type"]),
            "role_anchor": json.dumps(sorted(n.split()[0] for n in anchor)),
            "role_intermediate": json.dumps(sorted(n.split()[0] for n in intermediates)),
            "role_answer": json.dumps(sorted(n.split()[0] for n in answer_set)),
            "role_bystander": json.dumps(sorted({n.split()[0] for n in bystanders})),
            "inserted_names": json.dumps(inserted_names),
            "inserted_values": json.dumps(inserted_values),
            "has_insertion": bool(insert_block),
        })
    return rows


def main():
    rng = np.random.default_rng(SEED)

    # collision guard: bank entries must not exist in any original universe
    all_orig_tokens = set()
    for seed in SEEDS:
        arts, qs = load_universe(seed)
        for p in people_of(arts):
            all_orig_tokens.update(p.split())
    first_bank = [n for n in FIRST_BANK if n not in all_orig_tokens]
    last_bank = [n for n in LAST_BANK if n not in all_orig_tokens]
    print(f"bank after collision filter: {len(first_bank)} firsts, {len(last_bank)} lasts "
          f"(dropped {len(FIRST_BANK)-len(first_bank)}/{len(LAST_BANK)-len(last_bank)})")

    rows = []
    for seed in SEEDS:
        arts, qs = load_universe(seed)
        # copy 0: original names (identity mapping)
        rows += question_rows(arts, qs, seed, 0, {}, rng)
        for c in range(1, N_RENAMED + 1):
            mapping = build_name_mapping(people_of(arts), rng, first_bank, last_bank)
            new = question_rows(arts, qs, seed, c, mapping, rng)
            # QA: no original surname/first name may survive in renamed copies
            leaked = [t for t in {p.split()[0] for p in people_of(arts)}
                      if re.search(rf"\b{re.escape(t)}\b", new[0]["question"])]
            assert not leaked, f"seed {seed} copy {c}: leaked original names {leaked[:5]}"
            rows += new
        print(f"seed {seed}: {len(rows)} cumulative rows")

    df = pd.DataFrame(rows)
    assert df["pw_id"].is_unique
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}: {len(df)} rows "
          f"({(df.copy_id == 0).sum()} original + {(df.copy_id > 0).sum()} renamed)")
    print(f"insertion rows: {df.has_insertion.sum()} ({df.has_insertion.mean():.0%})")

    # bank-name role-support audit (drives probe viability)
    from collections import Counter
    for role in ["role_anchor", "role_answer", "role_intermediate", "inserted_names"]:
        c = Counter(n for s in df[role] for n in json.loads(s) if n in set(first_bank))
        top = c.most_common(5)
        print(f"{role:18s}: {len(c):3d} bank names used | top {top}")


if __name__ == "__main__":
    main()
