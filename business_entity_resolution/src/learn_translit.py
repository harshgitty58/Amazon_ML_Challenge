"""Learn a non-Latin -> Latin token dictionary from the training ground truth.

For every matched (S1, S2/S3) pair whose S2/S3 record contains non-Latin
tokens, count co-occurrence of each non-Latin token with the Latin tokens of
the paired S1 record.  A token t is mapped to the Latin token w that appears
with it in most pairs (P(w|t) high) and is not simply a frequent word.
Only the provided training data is used.
"""
import collections
import json
import random
import re
import sys

import pandas as pd
from rapidfuzz import fuzz
from unidecode import unidecode

from config import TRAIN_DIR, WORK_DIR
from normalize import raw_tokens, is_nonlatin, skeleton

NONLATIN_RE = re.compile(r'[ऀ-෿]')


def main(max_pairs=300_000, keep_frac=0.35):
    rng = random.Random(0)
    gt = pd.read_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    parent = {}
    for s1, ms in zip(gt.source1_entity_id, gt.matched_entity_ids):
        if ms:
            for m in ms.split(','):
                parent[m] = s1
    del gt

    recs = []  # (s1_id, nonlatin tokens)
    for src in ('source2', 'source3'):
        for ch in pd.read_csv(TRAIN_DIR / f'train_{src}.tsv', sep='\t', dtype=str,
                              keep_default_na=False, chunksize=500_000):
            txt = ch.business_name + ' , ' + ch.business_address
            mask = txt.str.contains(NONLATIN_RE)
            for eid, t in zip(ch.entity_id[mask], txt[mask]):
                if rng.random() > keep_frac:
                    continue
                p = parent.get(eid)
                if p is None:
                    continue
                toks = {tok for tok in raw_tokens(t) if is_nonlatin(tok)}
                if toks:
                    recs.append((p, toks))
    recs = recs[:max_pairs]
    need = {p for p, _ in recs}
    print('pairs with non-latin tokens:', len(recs), file=sys.stderr)

    s1_tok = {}
    for ch in pd.read_csv(TRAIN_DIR / 'train_source1.tsv', sep='\t', dtype=str,
                          keep_default_na=False, chunksize=500_000):
        ch = ch[ch.entity_id.isin(need)]
        for eid, n, a in zip(ch.entity_id, ch.business_name, ch.business_address):
            toks = [unidecode(t).lower() for t in raw_tokens(n + ' , ' + a)]
            # bigrams let one native token map to a two-word Latin phrase ("tamil nadu")
            s1_tok[eid] = set(toks) | {toks[i] + ' ' + toks[i + 1] for i in range(len(toks) - 1)}

    ct = collections.Counter()
    cw = collections.Counter()
    ctw = collections.Counter()
    for p, toks in recs:
        ws = s1_tok.get(p)
        if not ws:
            continue
        cw.update(ws)
        ct.update(toks)
        for t in toks:
            for w in ws:
                ctw[(t, w)] += 1
    n = len(recs)
    best = {}
    skel_cache = {}

    def skel(s):
        if s not in skel_cache:
            skel_cache[s] = ''.join(skeleton(x) for x in unidecode(s).lower().split())
        return skel_cache[s]

    for (t, w), c in ctw.items():
        if ct[t] < 2:
            continue
        p = c / ct[t]
        if p < 0.25:
            continue
        lift = p - cw[w] / n              # conditional prob minus base rate of w
        sim = fuzz.ratio(skel(t), skel(w)) / 100.0   # phonetic agreement
        if lift < 0.45 and sim < 0.5:
            continue
        score = lift + p + 2 * sim
        if score > best.get(t, (None, -1))[1]:
            best[t] = (w, score)
    mapping = {t: w for t, (w, s) in best.items()}
    print('learned tokens:', len(mapping), file=sys.stderr)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with open(WORK_DIR / 'translit.json', 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
