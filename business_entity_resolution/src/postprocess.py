"""Per-entity match selection by expected F0.5 (replaces the global threshold).

The metric is F0.5 per Source 1 entity, so the right number of matches to keep
depends on the entity's own candidates: with probabilities p_1 >= p_2 >= ...
of the records assigned to it, keeping the top k gives expected F0.5 about
1.25 * sum(p_1..p_k) / (0.25 * sum(all p) + k), and keeping none scores
P(no true match) = prod(1 - p_i).  We keep the k with the highest value.

usage: python postprocess.py        (after predict.py; rewrites matching_results.tsv)
"""
import numpy as np
import pandas as pd

from block import load
from config import WORK_DIR, OUT_DIR
from predict import id_lists, write_tsv
from train import assign

FLOOR = 0.05      # records below this probability are never considered


def ef_select(best):
    """best: one row per S2/S3 record (its most probable S1) with columns s1_row, p.
    Returns the rows to keep as matches."""
    b = best[best.p >= FLOOR].sort_values(['s1_row', 'p'], ascending=[True, False]).copy()
    g = b.groupby('s1_row')
    b['k'] = g.cumcount() + 1
    b['ef'] = 1.25 * g.p.cumsum() / (0.25 * g.p.transform('sum') + b.k)
    p_none = np.exp(np.log1p(-b.p.clip(upper=1 - 1e-6)).groupby(b.s1_row).transform('sum'))
    best_ef = b.groupby('s1_row').ef.transform('max')
    k_best = b.s1_row.map(b[b.ef == best_ef].groupby('s1_row').k.min())
    return b[(b.k <= k_best) & (best_ef > p_none)]


def main():
    df = pd.read_parquet(WORK_DIR / 'test_prob.parquet')
    best = assign(df, df.p2.values, 0.0)
    matches = ef_select(best)
    s1_ids = load('test', 'source1', ['entity_id']).entity_id.values
    q_ids = {i: load('test', f'source{i}', ['entity_id']).entity_id.values for i in (2, 3)}
    write_tsv(OUT_DIR / 'matching_results.tsv', 'matched_entity_ids', s1_ids,
              id_lists(s1_ids, matches, q_ids))
    n_match = matches.groupby('s1_row').size().reindex(np.arange(len(s1_ids)), fill_value=0)
    print(f'expected-F selection: {len(matches)} matches, {(n_match == 0).mean():.3%} of S1 left empty')


if __name__ == '__main__':
    main()
