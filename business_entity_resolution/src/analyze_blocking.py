"""Blocking diagnostics on the training split: pair recall vs. candidate budget.

usage: python analyze_blocking.py
"""
import numpy as np
import pandas as pd

from build_pairs import labels
from config import WORK_DIR, TRAIN_DIR
from features import context_features


def main():
    c = pd.read_parquet(WORK_DIR / 'train_cand_raw.parquet')
    c = context_features(c)
    c = labels('train', c)
    gt = pd.read_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    n_true = int(gt.matched_entity_ids.str.count(',').add(1).where(gt.matched_entity_ids != '', 0).sum())
    pos = c[c.label == 1]
    print(f'raw pairs {len(c)}  true pairs {n_true}  found {len(pos)}  recall {len(pos) / n_true:.4f}')
    print('pairs per query', round(len(c) / c.groupby(['src', 'q_row']).ngroups, 2))
    for k in (1, 2, 3, 5, 8, 12):
        print(f'  recall@rank<={k}: {(pos["rank"] <= k).sum() / n_true:.4f}   pairs {(c["rank"] <= k).sum()}')
    rel = c.score / (c.score + c.gap_top)
    prel = pos.score / (pos.score + pos.gap_top)
    for r in (0.4, 0.5, 0.55, 0.6, 0.7, 0.8):
        for k in (5, 8):
            m = (rel >= r) & (c['rank'] <= k)
            print(f'  rel>={r} rank<={k}: recall {((prel >= r) & (pos["rank"] <= k)).sum() / n_true:.4f} pairs {int(m.sum())}')
    print('rank of true pair:', pos['rank'].value_counts().sort_index().to_dict())
    for country_q in ('ncos', 'acos'):
        print(country_q, 'pos mean', pos[country_q].mean(), 'neg mean', c[c.label == 0][country_q].mean())


if __name__ == '__main__':
    main()
