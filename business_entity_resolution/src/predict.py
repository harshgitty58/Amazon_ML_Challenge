"""Score test candidate pairs and write the two submission files.

usage: python predict.py
Stage 1 -> sibling expansion -> stage 2 (see train.py); writes
output/matching_results.tsv and output/candidate_pairs.tsv
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from block import load
from config import WORK_DIR, OUT_DIR, N_JOBS
from train import feature_cols, assign, stage1_predict, stage2_context, stage2_predict, with_expansion


def id_lists(s1_ids, pairs, q_ids):
    """pairs: DataFrame(s1_row, src, q_row) -> Series of comma-joined IDs per S1 row."""
    src, q_row = pairs.src.values, pairs.q_row.values
    ids = np.empty(len(pairs), dtype=object)
    for s in (2, 3):
        m = src == s
        ids[m] = q_ids[s][q_row[m]]
    d = pd.DataFrame({'s1_row': pairs.s1_row.values, 'id': ids}).drop_duplicates()
    joined = d.groupby('s1_row').id.agg(','.join)
    return joined.reindex(np.arange(len(s1_ids)), fill_value='').values


def write_tsv(path, col, s1_ids, lists):
    # plain writer: no quoting, tab separator, empty field for no matches
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(f'source1_entity_id\t{col}\n')
        for a, b in zip(s1_ids, lists):
            f.write(f'{a}\t{b}\n')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    thr = json.load(open(WORK_DIR / 'threshold.json'))['threshold']
    path = WORK_DIR / 'test_pairs.parquet'
    models = [lgb.Booster(model_file=str(WORK_DIR / f'model_s1_{h}.txt')) for h in (0, 1)]
    model2 = lgb.Booster(model_file=str(WORK_DIR / 'model_s2.txt'))
    import pyarrow.parquet as pq
    feats1 = feature_cols(pd.DataFrame(columns=pq.read_schema(path).names))
    assert feats1 == models[0].feature_name(), 'feature mismatch between train and test'
    p1 = stage1_predict(models, path, feats1, split_by_half=False)
    paths, df = with_expansion('test', models, feats1, path, p1)
    del p1
    ctx = stage2_context('test', df)
    feats2 = model2.feature_name()
    p2 = stage2_predict(model2, paths, feats2, ctx)
    df.assign(p2=p2).to_parquet(WORK_DIR / 'test_prob.parquet', index=False)
    del ctx
    matches = assign(df, p2, thr)
    # the candidate set is everything the model scored: blocking + sibling expansion
    cand = df[['src', 'q_row', 's1_row']]

    s1_ids = load('test', 'source1', ['entity_id']).entity_id.values
    q_ids = {i: load('test', f'source{i}', ['entity_id']).entity_id.values for i in (2, 3)}
    write_tsv(OUT_DIR / 'candidate_pairs.tsv', 'candidate_entity_ids', s1_ids,
              id_lists(s1_ids, cand, q_ids))
    write_tsv(OUT_DIR / 'matching_results.tsv', 'matched_entity_ids', s1_ids,
              id_lists(s1_ids, matches, q_ids))
    n_match = (matches.groupby('s1_row').size().reindex(np.arange(len(s1_ids)), fill_value=0))
    print(f'threshold {thr}: {len(matches)} matches, '
          f'{(n_match == 0).mean():.3%} of S1 left empty')


if __name__ == '__main__':
    main()
