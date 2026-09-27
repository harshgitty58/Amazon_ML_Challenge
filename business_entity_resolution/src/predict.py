"""Score test candidate pairs and write the two submission files.

usage: python predict.py
Stage 1 -> sibling expansion -> stage 2 (see train.py); writes
output/matching_results.tsv and output/candidate_pairs.tsv.

When XGBoost and CatBoost models are present, predictions are blended with
the weights saved in threshold.json.
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from block import load
from config import WORK_DIR, OUT_DIR, N_JOBS
from train import (feature_cols, assign, stage1_predict, stage2_context,
                   stage2_predict, with_expansion_ensemble)


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
    from ensemble import (predict_xgb, predict_cat, load_xgb, load_cat,
                         blend as blend_preds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(WORK_DIR / 'threshold.json'))
    thr = meta['threshold']
    s1_weights = meta.get('s1_weights', {'lgb': 1.0})
    s2_weights = meta.get('s2_weights', {'lgb': 1.0})

    path = WORK_DIR / 'test_pairs.parquet'

    # ── Load stage-1 models ─────────────────────────────────────────────
    lgb_models = [lgb.Booster(model_file=str(WORK_DIR / f'model_s1_{h}.txt')) for h in (0, 1)]
    import pyarrow.parquet as pq
    feats1 = feature_cols(pd.DataFrame(columns=pq.read_schema(path).names))
    assert feats1 == lgb_models[0].feature_name(), 'feature mismatch between train and test'

    _xpfn = lambda m, X: predict_xgb(m, X, N_JOBS)
    _cpfn = lambda m, X: predict_cat(m, X, N_JOBS)
    pfns = {'lgb': None, 'xgb': _xpfn, 'cat': _cpfn}

    # ── Stage 1: predict with all available models ──────────────────────
    all_s1_models = {'lgb': lgb_models}
    p1_preds = {}
    p1_lgb = stage1_predict(lgb_models, path, feats1, split_by_half=False)
    p1_preds['lgb'] = p1_lgb.p1.values

    xgb_s1_path = WORK_DIR / 'model_s1_0_xgb.json'
    if xgb_s1_path.exists():
        xgb_models = [load_xgb(WORK_DIR / f'model_s1_{h}_xgb.json') for h in (0, 1)]
        all_s1_models['xgb'] = xgb_models
        p1_xgb = stage1_predict(xgb_models, path, feats1, split_by_half=False, predict_fn=_xpfn)
        p1_preds['xgb'] = p1_xgb.p1.values

    cat_s1_path = WORK_DIR / 'model_s1_0_cat.cbm'
    if cat_s1_path.exists():
        cat_models = [load_cat(WORK_DIR / f'model_s1_{h}_cat.cbm') for h in (0, 1)]
        all_s1_models['cat'] = cat_models
        p1_cat = stage1_predict(cat_models, path, feats1, split_by_half=False, predict_fn=_cpfn)
        p1_preds['cat'] = p1_cat.p1.values

    # Blend stage 1
    active_s1 = {k: v for k, v in s1_weights.items() if k in p1_preds}
    p1 = p1_lgb.copy()
    p1['p1'] = blend_preds(p1_preds, active_s1)
    del p1_preds
    print(f'stage 1 blended ({len(all_s1_models)} model types)', flush=True)

    # ── Expansion ───────────────────────────────────────────────────────
    paths, df = with_expansion_ensemble('test', all_s1_models, feats1, path, p1,
                                        active_s1, pfns)
    del p1

    # ── Stage 2: predict with all available models ──────────────────────
    lgb_model2 = lgb.Booster(model_file=str(WORK_DIR / 'model_s2.txt'))
    ctx = stage2_context('test', df)
    feats2 = lgb_model2.feature_name()

    p2_preds = {}
    p2_preds['lgb'] = stage2_predict(lgb_model2, paths, feats2, ctx)

    xgb_s2_path = WORK_DIR / 'model_s2_xgb.json'
    if xgb_s2_path.exists():
        xgb_model2 = load_xgb(xgb_s2_path)
        p2_preds['xgb'] = stage2_predict(xgb_model2, paths, feats2, ctx, predict_fn=_xpfn)

    cat_s2_path = WORK_DIR / 'model_s2_cat.cbm'
    if cat_s2_path.exists():
        cat_model2 = load_cat(cat_s2_path)
        p2_preds['cat'] = stage2_predict(cat_model2, paths, feats2, ctx, predict_fn=_cpfn)

    # Blend stage 2
    active_s2 = {k: v for k, v in s2_weights.items() if k in p2_preds}
    p2 = blend_preds(p2_preds, active_s2)
    print(f'stage 2 blended ({len(p2_preds)} model types)', flush=True)

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
