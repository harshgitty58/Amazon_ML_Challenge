"""Train the two-stage pair classifier and tune the decision threshold for
macro F0.5.

Stage 1: two LightGBM models on pair features, each trained on one half of the
         training queries (split by `qhash`), so every training pair gets an
         out-of-fold probability.  Validation / test use the mean of both.
Between the stages, expand.py adds candidates found through confidently
matched siblings (scored by stage 1 like any other pair).
Stage 2: pair features + competition and sibling features over stage-1
         probabilities (stack.py), trained on out-of-fold training pairs.

usage: python train.py [--stage2]   (--stage2: retrain stage 2 only, reusing stage-1 outputs)
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from block import load
from build_pairs import s1_fold, labels, query_hash
from expand import expand
from config import WORK_DIR, TRAIN_DIR, SEED, N_JOBS
from stack import prob_context, sibling_features, QueryStrings, S1_PARAMS, S1_ROUNDS, S2_PARAMS, S2_ROUNDS

NON_FEATURES = {'q_row', 's1_row', 'label', 'is_val', 'qhash'}
PAIRS = WORK_DIR / 'train_pairs.parquet'
KEYS = ['src', 'q_row', 's1_row']
# stage-1 half h trains on qhash in [500h, 500h + S1_TRAIN_SPAN); stage 2 on qhash < S2_TRAIN_SPAN
S1_TRAIN_SPAN = 350
S2_TRAIN_SPAN = 350
N_EARLY = 1_500_000


def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]


def assign(df, prob, thr):
    """Each S2/S3 record goes to its most probable S1 if prob >= thr."""
    d = pd.DataFrame({'src': df.src.values, 'q_row': df.q_row.values,
                      's1_row': df.s1_row.values, 'p': prob})
    d = d.sort_values('p', ascending=False).drop_duplicates(['src', 'q_row'])
    return d[d.p >= thr]


def macro_f05(pred, truth, s1_rows):
    """pred/truth: DataFrames (s1_row, key); averaged over every S1 in s1_rows."""
    tp = pred.merge(truth, on=['s1_row', 'key']).groupby('s1_row').size()
    npred = pred.groupby('s1_row').size()
    ntrue = truth.groupby('s1_row').size()
    idx = pd.Index(s1_rows)
    tp = tp.reindex(idx, fill_value=0).values.astype(float)
    npred = npred.reindex(idx, fill_value=0).values.astype(float)
    ntrue = ntrue.reindex(idx, fill_value=0).values.astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(npred > 0, tp / npred, 0.0)
        r = np.where(ntrue > 0, tp / ntrue, 0.0)
        f = np.where(p + r > 0, 1.25 * p * r / (0.25 * p + r), 0.0)
    f = np.where((npred == 0) & (ntrue == 0), 1.0, f)
    return f.mean()


def val_truth():
    """(s1_row, key) for every true match of a fold-0 S1, plus the fold-0 S1 rows."""
    gt = pd.read_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    s1 = load('train', 'source1', ['entity_id'])
    s1_pos = pd.Series(s1.row.values, index=s1.entity_id.values)
    gt['s1_row'] = s1_pos.reindex(gt.source1_entity_id).values
    gt = gt[s1_fold(gt.s1_row.values) == 0]
    s1_rows = gt.s1_row.values
    ex = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m')
    ex = ex[ex.m.str.len() > 0]
    qpos = {}
    for src_i, src in ((2, 'source2'), (3, 'source3')):
        q = load('train', src, ['entity_id'])
        qpos[src_i] = pd.Series(q.row.values, index=q.entity_id.values)
    src = np.where(ex.m.str.startswith('S2'), 2, 3)
    q_row = np.where(src == 2, qpos[2].reindex(ex.m).values, qpos[3].reindex(ex.m).values)
    truth = pd.DataFrame({'s1_row': ex.s1_row.values,
                          'key': src.astype(np.int64) * 1_000_000_000 + q_row.astype(np.int64)})
    return truth, s1_rows


def evaluate(val, prob, truth, s1_rows, thrs):
    res = {}
    for thr in thrs:
        a = assign(val, prob, thr)
        pred = pd.DataFrame({'s1_row': a.s1_row.values,
                             'key': a.src.values.astype(np.int64) * 1_000_000_000 + a.q_row.values})
        # only predictions attached to validation S1s are scored
        pred = pred[np.isin(pred.s1_row.values, s1_rows)]
        res[thr] = macro_f05(pred, truth, s1_rows)
    return res


def _f32(df):
    for c in df.columns:
        if df[c].dtype == np.float64:
            df[c] = df[c].astype(np.float32)
    return df


def read_pairs(filters, cols=None):
    t = pq.read_table(PAIRS, columns=cols, filters=filters)
    return _f32(t.to_pandas())


def fit(params, X, y, Xv, yv, rounds):
    params = dict(params, num_threads=N_JOBS, seed=SEED)
    dtr = lgb.Dataset(X, y, params=params, free_raw_data=True).construct()
    dva = lgb.Dataset(Xv, yv, params=params, reference=dtr).construct()
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])


def early_slice(feats, extra=None):
    """A fixed random slice of validation pairs for early stopping."""
    v = read_pairs([('is_val', '=', 1)], feats + ['label'] + KEYS[1:])
    idx = np.sort(np.random.default_rng(SEED).choice(len(v), min(N_EARLY, len(v)), replace=False))
    v = v.iloc[idx].reset_index(drop=True)
    if extra is not None:
        v = pd.concat([v, extra.iloc[idx].reset_index(drop=True)], axis=1)
    return v, idx


def stage1(feats, t0):
    models = []
    vs, _ = early_slice(feats)
    for h in (0, 1):
        lo = 500 * h
        tr = read_pairs([('is_val', '=', 0), ('qhash', '>=', lo), ('qhash', '<', lo + S1_TRAIN_SPAN)],
                        feats + ['label'])
        print(f'stage 1 half {h}: {len(tr)} pairs (pos {tr.label.mean():.3f})', flush=True)
        m = fit(S1_PARAMS, tr[feats], tr.label, vs[feats], vs.label, S1_ROUNDS)
        del tr
        m.save_model(str(WORK_DIR / f'model_s1_{h}.txt'))
        models.append(m)
        print(f'  trained ({time.time() - t0:.0f}s)', flush=True)
    return models


def stage1_predict(models, path, feats, split_by_half):
    """Stage-1 probability of every pair in `path`, in file order.
    split_by_half: out-of-fold for training pairs, mean of both for the rest."""
    pf = pq.ParquetFile(path)
    extra = ['is_val', 'qhash'] if split_by_half else []
    out = []
    for b in pf.iter_batches(batch_size=1_000_000, columns=feats + [k for k in KEYS if k not in feats] + extra):
        b = _f32(b.to_pandas())
        X = b[feats]
        if split_by_half:
            p = np.empty(len(b), np.float32)
            oof = b.is_val.values == 0
            half = (b.qhash.values >= 500).astype(int)
            for h in (0, 1):
                m = oof & (half == h)          # trained on half h -> scored by the other model
                if m.any():
                    p[m] = models[1 - h].predict(X[m], num_threads=N_JOBS)
            if (~oof).any():
                p[~oof] = 0.5 * (models[0].predict(X[~oof], num_threads=N_JOBS)
                                 + models[1].predict(X[~oof], num_threads=N_JOBS))
        else:
            p = 0.5 * (models[0].predict(X, num_threads=N_JOBS) + models[1].predict(X, num_threads=N_JOBS))
        out.append(pd.DataFrame({'src': b.src.values, 'q_row': b.q_row.values,
                                 's1_row': b.s1_row.values, 'p1': p.astype(np.float32)}))
    return pd.concat(out, ignore_index=True)


def stage2_context(split, df):
    """Probability-competition + sibling features for every pair of a split;
    df has src, q_row, s1_row, p1 and n_keys (expansion key hits, 0 = blocking)."""
    args = (df.src.values, df.q_row.values, df.s1_row.values, df.p1.values)
    ctx = prob_context(*args)
    strings = QueryStrings(split)
    sib = sibling_features(*args, strings)
    del strings
    ctx = pd.concat([ctx, sib], axis=1)
    ctx['n_keys'] = df.n_keys.values.astype(np.int8)
    return ctx


def with_expansion(split, models, feats1, path, p1):
    """Add sibling-expansion candidates.  Writes <split>_pairs_x.parquet and
    returns (paths, union of pairs with p1 and n_keys)."""
    old = pq.read_table(path, columns=['src', 'q_row', 's1_row', 'score', 'ncos', 'acos']).to_pandas()
    x = expand(split, p1, old)
    del old
    paths = [path]
    p1 = p1.assign(n_keys=np.int8(0))
    if x is None:
        return paths, p1
    xpath = WORK_DIR / f'{split}_pairs_x.parquet'
    cols = [c for c in feats1 if c not in KEYS] + KEYS + ['n_keys']
    if split == 'train':
        x = labels(split, x)
        qkey = x.src.values.astype(np.int64) * 1_000_000_000 + x.q_row.values
        x['qhash'] = query_hash(qkey)
        x['is_val'] = np.isin(qkey, val_queries(p1, x)).astype(np.int8)
        cols += ['label', 'qhash', 'is_val']
    x[cols].to_parquet(xpath, index=False)
    px = stage1_predict(models, xpath, feats1, split_by_half=(split == 'train'))
    px['n_keys'] = x.n_keys.values.astype(np.int8)
    del x
    return paths + [xpath], pd.concat([p1, px], ignore_index=True)


def val_queries(*frames):
    """Queries with any candidate in the validation S1 fold (over all frames)."""
    ks = [f.src.values.astype(np.int64) * 1_000_000_000 + f.q_row.values
          for f in frames]
    s1 = np.concatenate([f.s1_row.values for f in frames])
    k = np.concatenate(ks)
    return np.unique(k[s1_fold(s1) == 0])


def iter_rows(paths, cols, mask, batch=1_000_000):
    """Stream the rows of the concatenated files where mask is true."""
    pos = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for b in pf.iter_batches(batch_size=batch, columns=cols):
            n = b.num_rows
            sl = slice(pos, pos + n)
            pos += n
            keep = mask[sl]
            if keep.any():
                yield sl, keep, _f32(b.to_pandas())


def load_rows(paths, cols, mask, ctx=None):
    out = []
    for sl, keep, b in iter_rows(paths, cols, mask):
        if ctx is not None:
            b = pd.concat([b, ctx.iloc[sl].reset_index(drop=True)], axis=1)
        out.append(b[keep])
    return pd.concat(out, ignore_index=True)


def stage2_predict(model, paths, feats, ctx, mask=None):
    """Stage-2 probability of the masked pairs of the concatenated files; ctx aligned."""
    if mask is None:
        mask = np.ones(len(ctx), bool)
    out = []
    for sl, keep, b in iter_rows(paths, [f for f in feats if f not in ctx.columns], mask):
        X = pd.concat([b, ctx.iloc[sl].reset_index(drop=True)], axis=1)[feats]
        out.append(model.predict(X[keep], num_threads=N_JOBS).astype(np.float32))
    return np.concatenate(out)


def main(stage2_only=False):
    t0 = time.time()
    feats1 = feature_cols(pd.DataFrame(columns=pq.read_schema(PAIRS).names))
    meta = pq.read_table(PAIRS, columns=['qhash', 'label']).to_pandas()
    xpath = WORK_DIR / 'train_pairs_x.parquet'
    if stage2_only:
        # reuse stage-1 models, their out-of-fold probabilities and the expansion pairs
        u = pd.read_parquet(WORK_DIR / 'train_p1.parquet')
        paths = [PAIRS] + ([xpath] if xpath.exists() else [])
    else:
        models = stage1(feats1, t0)
        p1 = stage1_predict(models, PAIRS, feats1, split_by_half=True)
        print(f'stage 1 scored {len(p1)} pairs ({time.time() - t0:.0f}s)', flush=True)
        paths, u = with_expansion('train', models, feats1, PAIRS, p1)
        del p1
        u.to_parquet(WORK_DIR / 'train_p1.parquet', index=False)
    if len(paths) > 1:
        mx = pq.read_table(paths[1], columns=['qhash', 'label']).to_pandas()
        meta = pd.concat([meta, mx], ignore_index=True)
    print(f'union {len(u)} pairs ({time.time() - t0:.0f}s)', flush=True)
    ctx = stage2_context('train', u)
    feats2 = feats1 + list(ctx.columns)
    print(f'stage 2 context built ({time.time() - t0:.0f}s)', flush=True)

    qkey = u.src.values.astype(np.int64) * 1_000_000_000 + u.q_row.values
    is_val = np.isin(qkey, val_queries(u))
    truth, s1_rows = val_truth()
    va_keys = u[is_val].reset_index(drop=True)
    thrs = np.round(np.arange(0.3, 0.91, 0.05), 2)
    r1 = evaluate(va_keys, va_keys.p1.values, truth, s1_rows, thrs)
    print('stage 1 val:', {k: round(v, 5) for k, v in r1.items()}, flush=True)

    # stage 2 training rows: non-validation queries with qhash < S2_TRAIN_SPAN
    tr_mask = (~is_val) & (meta.qhash.values < S2_TRAIN_SPAN)
    tr = load_rows(paths, feats1 + ['label'], tr_mask, ctx)
    y = tr.label.values
    X = tr[feats2]
    del tr
    ev = np.zeros(len(u), bool)
    vi = np.flatnonzero(is_val)
    ev[np.random.default_rng(SEED).choice(vi, min(N_EARLY, len(vi)), replace=False)] = True
    vs = load_rows(paths, feats1 + ['label'], ev, ctx)
    print(f'stage 2: {len(X)} training pairs, pos {y.mean():.3f} ({time.time() - t0:.0f}s)', flush=True)
    model = fit(S2_PARAMS, X, y, vs[feats2], vs.label, S2_ROUNDS)
    del X, vs
    model.save_model(str(WORK_DIR / 'model_s2.txt'))
    print(f'stage 2 trained ({time.time() - t0:.0f}s)', flush=True)

    p2 = stage2_predict(model, paths, feats2, ctx, mask=is_val)
    res = evaluate(va_keys, p2, truth, s1_rows, thrs)
    for k, v in res.items():
        print(f'thr {k:.2f}: macro F0.5 {v:.5f}')
    best = max(res, key=res.get)
    fine = evaluate(va_keys, p2, truth, s1_rows, np.round(np.arange(best - 0.05, best + 0.051, 0.01), 3))
    best = max(fine, key=fine.get)
    print(f'best thr {best:.3f}: macro F0.5 {fine[best]:.5f}')
    imp = pd.Series(model.feature_importance('gain'), index=feats2).sort_values(ascending=False)
    print(imp.head(25).to_string())
    with open(WORK_DIR / 'threshold.json', 'w') as f:
        json.dump({'threshold': float(best), 'val_macro_f05': float(fine[best]),
                   'val_macro_f05_stage1': float(max(r1.values())),
                   'best_iter_s2': model.best_iteration}, f)
    va_keys.assign(label=meta.label.values[is_val], prob=p2).to_parquet(
        WORK_DIR / 'val_pred.parquet', index=False)
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    import sys
    main(stage2_only='--stage2' in sys.argv)
