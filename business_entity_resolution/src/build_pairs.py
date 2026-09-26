"""Turn raw blocking output into the final candidate set + pair features.

usage: python build_pairs.py train|test

train: features for every candidate, plus labels, `is_val` (every candidate of
       every query touching a validation-fold S1) and `qhash` (a per-query
       hash in [0, 1000) used to split and subsample training queries).
test : features for every candidate.
"""
import sys
import time

import numpy as np
import pandas as pd

from block import load
from config import WORK_DIR, TRAIN_DIR, SEED
from features import REC_COLS, pair_features, context_features

# Final candidate set = what the matching model runs inference over.
FINAL_K = 8           # max S1 candidates per S2/S3 record
REL_SCORE = 0.6       # keep candidates scoring >= 60% of the best score for that record
ADDR_KEEP_K = 4       # ...plus the best few by address cosine alone
ADDR_KEEP_MIN = 0.4
N_FOLDS = 10          # fold 0 of S1 = validation
CHUNK = 1_500_000


def prune(c):
    c = context_features(c)
    keep = (c['rank'] <= FINAL_K) & (c.score >= REL_SCORE * (c.score + c.gap_top))
    keep |= (c.acos_rank <= ADDR_KEEP_K) & (c.acos >= ADDR_KEEP_MIN)
    c = c[keep].reset_index(drop=True)
    # recompute competition features on the final set
    return context_features(c)


def s1_fold(s1_row):
    return (s1_row.astype(np.int64) * 2654435761 % 2 ** 32) % N_FOLDS


def query_hash(qkey):
    return ((qkey.astype(np.int64) * 40503 + SEED) % 2 ** 31 * 2654435761 % 2 ** 32 % 1000).astype(np.int16)


def labels(split, c):
    gt = pd.read_csv(TRAIN_DIR / 'train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False)
    s1ids = load(split, 'source1', ['entity_id'])
    s1_pos = pd.Series(s1ids.row.values, index=s1ids.entity_id.values)
    q_pos = {}
    for src_i, src in ((2, 'source2'), (3, 'source3')):
        q = load(split, src, ['entity_id'])
        q_pos[src_i] = pd.Series(q.row.values, index=q.entity_id.values)
    ex = gt.assign(m=gt.matched_entity_ids.str.split(',')).explode('m')
    ex = ex[ex.m.str.len() > 0]
    src = np.where(ex.m.str.startswith('S2'), 2, 3).astype(np.int8)
    q_row = np.where(src == 2, q_pos[2].reindex(ex.m).values, q_pos[3].reindex(ex.m).values)
    parent = pd.DataFrame({'src': src, 'q_row': q_row.astype(np.int64),
                           'true_s1': s1_pos.reindex(ex.source1_entity_id).values.astype(np.int64)})
    c = c.merge(parent, on=['src', 'q_row'], how='left')
    c['label'] = (c.true_s1 == c.s1_row).astype(np.int8)
    return c.drop(columns='true_s1')


def featurize(split, c, out_path):
    """Compute pair features one (country, source) slice at a time and append
    each chunk to a single parquet file, so memory stays bounded."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    s1_country = load(split, 'source1', ['country']).country.astype(str).values
    c_country = s1_country[c.s1_row.values]
    writer = None
    try:
        for country in sorted(set(c_country)):
            for src_i, src in ((2, 'source2'), (3, 'source3')):
                part = c[(c_country == country) & (c.src.values == src_i)]
                if not len(part):
                    continue
                S1 = load(split, 'source1', REC_COLS, rows=np.unique(part.s1_row.values)).set_index('row')
                Q = load(split, src, REC_COLS, rows=np.unique(part.q_row.values)).set_index('row')
                for s in range(0, len(part), CHUNK):
                    p = part.iloc[s:s + CHUNK]
                    t = time.time()
                    f = pair_features(S1.loc[p.s1_row.values].reset_index(drop=True),
                                      Q.loc[p.q_row.values].reset_index(drop=True))
                    tbl = pa.Table.from_pandas(pd.concat([p.reset_index(drop=True), f], axis=1),
                                               preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, tbl.schema)
                    writer.write_table(tbl.cast(writer.schema))
                    del f, tbl
                    print(f'  features {country} {src} {s + len(p)}/{len(part)} ({time.time() - t:.0f}s)',
                          flush=True)
                del S1, Q
    finally:
        if writer is not None:
            writer.close()


def main(split):
    t0 = time.time()
    c = pd.read_parquet(WORK_DIR / f'{split}_cand_raw.parquet')
    c = prune(c)
    print(f'final candidates: {len(c)}', flush=True)
    # final candidate set (written to candidate_pairs.tsv for test)
    c[['src', 'q_row', 's1_row']].to_parquet(WORK_DIR / f'{split}_cand_final.parquet', index=False)
    if split == 'train':
        c = labels(split, c)
        # validation = every candidate of every query touching a fold-0 S1, so
        # the per-query "best parent" decision sees all its competitors
        qkey = c.src.values.astype(np.int64) * 1_000_000_000 + c.q_row.values
        val_q = np.unique(qkey[s1_fold(c.s1_row.values) == 0])
        is_val = np.isin(qkey, val_q)
        c['is_val'] = is_val.astype(np.int8)
        c['qhash'] = query_hash(qkey)
        print(f'train pairs {int((~is_val).sum())}, val pairs {int(is_val.sum())}', flush=True)
    featurize(split, c, WORK_DIR / f'{split}_pairs.parquet')
    print(f'done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
