"""Normalise every source file of a split into work/<split>_<source>.parquet.

usage: python prep.py train|test
"""
import json
import sys
import time
from multiprocessing import Pool

import pandas as pd

from config import TRAIN_DIR, TEST_DIR, WORK_DIR, N_JOBS
import normalize as N


def _init(translit):
    N.set_translit(translit)


def _norm_chunk(df):
    names = [N.normalize_name(x) for x in df.business_name]
    addrs = [N.normalize_address(a, c) for a, c in zip(df.business_address, df.country)]
    out = pd.concat([pd.DataFrame(names), pd.DataFrame(addrs)], axis=1)
    out.insert(0, 'country', df.country.values)
    out.insert(0, 'entity_id', df.entity_id.values)
    for c in ('n_nonlatin', 'a_empty'):
        out[c] = out[c].astype('int8')
    return out


def prep_file(path, out_path, translit):
    t = time.time()
    reader = pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, chunksize=20_000)
    with Pool(N_JOBS, initializer=_init, initargs=(translit,)) as pool:
        parts = list(pool.imap(_norm_chunk, reader))
    df = pd.concat(parts, ignore_index=True)
    df['country'] = df['country'].astype('category')
    df.to_parquet(out_path, index=False)
    print(f'{path.name}: {len(df)} rows in {time.time() - t:.0f}s', flush=True)


def main(split):
    d = TRAIN_DIR if split == 'train' else TEST_DIR
    with open(WORK_DIR / 'translit.json', encoding='utf-8') as f:
        translit = json.load(f)
    for src in ('source1', 'source2', 'source3'):
        prep_file(d / f'{split}_{src}.tsv', WORK_DIR / f'{split}_{src}.parquet', translit)


if __name__ == '__main__':
    main(sys.argv[1])
