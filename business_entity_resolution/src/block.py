"""Candidate generation (blocking).

Every record is embedded as two sparse TF-IDF vectors (feature-hashed tokens,
IDF computed on Source 1 of the same country):

  name    : core-name words, consonant skeletons, word bigrams, the compact
            (space-less) name / website domain, DBA/AKA alternative name words
  address : address tokens (words + cleaned numbers) and consecutive bigrams,
            6-digit PIN codes, consonant skeletons of address words
  name also gets numbers inside the name and character 4-grams of the compact
  name (catches glued / split words such as "chiropracticcare").

The two halves are L2-normalised separately and concatenated, so the dot
product is the mean of name-cosine and address-cosine.  For every Source 2/3
record we keep the top-K Source 1 records *of the same country* by that score
(sparse_dot_topn, multithreaded), plus the top-ADDR_K by address cosine alone,
so records whose name is unrelated or in another script still meet their
parent through the address.  Blocking is country-partitioned but treats
country as an open label set (France is handled like any other value).

usage: python block.py train|test
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2norm
from sparse_dot_topn import sp_matmul_topn

from config import WORK_DIR, N_JOBS
from normalize import skeleton

N_FEATURES = 2 ** 23
TOP_K = 12
MIN_SCORE = 0.05
REL_MIN = 0.4       # keep candidates scoring >= 40% of the query's best
MAX_DF = 5000           # tokens in more S1 records than this are dropped from blocking (speed)
QUERY_CHUNK = 200_000
ADDR_K = 6          # extra address-only channel
GRAM = 4


def name_tokens(row):
    core, skel, compact, domain, alt, norm = row
    toks = ['n' + t for t in core.split()]
    toks += ['s' + t for t in skel.split()]
    w = core.split()
    toks += ['b' + w[i] + '_' + w[i + 1] for i in range(len(w) - 1)]
    if len(compact) >= 4:
        toks.append('c' + compact)
    if len(domain) >= 3:
        toks.append('c' + domain)
    toks += ['n' + t for t in alt.split()]
    toks += ['#' + t for t in norm.split() if t.isdigit()]
    if len(compact) >= 4:
        g = '^' + compact + '$'
        toks += ['g' + g[i:i + GRAM] for i in range(len(g) - GRAM + 1)]
    return toks


def addr_tokens(row):
    a_norm, postal, words = row
    w = a_norm.split()
    toks = ['a' + t for t in w]
    toks += ['p' + w[i] + '_' + w[i + 1] for i in range(len(w) - 1)]
    if postal:
        toks.append('z' + postal)
    toks += ['k' + skeleton(t) for t in words.split() if len(t) >= 4]
    return toks


_HV = dict(n_features=N_FEATURES, alternate_sign=False, norm=None, binary=True,
           lowercase=False)


def _vec_chunk(args):
    kind, rows = args
    fn = name_tokens if kind == 'n' else addr_tokens
    hv = HashingVectorizer(analyzer=fn, **_HV)
    return hv.transform(rows).astype(np.float32)


def vectorize(df, kind, pool):
    if kind == 'n':
        rows = list(zip(df.n_core, df.n_skel, df.n_compact, df.n_domain, df.n_alt, df.n_norm))
    else:
        rows = list(zip(df.a_norm, df.a_postal, df.a_words))
    step = 50_000
    parts = pool.map(_vec_chunk, [(kind, rows[i:i + step]) for i in range(0, len(rows), step)])
    return sp.vstack(parts, format='csr') if parts else sp.csr_matrix((0, N_FEATURES), dtype=np.float32)


def load(split, src, cols=None, country=None, rows=None):
    """Read one source (optionally one country or a set of row numbers),
    keeping each record's parquet row index in column `row`."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(WORK_DIR / f'{split}_{src}.parquet', columns=cols)
    idx = np.arange(t.num_rows, dtype=np.int32)
    if country is not None:
        mask = pc.equal(pc.cast(t['country'], pa.string()), country)
        idx = np.flatnonzero(mask.to_numpy(zero_copy_only=False)).astype(np.int32)
        t = t.take(idx)
    if rows is not None:
        idx = np.asarray(rows, dtype=np.int32)
        t = t.take(idx)
    df = t.to_pandas()
    df['row'] = idx
    return df


def countries(split):
    import pyarrow.parquet as pq
    c = pq.read_table(WORK_DIR / f'{split}_source1.parquet', columns=['country']).column(0)
    return sorted(set(c.to_pylist()))


def weigh(X, idf, keep):
    """Apply IDF, drop pruned columns, L2-normalise rows."""
    X = X @ sp.diags(idf * keep)
    X.eliminate_zeros()
    return l2norm(X, copy=False)


def _topk(Q, ST, k):
    """Top-k columns per query row, dropping those far below the row's best."""
    R = sp_matmul_topn(Q, ST, top_n=k, threshold=MIN_SCORE, sort=True, n_threads=N_JOBS).tocoo()
    best = np.zeros(Q.shape[0], np.float32)
    np.maximum.at(best, R.row, R.data)
    keep = R.data >= REL_MIN * best[R.row]
    return sp.coo_matrix((R.data[keep], (R.row[keep], R.col[keep])), shape=R.shape)


def rowdot(A, ai, B, bi, step=2_000_000):
    """Cosine of row pairs (A[ai[k]], B[bi[k]]) for L2-normalised csr A, B."""
    out = np.empty(len(ai), np.float32)
    for s in range(0, len(ai), step):
        out[s:s + step] = np.asarray(
            A[ai[s:s + step]].multiply(B[bi[s:s + step]]).sum(axis=1)).ravel()
    return out


def block_split(split):
    cols = ['entity_id', 'country', 'n_core', 'n_skel', 'n_compact', 'n_domain', 'n_alt',
            'n_norm', 'a_norm', 'a_postal', 'a_words']
    out, parts = [], []
    with Pool(N_JOBS) as pool:
        for country in countries(split):
            t0 = time.time()
            S = load(split, 'source1', cols, country)
            s1_idx = S.row.values
            Sn, Sa = vectorize(S, 'n', pool), vectorize(S, 'a', pool)
            del S
            n = len(s1_idx)
            mats = {}
            for kind, X in (('n', Sn), ('a', Sa)):
                df = np.asarray((X > 0).sum(axis=0)).ravel()
                idf = np.log((n + 1) / (df + 1)).astype(np.float32) + 1.0
                keep = ((df > 0) & (df <= MAX_DF)).astype(np.float32)
                mats[kind] = (idf, keep)
            Sn, Sa = weigh(Sn, *mats['n']), weigh(Sa, *mats['a'])
            SXT = (sp.hstack([Sn, Sa], format='csr') * np.float32(1 / np.sqrt(2))).T.tocsr()
            SaT = Sa.T.tocsr()
            for src_i, src in ((2, 'source2'), (3, 'source3')):
                Q = load(split, src, cols, country)
                for c0 in range(0, len(Q), QUERY_CHUNK):
                    Qc = Q.iloc[c0:c0 + QUERY_CHUNK]
                    Qn = weigh(vectorize(Qc, 'n', pool), *mats['n'])
                    Qa = weigh(vectorize(Qc, 'a', pool), *mats['a'])
                    QX = sp.hstack([Qn, Qa], format='csr') * np.float32(1 / np.sqrt(2))
                    R = _topk(QX, SXT, TOP_K)
                    Ra = _topk(Qa, SaT, ADDR_K)
                    # union of both channels; combined score recomputed for address-channel pairs
                    R = sp.coo_matrix((np.ones(len(R.data) + len(Ra.data), np.float32),
                                       (np.r_[R.row, Ra.row], np.r_[R.col, Ra.col])), shape=R.shape).tocsr()
                    R.sum_duplicates()
                    R = R.tocoo()
                    print(f'  {country} {src} {c0 + len(Qc)}/{len(Q)} ({time.time() - t0:.0f}s)', flush=True)
                    ncos = rowdot(Qn, R.row, Sn, R.col)
                    acos = rowdot(Qa, R.row, Sa, R.col)
                    out.append(pd.DataFrame({
                        'src': np.int8(src_i),
                        'q_row': Qc.row.values[R.row],
                        's1_row': s1_idx[R.col],
                        'score': (0.5 * (ncos + acos)).astype(np.float32),
                        'ncos': ncos,
                        'acos': acos,
                    }))
                del Q
            # flush this country's pairs to disk so memory does not accumulate
            part = pd.concat(out, ignore_index=True)
            out.clear()
            part.to_parquet(WORK_DIR / f'{split}_cand_raw_{country}.parquet', index=False)
            parts.append(WORK_DIR / f'{split}_cand_raw_{country}.parquet')
            del part
            print(f'[{split}] {country}: S1={n} done in {time.time() - t0:.0f}s', flush=True)
    import pyarrow.parquet as pq
    tables = [pq.read_table(p) for p in parts]
    import pyarrow as pa
    pq.write_table(pa.concat_tables(tables), WORK_DIR / f'{split}_cand_raw.parquet')
    for p in parts:
        p.unlink()
    print(f'[{split}] candidate pairs: {sum(t.num_rows for t in tables)}', flush=True)


if __name__ == '__main__':
    block_split(sys.argv[1])
