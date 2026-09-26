"""Candidate expansion through siblings (runs after stage 1).

Copies of one business inside a source share an entity-level variant, so a
record that blocking missed often carries exactly the same normalised address,
core name, name skeleton or house number + street as a sibling that stage 1
matched confidently.  For every record without a confident match we look up
those keys among confidently matched records of the same country and add the
siblings' Source 1 entity as a new candidate.  Keys shared by more than
MAX_X different entities are too generic and are skipped.

The new pairs get the same blocking cosines and pair features as ordinary
candidates, so both stages score them like any other pair.
"""
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from block import load, countries, vectorize, weigh, rowdot, MAX_DF
from config import N_JOBS
from features import _segment_rank, REC_COLS, pair_features, context_features
from normalize import skeleton

CONF_P = 0.5
MAX_X = 3
CHUNK = 1_500_000
KEY_COLS = ['country', 'n_core', 'n_skel', 'a_norm', 'a_words', 'a_first_num']
BLOCK_COLS = ['n_core', 'n_skel', 'n_compact', 'n_domain', 'n_alt', 'n_norm', 'a_norm', 'a_postal', 'a_words']
QK = 1_000_000_000


def _street(words):
    w = [x for x in words.split() if len(x) >= 4]
    return skeleton(w[0]) if w else ''


def _keys(df):
    """Hashed lookup keys (0 = no key) for each record, country included."""
    c = df.country.astype(str).values
    out = {}
    raw = {
        'addr': np.where(df.a_norm.str.len().values >= 8, df.a_norm.values, ''),
        'name': np.where(df.n_core.str.len().values >= 4, df.n_core.values, ''),
        'skel': np.where(df.n_skel.str.len().values >= 4, df.n_skel.values, ''),
        'num': np.array([f'{n}|{_street(w)}' if n and _street(w) else ''
                         for n, w in zip(df.a_first_num.values, df.a_words.values)], dtype=object),
    }
    cs = pd.Series(c, dtype=object)
    for k, v in raw.items():
        v = pd.Series(v, dtype=object)
        h = pd.util.hash_pandas_object(cs + f'|{k}|' + v, index=False).values
        out[k] = np.where(v.values == '', 0, h).astype(np.uint64)
    return out


def find_new(split, pairs):
    """pairs: DataFrame(src, q_row, s1_row, p1).  Returns new (src, q_row, s1_row, n_keys)."""
    qkey = pairs.src.values.astype(np.int64) * QK + pairs.q_row.values
    r, _, _, _ = _segment_rank(qkey, pairs.p1.values.astype(np.float64))
    conf = (r == 1) & (pairs.p1.values >= CONF_P)
    conf_q = pd.Series(pairs.s1_row.values[conf], index=qkey[conf])   # confident query -> its S1
    keys, qk_all = {}, []
    for s in (2, 3):
        df = load(split, f'source{s}', KEY_COLS)
        for k, v in _keys(df).items():
            keys.setdefault(k, []).append(v)
        qk_all.append(s * QK + df.row.values.astype(np.int64))
        del df
    qk_all = np.concatenate(qk_all)
    is_conf = np.isin(qk_all, conf_q.index.values)
    parent = np.full(len(qk_all), -1, np.int64)
    parent[is_conf] = conf_q.reindex(qk_all[is_conf]).values
    found = []
    for k in keys:
        h = np.concatenate(keys[k])
        ref = pd.DataFrame({'h': h[is_conf], 'x': parent[is_conf]})
        ref = ref[ref.h != 0].drop_duplicates()
        nx = ref.groupby('h').x.transform('size')
        ref = ref[nx.values <= MAX_X]
        weak = (~is_conf) & (h != 0)
        hit = pd.DataFrame({'h': h[weak], 'qk': qk_all[weak]}).merge(ref, on='h')
        found.append(hit[['qk', 'x']])
    new = pd.concat(found, ignore_index=True)
    new = new.groupby(['qk', 'x']).size().rename('n_keys').reset_index()
    # drop pairs already in the candidate set
    old = pd.DataFrame({'qk': qkey, 'x': pairs.s1_row.values.astype(np.int64), 'o': np.int8(1)})
    new = new.merge(old, on=['qk', 'x'], how='left')
    new = new[new.o.isna()]
    return pd.DataFrame({'src': (new.qk.values // QK).astype(np.int8),
                         'q_row': (new.qk.values % QK).astype(np.int32),
                         's1_row': new.x.values.astype(np.int32),
                         'n_keys': new.n_keys.values.astype(np.int8)})


def _mats(X, n):
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log((n + 1) / (df + 1)).astype(np.float32) + 1.0
    return idf, ((df > 0) & (df <= MAX_DF)).astype(np.float32)


def blocking_scores(split, new):
    """ncos / acos / score for new pairs, with the same vectors as block.py."""
    ncos = np.zeros(len(new), np.float32)
    acos = np.zeros(len(new), np.float32)
    s1c = load(split, 'source1', ['country']).country.astype(str).values
    nc = s1c[new.s1_row.values]
    with Pool(N_JOBS) as pool:
        for country in countries(split):
            m = np.flatnonzero(nc == country)
            if not len(m):
                continue
            S = load(split, 'source1', BLOCK_COLS + ['country'], country)
            Sn, Sa = vectorize(S, 'n', pool), vectorize(S, 'a', pool)
            mats = {'n': _mats(Sn, len(S)), 'a': _mats(Sa, len(S))}
            Sn, Sa = weigh(Sn, *mats['n']), weigh(Sa, *mats['a'])
            s1_local = np.searchsorted(S.row.values, new.s1_row.values[m])
            del S
            for s in (2, 3):
                ms = m[new.src.values[m] == s]
                if not len(ms):
                    continue
                rows = np.unique(new.q_row.values[ms])
                Q = load(split, f'source{s}', BLOCK_COLS, rows=rows)
                Qn = weigh(vectorize(Q, 'n', pool), *mats['n'])
                Qa = weigh(vectorize(Q, 'a', pool), *mats['a'])
                qi = np.searchsorted(rows, new.q_row.values[ms])
                si = s1_local[np.searchsorted(m, ms)]
                ncos[ms] = rowdot(Qn, qi, Sn, si)
                acos[ms] = rowdot(Qa, qi, Sa, si)
                del Q, Qn, Qa
    out = new.copy()
    out['score'] = (0.5 * (ncos + acos)).astype(np.float32)
    out['ncos'], out['acos'] = ncos, acos
    return out


def featurize_new(split, new, old_ctx):
    """Context + pair features for the new pairs.  Context (rank, margin, ...)
    is computed over old + new candidates together; old pairs keep their
    original values.  old_ctx: DataFrame(src, q_row, s1_row, score, ncos, acos)."""
    t = time.time()
    n_old = len(old_ctx)
    u = pd.concat([old_ctx, new[['src', 'q_row', 's1_row', 'score', 'ncos', 'acos']]], ignore_index=True)
    u = context_features(u)
    c = pd.concat([u.iloc[n_old:].reset_index(drop=True), new[['n_keys']].reset_index(drop=True)], axis=1)
    del u
    parts = []
    for s in (2, 3):
        part = c[c.src.values == s]
        if not len(part):
            continue
        S1 = load(split, 'source1', REC_COLS, rows=np.unique(part.s1_row.values)).set_index('row')
        Q = load(split, f'source{s}', REC_COLS, rows=np.unique(part.q_row.values)).set_index('row')
        for a in range(0, len(part), CHUNK):
            p = part.iloc[a:a + CHUNK].reset_index(drop=True)
            f = pair_features(S1.loc[p.s1_row.values].reset_index(drop=True),
                              Q.loc[p.q_row.values].reset_index(drop=True))
            parts.append(pd.concat([p, f], axis=1))
        del S1, Q
    print(f'  expansion features for {len(c)} pairs ({time.time() - t:.0f}s)', flush=True)
    return pd.concat(parts, ignore_index=True)


def expand(split, pairs, old_ctx):
    t = time.time()
    new = find_new(split, pairs)
    print(f'  expansion: {len(new)} new pairs ({time.time() - t:.0f}s)', flush=True)
    if not len(new):
        return None
    new = blocking_scores(split, new)
    return featurize_new(split, new, old_ctx)
