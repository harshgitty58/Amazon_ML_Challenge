"""Stage-2 features: competition among candidates measured with stage-1
probabilities instead of the blocking score.

A record in Source 2/3 has at most one true parent, and a Source 1 entity that
already owns several confident matches is a likely parent for one more.  Both
views need the stage-1 probability of every candidate pair, so these features
are computed over the whole candidate set of a split.
"""
import numpy as np
import pandas as pd

from features import _segment_rank

S1_PARAMS = dict(objective='binary', learning_rate=0.15, num_leaves=127, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                 max_bin=127, verbose=-1)
S1_ROUNDS = 800
S2_PARAMS = dict(objective='binary', learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
                 feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
                 max_bin=127, verbose=-1)
S2_ROUNDS = 1000
WIN_P = 0.5


def prob_context(src, q_row, s1_row, p):
    """Features of each pair given stage-1 probabilities p of all pairs."""
    f = {}
    p = p.astype(np.float32)
    p64 = p.astype(np.float64)
    qkey = src.astype(np.int64) * 1_000_000_000 + q_row
    r, top, sec, _ = _segment_rank(qkey, p64)
    f['p1'] = p
    f['p_rank_q'] = np.minimum(r, 127).astype(np.int8)
    f['p_other_q'] = np.where(r == 1, sec, top).astype(np.float32)   # best competing parent
    f['p_sum_q'] = pd.Series(p64).groupby(qkey).transform('sum').values.astype(np.float32)
    r1, top1, _, _ = _segment_rank(s1_row.astype(np.int64), p64)
    f['p_rank_s1'] = np.minimum(r1, 1000).astype(np.int16)
    f['p_top_s1'] = top1.astype(np.float32)
    f['p_sum_s1'] = (pd.Series(p64).groupby(s1_row).transform('sum').values - p64).astype(np.float32)
    # other queries for which this S1 is the confident first choice
    win = ((r == 1) & (p >= WIN_P)).astype(np.int32)
    f['n_win_s1'] = (pd.Series(win).groupby(s1_row).transform('sum').values - win).astype(np.int16)
    return pd.DataFrame(f)


# --------------------------------------------------------------------------
# Sibling features.  Copies of one business inside a source share an
# entity-level variant (same address rewrite, same typo, same spacing), so a
# weak record often looks much more like a confidently matched sibling than
# like the Source 1 record itself.
# --------------------------------------------------------------------------
SIB_P = 0.5
SIB_CHUNKS = 16
SIB_COLS = ['n_sib', 'sib_rn', 'sib_ra', 'sib_nn', 'sib_an', 'sib_tnn', 'sib_tan', 'sib_exn', 'sib_exa',
            'sib_fn', 'ss_ra', 'ss_rn', 'sib_p_ra', 'sib_nj']


def write_raw(split):
    """Lower-cased raw name/address (original spacing kept) per S2/S3 record."""
    from config import WORK_DIR, TRAIN_DIR, TEST_DIR
    d = TRAIN_DIR if split == 'train' else TEST_DIR
    for s in (2, 3):
        out = WORK_DIR / f'{split}_source{s}_raw.parquet'
        if out.exists():
            continue
        df = pd.read_csv(d / f'{split}_source{s}.tsv', sep='\t', dtype=str, keep_default_na=False,
                         usecols=['business_name', 'business_address'])
        pd.DataFrame({'rn': df.business_name.str.lower(), 'ra': df.business_address.str.lower()}) \
            .to_parquet(out, index=False)


class QueryStrings:
    """Row-indexed access to the strings of Source 2/3 records."""

    def __init__(self, split):
        import pyarrow.parquet as pq
        from config import WORK_DIR
        write_raw(split)
        self.t = {}
        for s in (2, 3):
            raw = pq.read_table(WORK_DIR / f'{split}_source{s}_raw.parquet')
            nz = pq.read_table(WORK_DIR / f'{split}_source{s}.parquet',
                               columns=['n_norm', 'a_norm', 'a_first_num', 'a_nums'])
            self.t[s] = {'rn': raw.column('rn'), 'ra': raw.column('ra'), 'nn': nz.column('n_norm'),
                         'an': nz.column('a_norm'), 'fn': nz.column('a_first_num'), 'nm': nz.column('a_nums')}

    def get(self, keys):
        """keys: sorted unique src*1e9+row -> DataFrame of strings, one row per key."""
        parts = []
        for s in (2, 3):
            rows = (keys[keys // 1_000_000_000 == s] % 1_000_000_000).astype(np.int64)
            parts.append(pd.DataFrame({c: col.take(rows).to_numpy(zero_copy_only=False)
                                       for c, col in self.t[s].items()}))
        return pd.concat(parts, ignore_index=True)


def _jaccard100(a, b):
    """Jaccard of whitespace token sets x100; -1 when the first side is empty."""
    out = np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x:
            sx, sy = set(x.split()), set(y.split())
            out[i] = 100 * len(sx & sy) / len(sx | sy)
    return out


def sibling_features(src, q_row, s1_row, p, strings):
    """For each pair (X, q): how q compares with the other records confidently
    assigned to X (their best candidate is X with probability >= SIB_P)."""
    from rapidfuzz import fuzz, process
    from config import N_JOBS
    n = len(p)
    qkey = src.astype(np.int64) * 1_000_000_000 + q_row
    r, _, _, _ = _segment_rank(qkey, p.astype(np.float64))
    conf = (r == 1) & (p >= SIB_P)
    # similarities are 0-100 ratios: int8 keeps the full test set in memory
    out = {c: np.full(n, -2, np.int8) for c in SIB_COLS}
    out['n_sib'] = np.zeros(n, np.int16)
    s1 = s1_row.astype(np.int64)
    for k in range(SIB_CHUNKS):
        rid = np.flatnonzero(s1 % SIB_CHUNKS == k)
        cm = rid[conf[rid]]
        E = pd.DataFrame({'rid': rid, 's1': s1[rid], 'qk': qkey[rid]}).merge(
            pd.DataFrame({'s1': s1[cm], 'sk': qkey[cm], 'sp': p[cm]}), on='s1')
        E = E[E.qk.values != E.sk.values]
        if not len(E):
            continue
        keys = np.unique(np.r_[E.qk.values, E.sk.values])
        S = strings.get(keys)
        qi = np.searchsorted(keys, E.qk.values)
        si = np.searchsorted(keys, E.sk.values)
        F = {'rid': E.rid.values}
        for c in ('rn', 'ra', 'nn', 'an'):
            a = S[c].values
            F['r_' + c] = process.cpdist(a[qi].tolist(), a[si].tolist(), scorer=fuzz.ratio,
                                         workers=N_JOBS, dtype=np.float32)
            if c in ('nn', 'an'):
                F['t_' + c] = process.cpdist(a[qi].tolist(), a[si].tolist(), scorer=fuzz.token_set_ratio,
                                             workers=N_JOBS, dtype=np.float32)
        F = pd.DataFrame(F)
        empty = S.an.values[qi] == ''
        F.loc[empty, ['r_ra', 'r_an', 't_an']] = -1
        F['exn'] = (S.rn.values[qi] == S.rn.values[si]).astype(np.int8)
        F['exa'] = ((S.ra.values[qi] == S.ra.values[si]) & ~empty).astype(np.int8)
        fn = S.fn.values
        F['fn'] = np.where(fn[qi] == '', -1, fn[qi] == fn[si]).astype(np.int8)
        F['p_ra'] = np.where(empty, -1, F.r_ra.values * E.sp.values).astype(np.float32)
        F['nj'] = _jaccard100(S.nm.values[qi], S.nm.values[si])
        same = (E.qk.values // 1_000_000_000) == (E.sk.values // 1_000_000_000)
        g = F.groupby('rid')
        agg = g.agg(n_sib=('fn', 'size'), sib_rn=('r_rn', 'max'), sib_ra=('r_ra', 'max'),
                    sib_nn=('r_nn', 'max'), sib_an=('r_an', 'max'), sib_tnn=('t_nn', 'max'),
                    sib_tan=('t_an', 'max'), sib_exn=('exn', 'max'), sib_exa=('exa', 'max'),
                    sib_fn=('fn', 'max'), sib_p_ra=('p_ra', 'max'), sib_nj=('nj', 'max'))
        ss = F[same].groupby('rid').agg(ss_ra=('r_ra', 'max'), ss_rn=('r_rn', 'max'))
        for c in agg.columns:
            out[c][agg.index.values] = np.minimum(agg[c].values, 32000 if c == 'n_sib' else 100)
        for c in ss.columns:
            out[c][ss.index.values] = ss[c].values
    return pd.DataFrame(out)

