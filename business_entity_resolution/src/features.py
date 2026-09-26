"""Pairwise features for (Source 1, Source 2/3) candidate pairs.

All features are country-agnostic similarities (no country one-hot), so the
model transfers to country labels unseen in training (France).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from config import N_JOBS

REC_COLS = ['n_norm', 'n_core', 'n_compact', 'n_skel', 'n_domain', 'n_alt', 'n_nonlatin',
            'a_norm', 'a_words', 'a_nums', 'a_first_num', 'a_postal', 'a_state', 'a_empty']


def _cp(scorer, a, b, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=N_JOBS, dtype=np.float32, **kw)


def _set_overlap(a, b):
    """|A∩B| / min(|A|,|B|) and |A∩B| over whitespace tokens (python loop)."""
    ov = np.zeros(len(a), np.float32)
    inter = np.zeros(len(a), np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if x and y:
            sx, sy = set(x.split()), set(y.split())
            k = len(sx & sy)
            inter[i] = k
            ov[i] = k / min(len(sx), len(sy))
    return ov, inter


def _num_sets(a, b):
    """Address number sets: R-only count, L-only count, Jaccard (-1 when R has none).
    Near-miss distractors share the building but differ in one unit number."""
    r_extra = np.full(len(a), -1, np.float32)
    l_extra = np.full(len(a), -1, np.float32)
    jac = np.full(len(a), -1, np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if y:
            sx, sy = set(x.split()), set(y.split())
            k = len(sx & sy)
            r_extra[i] = len(sy) - k
            l_extra[i] = len(sx) - k
            jac[i] = k / len(sx | sy)
    return r_extra, l_extra, jac


def _num_match(a, b):
    """-1 no numbers on either side, 2 same number set, 1 overlap, 0 disjoint/one-sided."""
    x = {t for t in a.split() if t.isdigit()}
    y = {t for t in b.split() if t.isdigit()}
    if not x and not y:
        return -1
    if x == y:
        return 2
    return 1 if x & y else 0


def pair_features(L, R):
    """L = S1 side, R = S2/S3 side; aligned DataFrames with REC_COLS."""
    f = {}
    ln, rn = L.n_norm.tolist(), R.n_norm.tolist()
    lc, rc = L.n_core.tolist(), R.n_core.tolist()
    f['n_ratio'] = _cp(fuzz.ratio, ln, rn)
    f['n_tset'] = _cp(fuzz.token_set_ratio, ln, rn)
    f['n_tsort'] = _cp(fuzz.token_sort_ratio, ln, rn)
    f['c_ratio'] = _cp(fuzz.ratio, lc, rc)
    f['c_tset'] = _cp(fuzz.token_set_ratio, lc, rc)
    f['c_tsort'] = _cp(fuzz.token_sort_ratio, lc, rc)
    f['c_partial'] = _cp(fuzz.partial_ratio, lc, rc)
    f['c_wratio'] = _cp(fuzz.WRatio, lc, rc)
    lcp, rcp = L.n_compact.tolist(), R.n_compact.tolist()
    f['cp_jw'] = _cp(JaroWinkler.normalized_similarity, lcp, rcp)
    f['cp_lev'] = _cp(Levenshtein.distance, lcp, rcp)
    f['sk_tset'] = _cp(fuzz.token_set_ratio, L.n_skel.tolist(), R.n_skel.tolist())
    f['sk_ratio'] = _cp(fuzz.ratio, L.n_skel.tolist(), R.n_skel.tolist())
    # website-style names: compare domain with the S1 compact name
    rd = R.n_domain.tolist()
    dom = _cp(fuzz.ratio, lcp, rd)
    f['dom_ratio'] = np.where(R.n_domain.str.len().values > 0, dom, -1).astype(np.float32)
    dom_p = _cp(fuzz.partial_ratio, lcp, rd)
    f['dom_partial'] = np.where(R.n_domain.str.len().values > 0, dom_p, -1).astype(np.float32)
    # DBA / AKA alternative name
    ra = R.n_alt.tolist()
    alt = _cp(fuzz.token_set_ratio, lc, ra)
    f['alt_tset'] = np.where(R.n_alt.str.len().values > 0, alt, -1).astype(np.float32)
    ov, inter = _set_overlap(lc, rc)
    f['c_overlap'], f['c_inter'] = ov, inter
    f['l_ntok'] = L.n_core.str.count(' ').values.astype(np.float32) + 1
    f['r_ntok'] = R.n_core.str.count(' ').values.astype(np.float32) + 1
    f['r_nonlatin'] = R.n_nonlatin.values.astype(np.int8)
    # first-token agreement (brand word usually first)
    lf = L.n_core.str.split(' ', n=1).str[0].tolist()
    rf = R.n_core.str.split(' ', n=1).str[0].tolist()
    f['first_tok_ratio'] = _cp(fuzz.ratio, lf, rf)
    # numbers inside the name ("Local No 402", "Fund IV" -> digits only); -1 = neither side has any
    f['name_num'] = np.array([_num_match(a, b) for a, b in zip(ln, rn)], np.int8)

    # ---------------- address ----------------
    la, ra_ = L.a_norm.tolist(), R.a_norm.tolist()
    r_empty = R.a_empty.values.astype(bool)
    f['r_addr_empty'] = r_empty.astype(np.int8)

    def masked(x):
        return np.where(r_empty, -1, x).astype(np.float32)

    f['a_tset'] = masked(_cp(fuzz.token_set_ratio, la, ra_))
    f['a_tsort'] = masked(_cp(fuzz.token_sort_ratio, la, ra_))
    f['a_ratio'] = masked(_cp(fuzz.ratio, la, ra_))
    f['a_partial'] = masked(_cp(fuzz.partial_token_set_ratio, la, ra_))
    lw, rw = L.a_words.tolist(), R.a_words.tolist()
    f['aw_tset'] = masked(_cp(fuzz.token_set_ratio, lw, rw))
    wov, winter = _set_overlap(lw, rw)
    f['aw_overlap'], f['aw_inter'] = masked(wov), masked(winter)
    lnum, rnum = L.a_nums.tolist(), R.a_nums.tolist()
    nov, ninter = _set_overlap(lnum, rnum)
    has_rnum = R.a_nums.str.len().values > 0
    f['num_overlap'] = np.where(has_rnum, nov, -1).astype(np.float32)
    f['num_inter'] = ninter
    f['num_r_extra'], f['num_l_extra'], f['num_jacc'] = _num_sets(lnum, rnum)
    lfn, rfn = L.a_first_num.values, R.a_first_num.values
    f['first_num_eq'] = np.where(rfn == '', -1, (lfn == rfn)).astype(np.int8)
    f['first_num_lev'] = np.where(rfn == '', -1, _cp(Levenshtein.distance, lfn.tolist(), rfn.tolist())).astype(np.float32)
    # house number contained in the other's number list (e.g. "1056" vs "1056 1060")
    f['first_num_in'] = np.array([(b != '' and b in a.split()) for a, b in zip(lnum, rfn)], np.int8)
    lp, rp = L.a_postal.values, R.a_postal.values
    f['postal_eq'] = np.where((lp == '') | (rp == ''), -1, lp == rp).astype(np.int8)
    ls, rs = L.a_state.values, R.a_state.values
    f['state_eq'] = np.where((ls == '') | (rs == ''), -1, ls == rs).astype(np.int8)
    return pd.DataFrame(f)


def _segment_rank(keys, score):
    """Sort by (key, -score); return order, rank within key, top & second score."""
    order = np.lexsort((-score, keys))
    k = keys[order]
    sc = score[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, len(k)])
    rank = np.arange(len(k)) - first[gid] + 1
    top = sc[first][gid]
    sec_pos = np.minimum(first + 1, len(k) - 1)
    second = np.where(size > 1, sc[sec_pos], 0.0)[gid]
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    return rank[inv], top[inv], second[inv], size[gid][inv]


def context_features(c):
    """Features describing competition among candidates (in place).

    Each S2/S3 record has at most one true parent, so how a candidate ranks
    against the query's other candidates is highly informative; the S1-side
    view (how many queries point at this S1) helps with singletons.
    """
    score = c.score.values.astype(np.float64)
    qkey = c.src.values.astype(np.int64) * 1_000_000_000 + c.q_row.values
    rank, top, second, size = _segment_rank(qkey, score)
    c['rank'] = rank.astype(np.int8)
    c['gap_top'] = (top - score).astype(np.float32)
    c['margin'] = np.where(rank == 1, score - second, score - top).astype(np.float32)
    c['n_cand_q'] = size.astype(np.int16)
    for col in ('ncos', 'acos'):
        if col in c:
            v = c[col].values.astype(np.float64)
            r2, t2, s2, _ = _segment_rank(qkey, v)
            c[col + '_gap'] = (t2 - v).astype(np.float32)
            c[col + '_rank'] = r2.astype(np.int8)
    rank1, top1, second1, size1 = _segment_rank(c.s1_row.values.astype(np.int64), score)
    c['rank_s1'] = np.minimum(rank1, 32000).astype(np.int16)
    c['gap_top_s1'] = (top1 - score).astype(np.float32)
    c['n_cand_s1'] = np.minimum(size1, 32000).astype(np.int16)
    # number of queries for which this S1 is the top candidate
    top_hits = pd.Series((rank == 1).astype(np.int32)).groupby(c.s1_row.values).transform('sum')
    c['n_top1_s1'] = top_hits.values.astype(np.int16)
    return c
