# -*- coding: utf-8 -*-
"""H2-L construction verification: three fully-compatible site-set
constructions + machine verification of the lemma.

Constructions (W >= 3, site modes need p >= 2):
  W even:    S = {(i,j): i,j odd, 3<=i<=L-1, 1<=j<=W-1}
             density -> 1/4
  W odd >=5: S = {(i,j): i,j odd, 3<=i<=L-1, 1<=j<=W-1,
                  (i+j)%W in {0,2,..,W-3}}
             density -> 1/8 (smallest W=5: 2/25)
  W = 3:     S = {(i,1): i = 0 (mod 3), 3<=i<=L-1}
             density -> 1/9

Verification items:
  V1 per-stripe interval conditions (U disjoint / V disjoint / U-V nested)
     => fully compatible (sufficient condition, lemma L1)
  V2 direct pairwise walk verification (all pairs)
  V3 small-window all-subset admissibility + pairwise-distinct fields
     (injectivity)
  V4 end to end: omega_formula(L,W) >= 2^|S| (small-F cases)
  V5 negative control: the bare odd-W checkerboard must contain a dead pair
     (including d+e = +/-1 offset pairs)
"""
import os, sys, json, random, time
from collections import defaultdict
from itertools import product

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths_common as PC  # noqa: E402
from h2l_strip import (check_adm, field_of, site_intervals, pair_ok,
                       omega_formula, omega_brute)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(str(PC.DATA_ROOT), 'lower_bounds')


def construct_S(L, W):
    if W % 2 == 0:
        R = None
    elif W >= 5:
        R = set(range(0, W - 1, 2))          # {0,2,...,W-3}
    else:                                    # W = 3
        R = 'col3'
    S = []
    if R == 'col3':
        for i in range(3, L):
            if i % 3 == 0:
                S.append((i, 1))
        return S
    for i in range(3, L):
        if i % 2 == 0:
            continue
        for j in range(1, W):
            if j % 2 == 0:
                continue
            if R is None or (i + j) % W in R:
                S.append((i, j))
    return S


def stripe_conditions_ok(L, W, S):
    """Per stripe: U pairwise disjoint, V pairwise disjoint, U-V nested.
    Columns = arcs mod W; rows/diagonals = line segments."""
    def arc(fam, a, l):
        if fam == 'col':
            return frozenset((a + k) % W for k in range(l))
        return frozenset(range(a, a + l))
    per = defaultdict(list)
    for t in S:
        for (fam, sid, o, a, l) in site_intervals(*t, W):
            per[(fam, sid)].append((o, arc(fam, a, l)))
    for key, lst in per.items():
        for m in range(len(lst)):
            for n in range(m + 1, len(lst)):
                (o1, s1), (o2, s2) = lst[m], lst[n]
                if o1 == o2:
                    if s1 & s2:
                        return False, key, (lst[m], lst[n])
                else:
                    if not (s1 <= s2 or s2 <= s1):
                        return False, key, (lst[m], lst[n])
    return True, None, None


def all_subsets_full(L, W, S):
    """All subsets: admissible + pairwise-distinct fields.
    Returns (ok, n_distinct, bad_T)."""
    fields = {}
    for mask in range(1, 1 << len(S)):
        T = [S[k] for k in range(len(S)) if (mask >> k) & 1]
        f = field_of(T, W)
        key = tuple(sorted(f.items()))
        if key in fields:
            return False, None, ('dup', T)
        fields[key] = True
        if not check_adm(L, W, f):
            return False, None, ('inadm', T)
    return True, len(fields), None


def omega_strip_dfs(L, W):
    """Incremental pruned DFS for the open-strip Omega (row-major assignment
    of a in {-1,0,1}; the three family (cur,lo,hi) states maintained
    online).
    Open-strip convention: columns = y closed loops (net 0 checked at the
    end, multiply 2-range); rows/diagonals = x open chains (no net-0
    condition, multiply 2-range at the end). Same formula as
    omega_formula."""
    col_c = [0] * L; col_lo = [0] * L; col_hi = [0] * L
    row_c = [0] * W; row_lo = [0] * W; row_hi = [0] * W
    dg_c = [0] * W; dg_lo = [0] * W; dg_hi = [0] * W
    total = 0
    F = L * W

    def rec(pos, weight):
        nonlocal total
        if pos == F:
            total += weight
            return
        i, j = divmod(pos, W)
        sg = (i + j) % W
        last_j = (j == W - 1)
        last_i = (i == L - 1)
        for v in (0, 1, -1):
            c = col_c[i] + v; lo = col_lo[i]; hi = col_hi[i]
            if c < lo: lo = c
            if c > hi: hi = c
            if hi - lo > 1: continue
            if last_j and c != 0: continue
            r = row_c[j] + v; rlo = row_lo[j]; rhi = row_hi[j]
            if r < rlo: rlo = r
            if r > rhi: rhi = r
            if rhi - rlo > 1: continue
            f = dg_c[sg] - v; flo = dg_lo[sg]; fhi = dg_hi[sg]
            if f < flo: flo = f
            if f > fhi: fhi = f
            if fhi - flo > 1: continue
            w = weight
            if last_j:
                w *= 2 - (hi - lo)
            if last_i:
                w *= (2 - (rhi - rlo)) * (2 - (fhi - flo))
            s = (col_c[i], col_lo[i], col_hi[i], row_c[j], row_lo[j], row_hi[j],
                 dg_c[sg], dg_lo[sg], dg_hi[sg])
            col_c[i], col_lo[i], col_hi[i] = c, lo, hi
            row_c[j], row_lo[j], row_hi[j] = r, rlo, rhi
            dg_c[sg], dg_lo[sg], dg_hi[sg] = f, flo, fhi
            rec(pos + 1, w)
            (col_c[i], col_lo[i], col_hi[i], row_c[j], row_lo[j], row_hi[j],
             dg_c[sg], dg_lo[sg], dg_hi[sg]) = s
    rec(0, 1)
    return total


def main():
    rep = {}
    print('== V1+V2: per-stripe conditions of the three constructions + all-pairs direct verification (W=3..24, L=4W) ==')
    rep['constructions'] = {}
    all_ok = True
    for W in range(3, 25):
        L = 4 * W
        S = construct_S(L, W)
        cond, badkey, badpair = stripe_conditions_ok(L, W, S)
        pair_ok_all = True
        memo = {}
        for m in range(len(S)):
            for n in range(m + 1, len(S)):
                if not pair_ok(L, W, S[m], S[n], memo):
                    pair_ok_all = False
        dens = len(S) / (L * W)
        ok = cond and pair_ok_all
        all_ok &= ok
        rep['constructions']['%d' % W] = {
            'L': L, 'W': W, 'n': len(S), 'density': dens,
            'stripe_conditions': cond, 'all_pairs': pair_ok_all}
        print('  W=%2d L=%3d: |S|=%4d dens=%.4f  stripe cond:%s all pairs:%s' % (
            W, L, len(S), dens, 'PASS' if cond else 'FAIL', 'PASS' if pair_ok_all else 'FAIL'))
        if not cond:
            print('    bad stripe:', badkey, badpair)
    print('  V1+V2:', 'ALL PASS' if all_ok else 'HAS FAIL')

    print('== V2b: large-L density asymptotics (W=5..11, L=200) ==')
    for W in range(5, 12, 2):
        L = 200
        S = construct_S(L, W)
        print('  W=%2d L=%3d: |S|=%4d dens=%.4f (theory (W-1)^2/(8W^2)=%.4f)' % (
            W, L, len(S), len(S) / (L * W), (W - 1) ** 2 / (8.0 * W ** 2)))

    print('== V3: small-window all-subset admissibility + injectivity ==')
    rep['subsets'] = {}
    v3_ok = True
    for (W, L) in [(3, 9), (4, 8), (5, 15), (6, 10), (7, 21), (8, 12), (9, 27)]:
        S = construct_S(L, W)
        # truncate the window to <= 14 sites (all 2^14 subsets verified)
        Ssub = S[:14]
        ok, nd, bad = all_subsets_full(L, W, Ssub)
        v3_ok &= ok
        print('  W=%d L=%2d: |S|=%2d all subsets: %s (2^%d fields pairwise distinct and admissible) %s' % (
            W, L, len(Ssub), 'PASS' if ok else 'FAIL', len(Ssub), bad or ''))
        rep['subsets']['%d_%d' % (W, L)] = {'n': len(Ssub), 'ok': ok}
    print('  V3:', 'ALL PASS' if v3_ok else 'HAS FAIL')

    print('== V4: end-to-end Omega_strip >= 2^|S| ==')
    v4_ok = True
    print('  V4a: omega_strip_dfs vs omega_formula cross-check:')
    for (L, W) in [(2, 2), (3, 2), (2, 3), (3, 3), (2, 4), (4, 2), (4, 3)]:
        a = omega_formula(L, W)
        b = omega_strip_dfs(L, W)
        ok = (a == b)
        v4_ok &= ok
        print('    (%d,%d): formula=%d dfs=%d %s' % (L, W, a, b, 'PASS' if ok else 'FAIL'))
    print('  V4b: construction lower bound (larger W=3 cases):')
    for (L, W) in [(5, 3), (6, 3), (8, 3), (12, 3), (6, 4), (8, 5)]:
        S = construct_S(L, W)
        om = omega_strip_dfs(L, W)
        need = 2 ** len(S)
        ok = om >= need
        v4_ok &= ok
        print('    (%d,%d): Omega_strip=%d >= 2^%d=%d %s' % (
            L, W, om, len(S), need, 'PASS' if ok else 'FAIL'))
    all_ok &= v4_ok
    print('  V4:', 'ALL PASS' if v4_ok else 'HAS FAIL')

    print('== V5: negative control (the bare odd-W checkerboard must fail) ==')
    for W in (5, 7, 9):
        L = 4 * W
        bad_found = None
        S = [(i, j) for i in range(3, L, 2) for j in range(1, W, 2)]
        Sm = set(S)
        for (i, j) in S:
            for (di, dj) in [(W - 1, 0), (W + 1, 0)]:
                t2 = (i + di, j + dj)
                if t2 in Sm:
                    if not pair_ok(L, W, (i, j), t2):
                        bad_found = ((i, j), t2, (di, dj))
                        break
            if bad_found:
                break
        print('  W=%d: dead pair in the bare checkerboard: %s' % (
            W, bad_found if bad_found else 'none found (anomaly!)'))
        all_ok &= (bad_found is not None)

    with open(os.path.join(OUT_DIR, 'h2l_construct_result.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)
    print('saved h2l_construct_result.json')
    print('OVERALL:', 'ALL PASS' if all_ok and v3_ok else 'HAS FAIL')


if __name__ == '__main__':
    main()
