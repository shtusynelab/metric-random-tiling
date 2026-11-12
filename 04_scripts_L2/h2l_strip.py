# -*- coding: utf-8 -*-
"""H2-L lemma session: full-compatibility site sets of the open strip
(x open / y periodic W) -- verification platform + search + patterns.

Coords: face (i,j), i in [0,L) with open x boundary, j in [0,W) y-periodic.
Three stripe families (open-strip version):
  column i:    accumulate +a over j=0..W-1, closed loop (must end at 0),
               weight (2-range)
  row j:       accumulate +a over i=0..L-1, open chain, weight (2-range)
  diagonal sg: visit (i,(sg-i)%W) for i=0..L-1, accumulate -a, open chain,
               weight (2-range)
Admissible a: every column nets 0 and every stripe walk has range <= 1;
Omega_strip = Sum_a prod_s (2-range_s).

G1  open-strip counting by two independent algorithms cross-checked: raw
    edge-variable per-face DFS vs stripe formula (no registered anchor; the
    two independent implementations witness each other)
G2  site mode (third difference, 6 charges) self-consistency + single-mode
    admissibility + interval-characterisation cross-check
G3  offset verdict table (per width) + per-width maximal fully-compatible
    set search + pairwise=>global reduction check
"""
import os, sys, math, json, random, time
from collections import defaultdict
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths_common as PC  # noqa: E402

OUT_DIR = os.path.join(str(PC.DATA_ROOT), 'lower_bounds')

OFF = {0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (0, 1), 4: (-1, 1), 5: (-1, 0)}

# ---------------- G1: open-strip platform ----------------

def build_strip(L, W):
    """slots->edges; y periodic, x open boundary (boundary edges have one
    face). Returns (E, faces, rels).
    Face closure: v3-v0 = v5-v2 = v1-v4 (eliminated as a_h, -a_h, a_h)."""
    slot2edge = {}
    nxt = 0
    for i in range(L):
        for j in range(W):
            for k in range(6):
                if (i, j, k) in slot2edge:
                    continue
                ii = i + OFF[k][0]
                jj = (j + OFF[k][1]) % W
                if 0 <= ii < L:
                    slot2edge[(i, j, k)] = nxt
                    slot2edge[(ii, jj, (k + 3) % 6)] = nxt
                else:
                    slot2edge[(i, j, k)] = nxt
                nxt += 1
    E = nxt
    faces, rels = [], []
    for i in range(L):
        for j in range(W):
            h = i * W + j
            fe = [slot2edge[(i, j, k)] for k in range(6)]
            faces.append(fe)
            rels.append((fe[0], fe[3], 1, h))
            rels.append((fe[1], fe[4], -1, h))
            rels.append((fe[2], fe[5], 1, h))
    return E, faces, rels


def omega_brute(L, W, node_cap=200_000_000):
    """Independent algorithm 1: raw edge variables v in {0,1}^E with
    per-face DFS (both equations checked as soon as a face has all 6 edges
    assigned)."""
    E, faces, rels = build_strip(L, W)
    F = L * W
    val = [-1] * E
    total, nodes = 0, 0

    def rec(fi):
        nonlocal total, nodes
        nodes += 1
        if nodes > node_cap:
            raise RuntimeError('node cap')
        if fi == F:
            total += 1
            return
        fe = faces[fi]
        todo = [e for e in fe if val[e] == -1]
        pre = [(e, val[e]) for e in todo]
        for bits in range(1 << len(todo)):
            for n, e in enumerate(todo):
                val[e] = (bits >> n) & 1
            a0, a1, a2, a3, a4, a5 = (val[e] for e in fe)
            if a3 - a0 == a5 - a2 == a1 - a4:
                rec(fi + 1)
        for e, v in pre:
            val[e] = v

    rec(0)
    return total, nodes


def stripe_components(L, W):
    E, faces, rels = build_strip(L, W)
    adj = defaultdict(list)
    for idx, (e_lo, e_hi, sg, h) in enumerate(rels):
        adj[e_lo].append((e_hi, sg, h, idx))
        adj[e_hi].append((e_lo, -sg, h, idx))
    seen = set()
    comps = []
    for idx0 in range(len(rels)):
        if idx0 in seen:
            continue
        e_lo, e_hi, sg, h = rels[idx0]
        seen.add(idx0)
        sigs = [(sg, h)]
        cur, prev = e_hi, idx0
        closed = False
        while True:
            cand = [x for x in adj[cur] if x[3] != prev]
            if not cand:
                break
            ee, ss, hh, ii = cand[0]
            if len(cand) > 1:
                raise AssertionError('branch')
            if ii == idx0:
                closed = True
                break
            sigs.append((ss, hh))
            seen.add(ii)
            cur, prev = ee, ii
        if not closed:
            # open chain: walk backwards from e_lo to the other end;
            # negate the backward steps and prepend them to the forward part
            cur, prev = e_lo, idx0
            back = []
            while True:
                cand = [x for x in adj[cur] if x[3] != prev]
                if not cand:
                    break
                ee, ss, hh, ii = cand[0]
                if ii == idx0:
                    closed = True
                    break
                back.append((ss, hh))
                seen.add(ii)
                cur, prev = ee, ii
            sigs = [(-ss, hh) for (ss, hh) in reversed(back)] + sigs
        comps.append((closed, sigs))
    return comps


def omega_formula(L, W):
    """Independent algorithm 2: stripe formula Omega = Sum_a prod(2-range);
    closed column loops checked for net 0."""
    comps = stripe_components(L, W)
    n_closed = sum(1 for (c, s) in comps if c)
    assert n_closed == L, ('closed stripe count', n_closed, L)
    assert len(comps) == L + 2 * W, ('total stripe count', len(comps), L + 2 * W)
    F = L * W
    total = 0
    for a in product((0, 1, -1), repeat=F):
        w = 1
        for closed, sigs in comps:
            pos = lo = hi = 0
            for sg, h in sigs:
                pos += sg * a[h]
                if pos < lo:
                    lo = pos
                elif pos > hi:
                    hi = pos
            if closed and pos != 0:
                w = 0
                break
            if hi - lo > 1:
                w = 0
                break
            w *= (2 - (hi - lo))
        total += w
    return total


def g1_gate(cases):
    print('== G1 open-strip two-algorithm cross-check ==')
    out = {}
    for (L, W) in cases:
        t0 = time.time()
        bf, nodes = omega_brute(L, W)
        fm = omega_formula(L, W)
        ok = (bf == fm)
        print('  (%d,%d): brute=%d formula=%d %s [nodes=%d, %.1fs]' % (
            L, W, bf, fm, 'PASS' if ok else 'FAIL', nodes, time.time() - t0))
        out['%dx%d' % (L, W)] = {'brute': bf, 'formula': fm, 'pass': ok}
    return out


# ---------------- G2: site modes and interval characterisation ----------------

MODE_REL = ((0, 0, 1), (0, 1, -1), (-1, 0, -1), (-2, 1, 1), (-1, 2, 1), (-2, 2, -1))


def third_diff_mode(p, q, W):
    """a = Delta_{o0} Delta_{o1} Delta_{o2} delta_(p,q)
    (code convention: +v at x, -v at x-d; y mod W)"""
    g = {(p, q): 1}
    for d in ((1, 0), (1, -1), (0, -1)):
        h = defaultdict(int)
        for (i, j), v in g.items():
            h[(i, j)] += v
            h[(i - d[0], (j - d[1]) % W)] -= v
        g = dict(h)
    return {k: v for k, v in g.items() if v != 0}


def mode_charges(p, q, W):
    return third_diff_mode(p, q, W)


def site_intervals(p, q, W):
    """Sign intervals of the site mode on each stripe:
    (fam, id, orient, start, length).
    orient U = + first then - (prefix raised to 1), V = - first then +
    (prefix lowered to -1). Column stripe position j (mod W); row/diagonal
    position i (open chain, not reduced). Derived programmatically from the
    charge table."""
    mode = mode_charges(p, q, W)
    per = defaultdict(list)
    for (i, j), s in mode.items():
        per[('col', i)].append((j, s))
        per[('row', j)].append((i, s))
        per[('diag', (i + j) % W)].append((i, -s))
    out = []
    for (fam, sid), lst in per.items():
        assert len(lst) == 2, (fam, sid, lst)
        (a, sa), (b, sb) = sorted(lst)
        assert sa == -sb and sa in (1, -1), (fam, lst)
        if sa == 1:
            out.append((fam, sid, 'U', a, b - a))
        else:
            out.append((fam, sid, 'V', a, b - a))
    return out


def check_adm(L, W, afield):
    """Stripe criterion: columns net 0 + every walk range <= 1."""
    for i in set(ii for (ii, _) in afield):
        pos = lo = hi = 0
        for j in range(W):
            pos += afield.get((i, j), 0)
            if pos < lo:
                lo = pos
            if pos > hi:
                hi = pos
        if pos != 0 or hi - lo > 1:
            return False
    for j in set(jj for (_, jj) in afield):
        pos = lo = hi = 0
        for i in range(L):
            pos += afield.get((i, j), 0)
            if pos < lo:
                lo = pos
            if pos > hi:
                hi = pos
        if hi - lo > 1:
            return False
    for sg in set(((i + j) % W) for (i, j) in afield):
        pos = lo = hi = 0
        for i in range(L):
            pos -= afield.get((i, (sg - i) % W), 0)
            if pos < lo:
                lo = pos
            if pos > hi:
                hi = pos
        if hi - lo > 1:
            return False
    return True


def field_of(sites, W):
    d = defaultdict(int)
    for (p, q) in sites:
        for (i, j), s in mode_charges(p, q, W).items():
            d[(i, j)] += s
    return dict(d)


def pair_key(t1, t2, W):
    (p1, q1), (p2, q2) = t1, t2
    dp = p2 - p1
    if dp < 0 or (dp == 0 and q2 < q1):
        p1, q1, p2, q2 = p2, q2, p1, q1
        dp = -dp
    return (dp, q1 % W, q2 % W)


def pair_ok(L, W, t1, t2, memo=None):
    if memo is None:
        memo = {}
    k = pair_key(t1, t2, W)
    if k in memo:
        return memo[k]
    ok = check_adm(L, W, field_of([t1, t2], W))
    memo[k] = ok
    return ok


def pair_ok_intervals(t1, t2, W):
    """Interval-characterisation verdict: same orientation disjoint,
    opposite orientation nested. Columns = arcs mod W; rows/diagonals =
    full line segments."""
    def arc(fam, a, l):
        if fam == 'col':
            return frozenset((a + k) % W for k in range(l))
        return frozenset(range(a, a + l))
    t1i = site_intervals(*t1, W)
    t2i = site_intervals(*t2, W)
    for f1, id1, o1, a1, l1 in t1i:
        for f2, id2, o2, a2, l2 in t2i:
            if f1 == f2 and id1 == id2:
                s1, s2 = arc(f1, a1, l1), arc(f2, a2, l2)
                if o1 == o2:
                    if s1 & s2:
                        return False
                else:
                    if not (s1 <= s2 or s2 <= s1):
                        return False
    return True


def g2_gate(Ws=(5, 6, 8), window=4):
    print('== G2 mode self-consistency + interval cross-check ==')
    # mode self-consistency: third_diff vs the relative table;
    # single-mode admissibility
    for W in (3, 4, 5, 7):
        m = mode_charges(4, 2, W)
        exp = defaultdict(int)
        for (di, dj, s) in MODE_REL:
            exp[(4 + di, (2 + dj) % W)] += s
        assert m == dict(exp), ('MODE table mismatch', W, m, dict(exp))
        assert check_adm(W + 4, W, m), ('single mode not admissible?!', W)
    print('  mode self-consistency (third difference = 6-charge table) + single-mode admissibility: PASS (W=3,4,5,7)')
    # interval verdict vs direct walk verdict
    all_ok = True
    for W in Ws:
        L = 20
        bad = 0
        for dp in range(0, window + 1):
            for q1 in range(W):
                for dq in range(-W + 1, W):
                    q2 = (q1 + dq) % W
                    t1, t2 = (6, q1), (6 + dp, q2)
                    if dp == 0 and q1 == q2:
                        continue
                    v1 = pair_ok(L, W, t1, t2)
                    v2 = pair_ok_intervals(t1, t2, W)
                    if v1 != v2:
                        bad += 1
                        all_ok = False
        print('  W=%d: offset window dp <= %d, all (q1,q2): %s' % (W, window, 'PASS' if bad == 0 else 'FAIL x%d' % bad))
    return all_ok


# ---------------- G3: verdict table + search ----------------

def verdict_table(W, dp_max, L):
    """V[(dp,q1,q2)] = pair compatible? (dp >= 0; x-translation invariant;
    q mod W)"""
    memo = {}
    for dp in range(0, dp_max + 1):
        for q1 in range(W):
            for q2 in range(W):
                if dp == 0 and q1 == q2:
                    continue
                memo[(dp, q1, q2)] = check_adm(L, W, field_of([(6, q1), (6 + dp, q2)], W))
    return memo


def search_max(L, W, seconds=60, restarts=600, seed=1):
    cands = [(i, j) for i in range(2, L) for j in range(W)]
    memo = {}

    def ok(t1, t2):
        return pair_ok(L, W, t1, t2, memo)

    rnd = random.Random(seed)
    best = []
    t0 = time.time()
    for r in range(restarts):
        order = cands[:]
        rnd.shuffle(order)
        S = []
        for t in order:
            if all(ok(t, m) for m in S):
                S.append(t)
        if len(S) > len(best):
            best = S
        if time.time() - t0 > seconds * 0.6:
            break
    # branch and bound (greedy solution as lower bound, include-first)
    n = len(cands)
    bestsz = [len(best)]
    cur = []

    def rec(pos):
        if time.time() - t0 > seconds:
            raise TimeoutError
        if len(cur) + (n - pos) < bestsz[0]:
            return
        if pos == n:
            if len(cur) > bestsz[0]:
                bestsz[0] = len(cur)
                best[:] = cur[:]
            return
        t = cands[pos]
        if all(ok(t, m) for m in cur):
            cur.append(t)
            rec(pos + 1)
            cur.pop()
        rec(pos + 1)

    try:
        rec(0)
    except TimeoutError:
        pass
    return best, time.time() - t0


def all_subsets_ok(L, W, S):
    n = len(S)
    for mask in range(1, 1 << n):
        T = [S[k] for k in range(n) if (mask >> k) & 1]
        if not check_adm(L, W, field_of(T, W)):
            return (False, T)
    return (True, None)


def grid_str(L, W, S):
    g = [['.' for _ in range(L)] for _ in range(W)]
    for (i, j) in S:
        g[j][i] = 'o'
    return '\n'.join(''.join(row) for row in g)


def main():
    rep = {}
    # G1: small systems, both algorithms (brute only for E <= 24)
    rep['g1'] = g1_gate([(2, 2), (3, 2), (2, 3)])
    # a few larger sizes on the formula side (compared against torus
    # anchors: strip >= torus)
    print('== G1b larger sizes on the formula side (reference: strip >= torus anchors) ==')
    torus_anchor = {(2, 4): 272, (4, 2): 272, (3, 3): 518, (4, 3): 256, (5, 2): 256}
    rep['g1b'] = {}
    for (L, W), ta in sorted(torus_anchor.items()):
        fm = omega_formula(L, W)
        ok = fm >= ta
        print('  (%d,%d): strip_formula=%d vs torus=%d %s' % (
            L, W, fm, ta, 'PASS' if ok else 'FAIL'))
        rep['g1b']['%dx%d' % (L, W)] = {'strip': fm, 'torus': ta, 'pass': ok}

    # G2
    rep['g2'] = g2_gate()

    # G3: print the verdict table (W=8, internal verdicts, dp <= 6)
    print('== G3 offset verdict table W=8 (rows dp, columns q2-q1; x=dead o=alive .=offset absent) ==')
    W = 8
    vt = verdict_table(W, 6, 24)
    hdr = '    ' + ''.join('%2d' % (dq - W + 1) for dq in range(2 * W - 1))
    print(hdr)
    for dp in range(1, 7):
        row = []
        for dq in range(-(W - 1), W):
            col = ''
            any_bad = any_good = False
            for q1 in range(W):
                q2 = (q1 + dq) % W
                if (dp, q1, q2) in vt:
                    if vt[(dp, q1, q2)]:
                        any_good = True
                    else:
                        any_bad = True
            col = 'o' if (any_good and not any_bad) else ('x' if (any_bad and not any_good) else '?' if (any_bad and any_good) else '.')
            row.append(' ' + col)
        print('dp=%d %s' % (dp, ''.join(row)))

    # G3: per-width search
    print('== G3 maximal fully-compatible set per width (greedy restarts + B&B, within the time limit) ==')
    rep['search'] = {}
    for W in range(3, 10):
        L = 3 * W
        S, el = search_max(L, W, seconds=45)
        dens = len(S) / (L * W)
        red = all_subsets_ok(L, W, S) if len(S) <= 20 else None
        ok_red = (red is None) or red[0]
        print('  W=%d L=%d: |S|=%d density=%.4f (%.1fs) reduction check: %s' % (
            W, L, len(S), dens, el,
            ('PASS' if ok_red else 'FAIL!') if red is not None else 'skipped(>20)'))
        print(grid_str(L, W, S))
        rep['search']['%d' % W] = {'L': L, 'W': W, 'S': S, 'density': dens,
                                   'reduction_ok': ok_red if red is not None else None}
        if red is not None and not red[0]:
            print('  !! pairwise-compatible but a subset failed: ', red[1])
    with open(os.path.join(OUT_DIR, 'h2l_strip_result.json'), 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1, default=list)
    print('saved h2l_strip_result.json')


if __name__ == '__main__':
    main()
