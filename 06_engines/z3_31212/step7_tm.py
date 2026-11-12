# -*- coding: utf-8 -*-
"""Step 7: 3.12.12 mid-subsystem transfer-matrix spectral radius rho(W).

Principle: the mid subsystem (the entropy carrier after row splitting; 3W mid keys
per column, 2 equations per dodecagon) on a width-W cylinder, transferred along L:
T; state = the assignment of the 2W mid keys on the cut line (packed
[A_0..A_{W-1}, B_0..B_{W-1}]); one step = process one column of W dodecagons
(2 equations each; local slots: pA=A_i, pB=B_i (state), E_i/E_{i-1} (in-column),
nA=A'_i, nB=B'_{i-1} (next cut)). The Perron root rho(W) = e^{lambda(W)} is the
exact per-column growth factor as L -> inf -- no finite-L transients.

Reading:
  gate  k=2: rho(W) == 2 exactly (binary non-resonant Omega_mid = 2^(L+W+1)
        => lambda = ln 2 for all W);
  k=3: ln rho(W) flat => perimeter law, s_3(inf)=0 settled (numerical-adjudication
        level, on the asymptotic object itself); linear => s_mid3 = slope/3
        (total, plus lambda_cut = ln 3, over 3W).
  k=4 included (monotonicity completeness).

Geometry (all derived from build_31212_rect local coordinates, asserted):
  dodecagon centroids -> (i,j); its 6 mid edges bucketed by midpoint direction:
  0 deg -> E_i, 60 deg -> A'_i, 120 deg -> B'_{i-1}, 180 deg -> E_{i-1},
  240 deg -> A_i (state), 300 deg -> B_i (state);
  adjacency: +/-A1 (in-column), A2 / A2-A1 (to j+1), -A2 / A1-A2 (to j-1).

Fail-closed: dual gates on the state count n = k^{2W} and the measured nnz
(2e8 states / 2e10 nnz; skip beyond); memory printed up front.
Parallelism: state rows blocked over a Pool (per-row DFS with table lookup);
power iteration vectorized with bincount.
Run: python3 step7_tm.py [Wmax]   (k=2 gate -> k=3 -> k=4)
Output: TSD31212_tm_rho.csv (append) + stdout.

Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step3_dp_strips import build_31212_rect  # noqa: E402

OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'TSD31212_tm_rho.csv')
GATE_STATES = 2 * 10**8
GATE_NNZ = 2 * 10**10


def extract_geometry(W, L=4):
    """Returns per-dodecagon (i, j, {bucket: eid}) and eqs {fi: (eqX, eqY)} (eid -> coef)."""
    t = build_31212_rect(W, L)
    N = W * L
    assert (t.V, t.E, t.F) == (6 * N, 9 * N, 3 * N)
    assert dict(sorted(t.face_census().items())) == {3: 2 * N, 12: N}
    size_of = [len(f[0]) for f in t.faces]
    g = 1 + 2 / math.sqrt(3)
    D0 = math.sqrt(3) * g
    A1 = np.array((D0, 0.0))
    A2 = np.array((D0 / 2, D0 * math.sqrt(3) / 2))
    Minv = np.linalg.inv(np.array([[A1[0], A2[0]], [A1[1], A2[1]]]))
    dodecs = [fi for fi in range(t.F) if size_of[fi] == 12]
    ij = {}
    for fi in dodecs:
        c = np.mean(t.faces[fi][1], axis=0)
        uv = Minv @ c
        key = (int(round(uv[0])) % W, int(round(uv[1])) % L)
        assert key not in ij, (fi, key)
        ij[fi] = key
    assert len(ij) == N
    edge_faces = defaultdict(list)
    for fi in range(t.F):
        for (eid, sgn) in t.face_edges[fi]:
            edge_faces[eid].append(fi)
    mid = {eid for eid, fs in edge_faces.items()
           if size_of[fs[0]] == 12 and size_of[fs[1]] == 12}
    assert len(mid) == 3 * N, (len(mid), 3 * N)
    info = {}
    for fi in dodecs:
        i, j = ij[fi]
        pts = t.faces[fi][1]
        c = np.mean(pts, axis=0)
        buckets = {}
        fe = t.face_edges[fi]
        npts = len(pts)
        for kk, (eid, sgn) in enumerate(fe):
            if eid not in mid:
                continue
            mid_pt = (pts[kk] + pts[(kk + 1) % npts]) / 2.0
            ang = math.degrees(math.atan2(mid_pt[1] - c[1],
                                          mid_pt[0] - c[0])) % 360
            b = int(round(ang / 60.0)) % 6
            assert b not in buckets, (fi, b, ang)
            buckets[b] = eid
        assert len(buckets) == 6, (fi, buckets)
        info[fi] = (i, j, buckets)
    eqs = {}
    midset = mid
    for fi in dodecs:
        mx = defaultdict(int)
        my = defaultdict(int)
        for (eid, sgn) in t.face_edges[fi]:
            if eid not in midset:
                continue
            mx[eid] += int(round(2 * float(t.xrows[fi][eid].b)))
            my[eid] += int(round(2 * float(t.yrows[fi][eid].a)))
        eqs[fi] = ({kk: v for kk, v in mx.items() if v != 0},
                   {kk: v for kk, v in my.items() if v != 0})
        for d in eqs[fi]:
            for v in d.values():
                assert abs(v) in (1, 2), d
    return ij, info, eqs


def build_tables(W, k, L=4):
    """Per-dodecagon transfer tables: tab[i][(pA, pB, e_prev)] = [(E_i, nA_i, nB_{i-1}), ...]
    (for i = W-1 the table values are pairs (nA, nB); E_{W-1} is enumerated by the outer layer)."""
    ij, info, eqs = extract_geometry(W, L)
    # per dodecagon: eid -> slot name (local)
    per = {}
    for fi, (i, j, bkt) in info.items():
        slot = {0: ('E', i), 3: ('E', (i - 1) % W),
                1: ('N', i, 'A'), 2: ('N', (i - 1) % W, 'B'),
                4: ('S', i, 'A'), 5: ('S', i, 'B')}
        per[fi] = (i, {eid: slot[b] for b, eid in bkt.items()})
    # assert: the 6 slots of each dodecagon are exactly {E_i, E_{i-1}, NA_i, NB_{i-1}, SA_i, SB_i}
    for fi, (i, m) in per.items():
        names = sorted(m.values())
        expect = sorted([('E', i), ('E', (i - 1) % W),
                         ('N', i, 'A'), ('N', (i - 1) % W, 'B'),
                         ('S', i, 'A'), ('S', i, 'B')])
        assert names == expect, (fi, names)
    tabs = []
    j0 = 2  # any inner column (geometry invariant under j translation); tables built per in-column i = 0..W-1
    for i in range(W):
        fi = [f for f in per if per[f][0] == i and ij[f][1] == j0]
        assert len(fi) == 1, (i, fi)
        fi = fi[0]
        eqX, eqY = eqs[fi]
        # bucket indices: 0=0 deg (E_i) 1=60 deg (nA_i) 2=120 deg (nB_{i-1})
        # 3=180 deg (E_{i-1}) 4=240 deg (pA_i) 5=300 deg (pB_i)
        terms = []
        for (b, dst) in ((4, 'pA'), (5, 'pB'), (3, 'eprev'),
                         (0, 'E'), (1, 'nA'), (2, 'nB')):
            eid = info[fi][2][b]
            terms.append((dst, eqX.get(eid, 0), eqY.get(eid, 0)))
        tab = defaultdict(list)
        rng = range(k)
        if i < W - 1:
            # keys = (pA,pB,E_{i-1}); free = (E_i, nA_i, nB_{i-1})
            for pA in rng:
                for pB in rng:
                    for ep in rng:
                        for vE in rng:
                            for vna in rng:
                                for vnb in rng:
                                    vals = {'pA': pA, 'pB': pB, 'eprev': ep,
                                            'E': vE, 'nA': vna, 'nB': vnb}
                                    sx = sum(c * vals[t] for t, c, _ in terms)
                                    sy = sum(c * vals[t] for t, _, c in terms)
                                    if sx == 0 and sy == 0:
                                        tab[(pA, pB, ep)].append((vE, vna, vnb))
        else:
            # i=W-1: E_{W-1} enumerated by the outer layer, E_{W-2} already assigned at i=W-2
            # keys = (pA,pB,E_{W-1},E_{W-2}); free = (nA_{W-1}, nB_{W-2})
            for pA in rng:
                for pB in rng:
                    for vE in rng:
                        for vep2 in rng:
                            for vna in rng:
                                for vnb in rng:
                                    vals = {'pA': pA, 'pB': pB, 'eprev': vep2,
                                            'E': vE, 'nA': vna, 'nB': vnb}
                                    sx = sum(c * vals[t] for t, c, _ in terms)
                                    sy = sum(c * vals[t] for t, _, c in terms)
                                    if sx == 0 and sy == 0:
                                        tab[(pA, pB, vE, vep2)].append((vna, vnb))
        tabs.append(dict(tab))
    return tabs


def _eid_of_bucket(info, fi, b):
    return info[fi][2][b]


# ---- worker globals ----
G = {}


def init_worker(W, k):
    G['W'] = W
    G['k'] = k
    G['tabs'] = build_tables(W, k)


def enum_row(row_idx):
    """Row index -> state [A_0..A_{W-1}, B_0..B_{W-1}]: all transitions (to_idx, cnt)."""
    W, k, tabs = G['W'], G['k'], G['tabs']
    d = []
    x = row_idx
    for _ in range(2 * W):
        d.append(x % k)
        x //= k
    to = defaultdict(int)
    for e_last in range(k):
        # (i, e_prev, new dict); the i=W-1 table key = (pA, pB, e_last, new[W-2])
        results = [(e_last, {})]
        for i in range(W):
            pA = d[i]
            pB = d[W + i]
            nxt = []
            for (e_prev, new) in results:
                if i == W - 1:
                    key = (pA, pB, e_last, new[W - 2])
                    for (vna, vnb) in tabs[i].get(key, ()):
                        nn = dict(new)
                        nn[W - 1] = e_last
                        nn[W + i] = vna
                        nn[2 * W + (i - 1) % W] = vnb
                        nxt.append((e_prev, nn))
                else:
                    key = (pA, pB, e_prev)
                    for (vE, vna, vnb) in tabs[i].get(key, ()):
                        nn = dict(new)
                        nn[i] = vE
                        nn[W + i] = vna
                        nn[2 * W + (i - 1) % W] = vnb
                        nxt.append((vE, nn))
            results = nxt
        for (_, new) in results:
            # the next state encodes only the cut keys [A'_0..A'_{W-1}, B'_0..B'_{W-1}]
            # (E are in-column keys, not in the frontier)
            to_idx = 0
            for i in range(W):
                to_idx += new[W + i] * (k ** i)
                to_idx += new[2 * W + i] * (k ** (W + i))
            to[to_idx] += 1
    return [(row_idx, t, c) for t, c in to.items()]


def build_tm(W, k, workers):
    n = k ** (2 * W)
    assert n <= GATE_STATES, (W, k, n)
    t0 = time.time()
    chunk = max(1, n // (workers * 8))
    chunks = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    froms, tos, cnts = [], [], []
    nnz = 0
    with Pool(workers, initializer=init_worker, initargs=(W, k)) as pool:
        for res in pool.imap_unordered(enum_rows_chunk, chunks, chunksize=1):
            f, t_, c = res
            froms.append(f)
            tos.append(t_)
            cnts.append(c)
            nnz += len(f)
            if nnz > GATE_NNZ:
                pool.terminate()
                raise MemoryError(f'nnz gate {nnz} > {GATE_NNZ}')
    from_i = np.concatenate(froms)
    to_i = np.concatenate(tos)
    cnt = np.concatenate(cnts).astype(np.float64)
    return n, from_i, to_i, cnt, time.time() - t0


def enum_rows_chunk(bounds):
    a, b = bounds
    F, T, C = [], [], []
    for r in range(a, b):
        for (f, t, c) in enum_row(r):
            F.append(f)
            T.append(t)
            C.append(c)
    return F, T, C


def perron(n, from_i, to_i, cnt, iters=600, tol=1e-13):
    v = np.ones(n) / n
    rho_old = 1.0
    hist = []
    for it in range(1, iters + 1):
        w = np.bincount(to_i, weights=cnt * v[from_i], minlength=n)
        s = w.sum()
        rho = s  # after L1-normalizing v, sum(w) = rho * sum(v)
        w /= s
        rel = abs(rho - rho_old) / rho
        v = w
        rho_old = rho
        hist.append(rho)
        if rel < tol:
            break
    tail = hist[-10:]
    return rho, it, rel, min(tail), max(tail)


def append_csv(row):
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['k', 'W', 'n_states', 'nnz', 'ln_rho', 'rho',
                        'iters', 'relerr', 'build_s', 'iter_s', 'tail_lo', 'tail_hi'])
        w.writerows(row)


def main():
    workers = min(140, os.cpu_count() or 8)
    Wmax = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    plan = [(2, range(2, Wmax + 1)), (3, range(2, Wmax + 1)), (4, range(2, min(Wmax, 6) + 1))]
    for k, Ws in plan:
        for W in Ws:
            n = k ** (2 * W)
            if n > GATE_STATES:
                print(f'k={k} W={W}: skip (states {n:.3g} > gate)', flush=True)
                continue
            try:
                n, f, t_, c, dtb = build_tm(W, k, workers)
            except MemoryError as e:
                print(f'k={k} W={W}: NNZ GATE {e}', flush=True)
                continue
            t1 = time.time()
            rho, it, rel, rlo, rhi = perron(n, f, t_, c)
            lnr = math.log(rho)
            print(f'k={k} W={W}: states={n} nnz={len(f)} rho={rho:.10f} '
                  f'ln_rho={lnr:.6f} iters={it} rel={rel:.2e} '
                  f'tail=[{rlo:.8f},{rhi:.8f}] '
                  f'[build {dtb:.0f}s iter {time.time()-t1:.0f}s]', flush=True)
            append_csv([[k, W, n, len(f), f'{lnr:.8f}', f'{rho:.8f}', it,
                         f'{rel:.2e}', f'{dtb:.0f}', f'{time.time()-t1:.0f}',
                         f'{rlo:.8f}', f'{rhi:.8f}']])
            del f, t_, c


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
