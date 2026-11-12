# -*- coding: utf-8 -*-
"""Step 8 (binary honeycomb cross-check): transfer-matrix spectral radius
rho(W) of the honeycomb edge-metric system.

Background: 3.12.12 measurements showed that a per-window bit-verified
"exact power-law closure" can be a transient sector (2^(L+W+1) breaks at
W >= 5). The binary honeycomb perimeter-law closure (Omega = 2^{Lx+Ly+1}
form, bit-verified at non-resonant windows) has the same epistemic status,
so this script takes the Perron root of the same T matrix directly (exact
L -> infinity asymptotics, no transients) as the final check:
  binary (0,1): ln rho(Ly) == ln2 => perimeter law promoted to TM level,
               strengthening the control line;
               ln rho(Ly) > ln2 (large Ly) => binary honeycomb is a weak
               carrier (rewrites the zero-entropy anchor).
  ternary (0,1,2): lambda(W) = ln rho(W)/W -> 0.45? => honeycomb 0.45
               promoted to TM level (no extrapolation needed).

Validation gates (must pass): Tr(T^Lx) reproduces archived exact counts:
  binary Ly=2: Lx=3/4/5/7 -> 64/272/256/1024; Ly=3: Lx=3/4/5/6 ->
  518/256/512/5638; ternary Ly=3: Lx=3/4/5/6 -> 22767/30321/185523/3267903.
Run: python3 step8_hc_tm.py [maxLy]   (binary up to maxLy, ternary up to
maxLy-1)
Output: TSDHC_tm_rho.csv + stdout. Memory: binary <= 2^16 states; ternary
Ly=7 ~ 4.8M states (reachable set smaller), < 2 GB; Ly=8 k=3 skipped (state
gate). Build parallelized: Pool chunked by starting tuple.
"""
import csv
import math
import os
import sys
import time
from collections import defaultdict
from itertools import product
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from P0_PM_hc_trace import TorusCtx, trace_power  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'TSDHC_tm_rho.csv')

ANCHORS = {
    (2, 0, 1): [(3, 64), (4, 272), (5, 256), (7, 1024)],
    (3, 0, 1): [(3, 518), (4, 256), (5, 512), (6, 5638)],
    (3, 0, 1, 2): [(3, 22767), (4, 30321), (5, 185523), (6, 3267903)],
}

_G = {}


def _init(alpha, Ly, Lx):
    _G['ctx'] = TorusCtx(Lx, Ly, alpha)


def _work(s_tuple):
    ctx = _G['ctx']
    o = ctx.column_transfer({s_tuple: 1})
    return s_tuple, (dict(o) if o else None)


def build_states_pool(alpha, Ly, Lx, workers):
    ctx = TorusCtx(Lx, Ly, alpha)
    t0 = {}
    all_starts = product(alpha, repeat=2 * Ly)
    with Pool(workers, initializer=_init, initargs=(alpha, Ly, Lx)) as pool:
        for s_tuple, o in pool.imap_unordered(_work, all_starts,
                                              chunksize=max(1, 20000 // max(1, Ly))):
            if o:
                t0[s_tuple] = o
    t0set = set(t0)          # build once; rebuilding set(t0) per assert row would be O(n^2)
    for s, o in t0.items():
        assert set(o) <= t0set, f'T not closed at {s}'
    ctx.t0, ctx.states = t0, t0set
    return ctx, t0, len(t0)


def build_sparse(t0):
    states = sorted(t0)
    idx = {s: i for i, s in enumerate(states)}
    F, Tt, C = [], [], []
    for s in states:
        fi = idx[s]
        for o, w in t0[s].items():
            F.append(fi)
            Tt.append(idx[o])
            C.append(w)
    return (np.array(F), np.array(Tt), np.array(C, dtype=np.float64))


def perron(n, F, Tt, C, iters=2000, tol=1e-13):
    v = np.ones(n) / n
    rho = 1.0
    rel = 1.0
    hist = []
    for it in range(1, iters + 1):
        w = np.bincount(Tt, weights=C * v[F], minlength=n)
        s = w.sum()
        rel = abs(s - rho) / s if it > 1 else 1.0
        rho = s
        v = w / s
        hist.append(rho)
        if rel < tol:
            break
    return rho, it, rel, min(hist[-8:]), max(hist[-8:])


def trace_gates(t0, Ly, alpha):
    key = (Ly,) + tuple(alpha)
    for Lx, exp in ANCHORS.get(key, []):
        tr = trace_power(t0, Lx)
        assert tr == exp, (Ly, Lx, tr, exp)
        print(f'  gate Ly={Ly} Lx={Lx}: Tr={tr} == {exp} OK', flush=True)


def rows_for(alpha, Ly_max, tag, workers, Ly_min=2):
    for Ly in range(Ly_min, Ly_max + 1):
        t0 = time.time()
        Lx = max(4, Ly)
        ctx, t0graph, n = build_states_pool(alpha, Ly, Lx, workers)
        print(f'{tag} Ly={Ly}: reachable states n={n} (build {time.time()-t0:.0f}s)',
              flush=True)
        trace_gates(t0graph, Ly, alpha)
        F, Tt, C = build_sparse(t0graph)
        t2 = time.time()
        rho, it, rel, rlo, rhi = perron(n, F, Tt, C)
        lnr = math.log(rho)
        print(f'{tag} Ly={Ly}: ln_rho={lnr:.6f} rho={rho:.6f} iters={it} '
              f'rel={rel:.2e} tail=[{rlo:.6f},{rhi:.6f}] '
              f'[sparse {t2-t0:.0f}s iter {time.time()-t2:.0f}s]', flush=True)
        yield [tag, ''.join(map(str, alpha)), Ly, n, len(F),
               f'{lnr:.8f}', f'{rho:.8f}', it, f'{rel:.2e}',
               f'{rlo:.8f}', f'{rhi:.8f}', f'{time.time()-t0:.0f}']


def main():
    # single-point mode: step8_hc_tm.py one <binary|ternary> <Ly> [workers]
    if len(sys.argv) > 1 and sys.argv[1] == 'one':
        tag = sys.argv[2]
        alpha = (0, 1) if tag == 'binary' else (0, 1, 2)
        Ly = int(sys.argv[3])
        wk = int(sys.argv[4]) if len(sys.argv) > 4 else 45
        exists = os.path.exists(OUT_CSV)
        f = open(OUT_CSV, 'a', newline='', encoding='utf-8-sig')
        w = csv.writer(f)
        if not exists:
            w.writerow(['alphabet', 'alpha_str', 'Ly', 'n_states', 'nnz',
                        'ln_rho', 'rho', 'iters', 'relerr', 'tail_lo',
                        'tail_hi', 'time_s'])
        for row in rows_for(alpha, Ly, tag, wk, Ly_min=Ly):
            w.writerow(row)
            f.flush()
        f.close()
        print('done', flush=True)
        return
    maxLy = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    workers = min(140, os.cpu_count() or 8)
    exists = os.path.exists(OUT_CSV)
    f = open(OUT_CSV, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not exists:
        w.writerow(['alphabet', 'alpha_str', 'Ly', 'n_states', 'nnz',
                    'ln_rho', 'rho', 'iters', 'relerr', 'tail_lo', 'tail_hi',
                    'time_s'])
    for tag, alpha, Lym in (('binary', (0, 1), maxLy),
                            ('ternary', (0, 1, 2), min(maxLy, 7))):
        for row in rows_for(alpha, Lym, tag, workers):
            w.writerow(row)
            f.flush()
    f.close()
    print('done', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
