# -*- coding: utf-8 -*-
"""Step 9 (non-arithmetic ternary honeycomb re-adjudication): transfer-matrix
spectral radius rho(W) of the {0,1,3} edge-metric system.

Background: 3.12.12 measurements showed that this system family has exact
power-law transient sectors, so per-width lambda(W) readings from short-chain
DP/trace methods are unreliable. The non-arithmetic ternary honeycomb {0,1,3}
previously had only short-chain evidence (A0-2 strip W=3 lambda=3.55; torus
Ly=2/3 point counts with a strong even-odd oscillating sub-law shape) -- under
the new criterion it must be re-adjudicated with the TM Perron root (exact
L -> infinity asymptotics, no transients).

Method: same T-matrix pipeline as step8_hc_tm.py (single-column transfer
operator, Perron vector iteration) with the alphabet changed to (0,1,3).
Validation gates:
  1) anchor gate: Tr(T^Lx) reproduces archived exact counts (is_anchor rows
     of P0_PM_torus_trace_shape_classes_Ly2Ly3.csv, previously accepted via
     independent DFS cross-checks): Ly=2: Lx=4/6 -> 6669/61965;
     Ly=3: Lx=3..6 -> 19707/6561/19683/576681;
  2) serial/parallel build cross-check gate: the t0 graph from serial
     build_states (original lazy single-thread) equals the parallel
     build_states_pool graph bit-for-bit (Ly=2,3; guards against parallel-path
     regression);
  3) containment gate: {0,1} subset of {0,1,3} => Omega_013 >= Omega_01 for
     every (Lx,Ly) (direct anchor lookup);
  4) growth-rate gate: the Perron rate agrees with the long-chain Tr(T^L)
     growth rate (|ln Tr(T^L)/L - ln rho| < 1e-2, L = 6W).

Memory: reachable states <= 3^(2Ly); Ly<=6 full-space bound 531k (dict peak
~0.5 GB, fits locally); Ly=7 bound 4.8M (dict peak ~2.5-3 GB) exceeds the
2 GB gate => run only on a machine with sufficient RAM.
Run: python3 step9_hc_tm_nonarith.py [maxLy]  (default 6; use 7 on a
large-RAM machine)
Output: TSDHC_tm_rho_nonarith.csv + stdout (kept in a separate file from the
step8 table to avoid concurrent writes to TSDHC_tm_rho.csv; merging is done
by the collect step).
"""
import csv
import math
import os
import sys
import time
from itertools import product

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from P0_PM_hc_trace import TorusCtx, trace_power  # noqa: E402
from step8_hc_tm import build_states_pool, build_sparse, perron  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'TSDHC_tm_rho_nonarith.csv')

ALPHA = (0, 1, 3)

# archived exact anchors (P0_PM_torus_trace_shape_classes_Ly2Ly3.csv,
# is_anchor=True rows)
ANCHORS = {
    2: [(4, 6669), (6, 61965)],
    3: [(3, 19707), (4, 6561), (5, 19683), (6, 576681)],
}
# exact binary {0,1} counts at the same sizes (containment-gate lower bound;
# from ANCHORS / old CSVs)
BIN_ANCHORS = {
    (2, 2): 64, (3, 2): 64, (4, 2): 272, (6, 2): 256,
    (3, 3): 518, (4, 3): 256, (5, 3): 512, (6, 3): 5638,
}


def serial_parallel_gate(Ly, workers):
    """Bit-for-bit cross-check of the t0 graph: serial build_states vs
    parallel build_states_pool."""
    Lx = max(4, Ly)
    ctx_s = TorusCtx(Lx, Ly, ALPHA)
    t0s, _ = ctx_s.build_states()
    ctx_p, t0p, n = build_states_pool(ALPHA, Ly, Lx, workers)
    assert n == len(t0s), (Ly, n, len(t0s))
    for s, o in t0s.items():
        assert t0p.get(s) == o, f't0 mismatch at {s} (Ly={Ly})'
    print(f'  gate serial==parallel Ly={Ly}: |S|={n} t0 graphs identical OK',
          flush=True)
    return ctx_p, t0p, n


def containment_gate(ctx, Ly):
    for (Lx, _), om_bin in sorted(BIN_ANCHORS.items()):
        if Lx > ctx.Lx:
            continue
        tr = trace_power(ctx.t0, Lx)
        assert tr >= om_bin, f'containment fail {Lx}x{Ly}: {tr} < {om_bin}'
        print(f'  gate {Lx}x{Ly}: Omega_013={tr} >= Omega_01={om_bin} OK',
              flush=True)


def growth_gate(ctx, rho, Ly, Lmax=None):
    """One-sided rigorous bound: (Tr T^L)^(1/L) <= rho * n^(1/L) (entrywise
    bound for non-negative matrices). If the TM rho were underestimated, the
    finite-L growth rate would break this bound. No general lower-side bound
    exists (complex/negative eigenvalue cancellation can make Tr smaller),
    so only informational prints."""
    Lmax = Lmax or 6 * Ly
    tr = trace_power(ctx.t0, Lmax)
    rate = math.log(tr) / Lmax
    bound = math.log(rho) + math.log(len(ctx.t0)) / Lmax
    assert rate <= bound + 1e-9, \
        f'growth fail Ly={Ly}: rate={rate:.6f} > ln_rho+ln n/L={bound:.6f}'
    print(f'  gate Ly={Ly} L={Lmax}: lnTr/L={rate:.6f} <= ln_rho+ln n/L='
          f'{bound:.6f} OK (gap={bound-rate:.4f})', flush=True)


def rows_nonarith(Ly_max, workers, Ly_min=2):
    for Ly in range(Ly_min, Ly_max + 1):
        t0 = time.time()
        Lx = max(4, Ly)
        ctx, t0graph, n = build_states_pool(ALPHA, Ly, Lx, workers)
        print(f'nonarith Ly={Ly}: reachable states n={n} (build {time.time()-t0:.0f}s)',
              flush=True)
        for Lxg, exp in ANCHORS.get(Ly, []):
            tr = trace_power(t0graph, Lxg)
            assert tr == exp, (Ly, Lxg, tr, exp)
            print(f'  gate Ly={Ly} Lx={Lxg}: Tr={tr} == {exp} OK', flush=True)
        if Ly <= 3:                      # serial/parallel gate only at small sizes (serial runtime manageable)
            ctx, t0graph, n = serial_parallel_gate(Ly, workers)
        containment_gate(ctx, Ly)
        F, Tt, C = build_sparse(t0graph)
        t2 = time.time()
        rho, it, rel, rlo, rhi = perron(n, F, Tt, C)
        lnr = math.log(rho)
        if os.environ.get('SKIP_GROWTH'):
            print(f'  growth gate SKIPPED (SKIP_GROWTH=1; companion eigs '
                  f'cross-check covers this)', flush=True)
        else:
            growth_gate(ctx, rho, Ly)
        print(f'nonarith Ly={Ly}: ln_rho={lnr:.6f} rho={rho:.6f} iters={it} '
              f'rel={rel:.2e} tail=[{rlo:.6f},{rhi:.6f}] '
              f'[sparse {t2-t0:.0f}s iter {time.time()-t2:.0f}s]', flush=True)
        yield ['nonarith', ''.join(map(str, ALPHA)), Ly, n, len(F),
               f'{lnr:.8f}', f'{rho:.8f}', it, f'{rel:.2e}',
               f'{rlo:.8f}', f'{rhi:.8f}', f'{time.time()-t0:.0f}']


def main():
    maxLy = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    minLy = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    workers = int(os.environ.get('TM_WORKERS', 0)) or \
        min(32 if os.environ.get('ON_SERVER') else 6, os.cpu_count() or 8)
    exists = os.path.exists(OUT_CSV)
    f = open(OUT_CSV, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not exists:
        w.writerow(['alphabet', 'alpha_str', 'Ly', 'n_states', 'nnz',
                    'ln_rho', 'rho', 'iters', 'relerr', 'tail_lo', 'tail_hi',
                    'time_s'])
    for row in rows_nonarith(maxLy, workers, minLy):
        w.writerow(row)
        f.flush()
    f.close()
    print('done', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
