# -*- coding: utf-8 -*-
"""Step 12: honeycomb long-chain trace method for the three alphabets
(Lx = 21..Lmax) -- asymptotic-convergence data for panel c.
Per-start integer DP (same pipeline as P0_PM_hc_trace); each series first
runs a timed probe and extrapolates, to guard against large-component blowup
(nonarith W >= 5 risk). Output: TSDHC_long_trace.csv.
Local memory < 2 GB (largest state space 59049); a series is truncated once
a single point exceeds TMAX seconds.
"""
import csv
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from P0_PM_hc_trace import TorusCtx, trace_power  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'TSDHC_long_trace.csv')

SERIES = [  # (tag, alphabet, Ly, Lmax, probe step)
    ('binary', (0, 1), 2, 150, 10),
    ('binary', (0, 1), 3, 150, 10),
    ('binary', (0, 1), 5, 150, 10),
    ('arithmetic', (0, 1, 2), 2, 150, 10),
    ('arithmetic', (0, 1, 2), 3, 150, 10),
    ('arithmetic', (0, 1, 2), 5, 120, 5),
    ('nonarith', (0, 1, 3), 2, 150, 10),
    ('nonarith', (0, 1, 3), 3, 150, 10),
    ('nonarith', (0, 1, 3), 4, 100, 5),
    ('nonarith', (0, 1, 3), 5, 100, 5),
]
TMAX = 600.0         # per-point walltime cap (s); exceeding it truncates the series
# (nonarith Ly=5 series extended from 60 to 100 to fill the fig2c data
#  points; TMAX 90->600; nona Ly5 ~40 s/pt at Lx=55, near-linear, ~100
#  expected < 2 min/pt)


def _done_points(out_path):
    """Resume support: read existing (tag, Ly) -> set(Lx) from the CSV and
    skip them on rerun."""
    done = {}
    if not os.path.exists(out_path):
        return done
    with open(out_path, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row) < 4 or not row[2].isdigit():
                continue
            done.setdefault((row[0].lstrip('\ufeff'), int(row[2])), set()).add(
                int(row[3]))
    return done


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. binary/2
    done = _done_points(OUT)
    new = not os.path.exists(OUT)
    f = open(OUT, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if new:
        w.writerow(['alphabet', 'alpha_str', 'Ly', 'Lx', 'Omega', 'lnOmega',
                    'apparent_lambda', 'seconds'])
    for tag, alpha, Ly, Lmax, probe in SERIES:
        if only and f'{tag}/{Ly}' != only:
            continue
        skip = done.get((tag, Ly), set())
        todo = [L for L in range(max(2, probe + 1), Lmax + 1)
                if L not in skip]
        if not todo:
            print(f'[{tag} Ly={Ly}] already complete (n={len(skip)}), '
                  f'skip', flush=True)
            continue
        t0 = time.time()
        ctx = TorusCtx(max(4, Lmax), Ly, alpha)
        ctx.build_states()
        n = len(ctx.states)
        print(f'[{tag} Ly={Ly}] n={n} build {time.time()-t0:.0f}s',
              flush=True)
        # probe: time L=probe first, extrapolate full-run feasibility
        t1 = time.time()
        trace_power(ctx.t0, probe)
        dt = time.time() - t1
        est = dt / probe * Lmax
        print(f'  probe L={probe}: {dt:.1f}s -> est full {est:.0f}s',
              flush=True)
        if est > 6 * 3600:
            print('  SKIP (est > 6h)', flush=True)
            continue
        done = 0
        for Lx in range(max(2, probe + 1), Lmax + 1):
            if Lx in skip:
                continue
            t2 = time.time()
            om = trace_power(ctx.t0, Lx)
            dt2 = time.time() - t2
            ln = math.log(om)
            w.writerow([tag, ''.join(map(str, alpha)), Ly, Lx, om,
                        f'{ln:.6f}', f'{ln/Lx:.6f}', f'{dt2:.1f}'])
            f.flush()
            done += 1
            if dt2 > TMAX:
                print(f'  STOP at Lx={Lx} (single point {dt2:.0f}s > '
                      f'{TMAX:.0f}s)', flush=True)
                break
            if done % 10 == 0:
                print(f'  Lx={Lx} lnOmega={ln:.2f} ({dt2:.1f}s/pt)', flush=True)
        print(f'  [{tag} Ly={Ly}] done {done} pts, '
              f'total {time.time()-t0:.0f}s', flush=True)
        del ctx
    f.close()
    print('all done ->', OUT, flush=True)


if __name__ == '__main__':
    main()
