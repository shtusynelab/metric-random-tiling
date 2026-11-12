# -*- coding: utf-8 -*-
"""Step 12b: multicore parallel backfill of the nonarith Ly=5 long chain
(parallel wrapper around the single-process step12).

Each worker builds its own TorusCtx independently (build takes seconds,
negligible) and points are sharded by Lx via imap_unordered; the main
process collects results and then appends to TSDHC_long_trace.csv
sequentially, one point at a time (multi-process append to the same file
causes BOM pollution and header races; writing in Lx order keeps row order).
Memory: state space <= 3^10 = 59049 per worker, 10 workers < 10 GB.
Usage: python3 step12b_parallel.py [lo:hi] (default 56:100; workers=10)
"""
import csv
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from P0_PM_hc_trace import TorusCtx, trace_power  # noqa: E402
import step12_long_trace as s12  # noqa: E402

ALPHA = (0, 1, 3)
TAG, LY = 'nonarith', 5
NW = 10
TMAX_WARN = 600.0          # per-point timeout warning (data still exact; cost bookkeeping only)


def work(lx):
    ctx = TorusCtx(max(4, lx), LY, ALPHA)
    ctx.build_states()
    t0 = time.time()
    om = trace_power(ctx.t0, lx)
    return lx, om, time.time() - t0


def main():
    lo, hi = 56, 100
    if len(sys.argv) > 1:
        lo, hi = (int(v) for v in sys.argv[1].split(':'))
    lxs = [L for L in range(lo, hi + 1)
           if L not in s12._done_points(s12.OUT).get((TAG, LY), set())]
    if not lxs:
        print('nothing to do (all points already in CSV)', flush=True)
        return
    print(f'[nonarith Ly={LY}] {len(lxs)} pts ({lxs[0]}..{lxs[-1]}), '
          f'{NW} workers', flush=True)
    t0 = time.time()
    results = []
    with Pool(NW) as pool:
        for lx, om, dt in pool.imap_unordered(work, lxs):
            results.append((lx, om, dt))
            warn = ' ** >TMAX_WARN' if dt > TMAX_WARN else ''
            print(f'  Lx={lx} lnOmega={math.log(om):.3f} ({dt:.0f}s)'
                  f'{warn}  [{len(results)}/{len(lxs)}]', flush=True)
    results.sort()
    with open(s12.OUT, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        for lx, om, dt in results:
            ln = math.log(om)
            w.writerow([TAG, ''.join(map(str, ALPHA)), LY, lx, om,
                        f'{ln:.6f}', f'{ln / lx:.6f}', f'{dt:.1f}'])
    print(f'all done: {len(results)} pts in {time.time() - t0:.0f}s '
          f'-> {s12.OUT}', flush=True)


if __name__ == '__main__':
    main()
