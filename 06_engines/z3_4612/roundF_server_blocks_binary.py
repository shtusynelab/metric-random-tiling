# -*- coding: utf-8 -*-
"""4.6.12 two-value {0,1} blocked-DFS driver (parallel blocked DFS).

Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

Blocks the parametrized-DFS parameter space by pinning the first b
BFS-order parameters; 2^b blocks; Pool parallel; per-block CSV
checkpoint (resume skips completed blocks). Summary:
Omega_01 = sum(blocks) x 2  (complement-shift orbit, gauge x_0 = 0),
appended to RTH4612_binary_anchors.csv as engine 'blocked-dfs' so the
frontier DP crosscheck picks it up.

Usage: python3 roundF_server_blocks_binary.py --L 3 --b 8 --workers 140
Memory: <120 MB per worker; 140 workers <= 18 GB (server 377 GB).
"""
import argparse
import csv
import itertools
import math
import os
import time
from multiprocessing import Pool

from roundE_4612_binary_anchors import count_param_dfs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'RTH4612_binary_L{L}_blocks.csv')
ANCHOR_CSV = os.path.join(HERE, 'RTH4612_binary_anchors.csv')


def block_count(task):
    Lx, Ly, b, blk = task
    t0 = time.time()
    cnt, nf, n_cls, N, dt = count_param_dfs(Lx, Ly, vals=(0, 1),
                                            block=(b, blk))
    return ('_'.join(map(str, blk)), cnt, N, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=None,
                    help='square torus size (L x L)')
    ap.add_argument('--Lx', type=int, default=None)
    ap.add_argument('--Ly', type=int, default=None)
    ap.add_argument('--b', type=int, default=8)
    ap.add_argument('--workers', type=int, default=140)
    args = ap.parse_args()
    if (args.Lx is None) != (args.Ly is None):
        ap.error('--Lx and --Ly must be given together')
    if args.Lx is None:
        if args.L is None:
            ap.error('need --L or --Lx/--Ly')
        Lx = Ly = args.L
    else:
        Lx, Ly = args.Lx, args.Ly
    b = args.b
    out = OUT.format(L=f'{Lx}x{Ly}')
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in csv.DictReader(f):
                done.add(line['block'])
    blocks = [blk for blk in itertools.product((0, 1), repeat=b)
              if '_'.join(map(str, blk)) not in done]
    print(f'L={Lx}x{Ly} b={b}: {2 ** b - len(done)}/{2 ** b} blocks to go '
          f'(workers={args.workers})', flush=True)
    new = not os.path.exists(out)
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if new:
        w.writerow(['block', 'Omega_block', 'seconds'])
        f.flush()
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (key, cnt, N, dt) in enumerate(
                pool.imap_unordered(block_count,
                                    [(Lx, Ly, b, blk) for blk in blocks])):
            w.writerow([key, cnt, round(dt, 1)])
            f.flush()
            print(f'  [{i + 1}/{len(blocks)}] {key}: {cnt} ({dt:.0f}s) '
                  f'elapsed {time.time() - t0:.0f}s', flush=True)
    f.close()
    tot, N = 0, None
    with open(out) as f:
        for line in csv.DictReader(f):
            tot += int(line['Omega_block'])
            N = Lx * Ly
    omega_total = tot * 2            # x2 complement-shift orbit (x_0 = 0)
    s01 = math.log(omega_total) / (6 * N)
    if Lx == Ly:
        # monotonicity gate (square tori only): binary <= ternary Knuth est
        KNUTH = {3: 6.566e12, 4: 8.351e19}
        if Lx in KNUTH:
            ok = omega_total <= KNUTH[Lx] * 1.01
            print(f'    monotonicity vs ternary Knuth est {KNUTH[Lx]:.3e}: '
                  f'{"OK" if ok else "VIOLATED -- check!"}', flush=True)
            assert ok, ('binary > ternary estimate', Lx, omega_total)
    print(f'L={Lx}x{Ly} BINARY summary: Omega_01={omega_total}  '
          f'ln={math.log(omega_total):.4f}  s_01={s01:.4f}', flush=True)
    with open(out.replace('_blocks.csv', '_summary.csv'), 'w') as f:
        f.write(f'L={Lx}x{Ly} Omega_01={omega_total} s01={s01:.6f}\n')
    with open(ANCHOR_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([Lx, '{0,1}', f'blocked-dfs-{Lx}x{Ly}', omega_total,
                    f'{math.log(omega_total):.4f}', f'{s01:.6f}',
                    '', '', N, ''])
    print('anchor CSV updated', flush=True)


if __name__ == '__main__':
    main()
