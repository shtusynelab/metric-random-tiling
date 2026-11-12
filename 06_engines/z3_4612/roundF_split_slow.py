# -*- coding: utf-8 -*-
"""Split-and-finish for slow prefix blocks of the 4x3 binary count.

Reads RTH4612_binary_L4x3_blocks.csv (b=12 blocks), finds the missing
(unfinished) 12-bit prefixes, splits each into 16 sub-blocks by pinning
4 more BFS-order parameters (block=(16, prefix12+ext4)), runs them in a
Pool, sums back to one 12-bit row per parent, and rewrites the CSV +
summary + anchor row. Completed blocks are untouched.

Usage: python3 roundF_split_slow.py --Lx 4 --Ly 3 --workers 70
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
BLOCKS = os.path.join(HERE, 'RTH4612_binary_L4x3_blocks.csv')
ANCHOR_CSV = os.path.join(HERE, 'RTH4612_binary_anchors.csv')


def sub_count(task):
    prefix12, ext4 = task
    prefix16 = tuple(prefix12) + tuple(ext4)
    t0 = time.time()
    cnt, nf, n_cls, N, dt = count_param_dfs(4, 3, vals=(0, 1),
                                            block=(16, prefix16))
    return (prefix12, ext4, cnt, time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--Lx', type=int, default=4)
    ap.add_argument('--Ly', type=int, default=3)
    ap.add_argument('--depth', type=int, default=16, help='extended prefix bits')
    ap.add_argument('--workers', type=int, default=70)
    args = ap.parse_args()
    Lx, Ly, D = args.Lx, args.Ly, args.depth
    parent_bits = 12
    assert D > parent_bits
    done = set()
    rows = {}
    with open(BLOCKS) as f:
        for r in csv.DictReader(f):
            key = tuple(int(x) for x in r['block'].split('_'))
            rows[key] = (int(r['Omega_block']), float(r['seconds']))
            done.add(key)
    missing = [blk for blk in itertools.product((0, 1), repeat=parent_bits)
               if blk not in done]
    print(f'{Lx}x{Ly}: {len(done)} done, {len(missing)} slow parents '
          f'to split at depth {D}', flush=True)
    tasks = [(m, ext) for m in missing
             for ext in itertools.product((0, 1), repeat=D - parent_bits)]
    t0 = time.time()
    sub_sums = {m: 0 for m in missing}
    sub_max = {m: 0.0 for m in missing}
    with Pool(args.workers) as pool:
        for k, (m, ext, cnt, dt) in enumerate(
                pool.imap_unordered(sub_count, tasks)):
            sub_sums[m] += cnt
            sub_max[m] = max(sub_max[m], dt)
            if (k + 1) % 128 == 0:
                print(f'  [{k + 1}/{len(tasks)}] elapsed '
                      f'{time.time() - t0:.0f}s', flush=True)
    # write missing rows back
    with open(BLOCKS, 'a', newline='') as f:
        w = csv.writer(f)
        for m in missing:
            w.writerow(['_'.join(map(str, m)), sub_sums[m],
                        round(sub_max[m], 1)])
    total = sum(v[0] for v in rows.values()) + sum(sub_sums.values())
    N = Lx * Ly
    omega_total = total * 2
    s01 = math.log(omega_total) / (6 * N)
    print(f'{Lx}x{Ly} BINARY summary: Omega_01={omega_total}  '
          f'ln={math.log(omega_total):.4f}  s_01={s01:.4f}', flush=True)
    KNUTH = {3: 6.566e12, 4: 8.351e19}
    if Lx == Ly and Lx in KNUTH:
        ok = omega_total <= KNUTH[Lx] * 1.01
        print(f'    monotonicity vs ternary Knuth est: '
              f'{"OK" if ok else "VIOLATED"}', flush=True)
        assert ok
    with open(BLOCKS.replace('_blocks.csv', '_summary.csv'), 'w') as f:
        f.write(f'L={Lx}x{Ly} Omega_01={omega_total} s01={s01:.6f}\n')
    with open(ANCHOR_CSV, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([Lx, '{0,1}', f'blocked-dfs-{Lx}x{Ly}', omega_total,
                    f'{math.log(omega_total):.4f}', f'{s01:.6f}',
                    '', '', N, ''])
    print('anchor CSV updated', flush=True)


if __name__ == '__main__':
    main()
