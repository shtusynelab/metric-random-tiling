# -*- coding: utf-8 -*-
"""Server-only driver (originally run on the internal cluster); pass explicit
arguments when used locally.

Single-point ternary Knuth estimation with sample blocks distributed over a Pool.

usage: python3 step4b_knuth_server.py W L n_workers total_samples seed0
Prints one CSV line: W,L,nsamp,hits,nf,n_free,E[X],Omega_est,s_face,time_s
Calibration anchors (vs exact DP): W=2 L=3 -> 68139; W=3 L=4 -> 7.12544e6; etc.
"""
import math
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step4_knuth_ternary import prepare_joint, knuth_ternary  # noqa: E402


def work(args):
    W, L, ns, seed = args
    nf, dep_list, p_of_pos, order, coef_of, n_free, n_cls = prepare_joint(W, L)
    t0 = time.time()
    mean, hits = knuth_ternary(nf, dep_list, p_of_pos, order, coef_of, ns, seed)
    return (mean * ns, hits, ns, time.time() - t0)


def main():
    W, L, nw, ns_total, seed0 = (int(x) for x in sys.argv[1:6])
    t0 = time.time()
    chunks = [(W, L, ns_total // nw + (1 if i < ns_total % nw else 0), seed0 + i)
              for i in range(nw)]
    with Pool(nw) as pool:
        res = pool.map(work, chunks)
    total_X = sum(r[0] for r in res)
    hits = sum(r[1] for r in res)
    ns = sum(r[2] for r in res)
    nf, dep_list, p_of_pos, order, coef_of, n_free, n_cls = prepare_joint(W, L)
    ex = total_X / ns
    om = ex * 3 * (3 ** n_free)
    s_face = math.log(om) / (3 * W * L)
    dt = time.time() - t0
    print(f'{W},{L},{ns},{hits},{nf},{n_free},{ex:.8g},{om:.8g},{s_face:.6f},{dt:.0f}',
          flush=True)


if __name__ == '__main__':
    main()
