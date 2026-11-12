# -*- coding: utf-8 -*-
"""Face-order probing for the 4.6.12 binary frontier DP.

Question: is the L=3 peak state count (~1e8 under the BFS wavefront
order) an artifact of the ORDER rather than of the problem? A
column-like order keeps the frontier at O(W) classes; if a cheap order
drops the peak by 2-3 orders of magnitude, torus DP at L=4/5 revives
and the column TM state space shrinks accordingly.

Orders probed (all cheap, no geometric lifting):
  natural      : face id order (half-edge discovery order)
  natural_rev  : reversed face id order
  bfsK         : edge-sharing face graph BFS from face K (K = 0,1,2)
  bfs0_rev     : BFS order reversed
Each order runs with probe_limit = 2e6: abort reports the magnitude;
a full finish reports the exact DP count as a bonus (order must not
change the count -- any two completed orders crosscheck each other).
L = 2 and 3, binary values, rank_check skipped (probing only).
"""
import time

from roundE_4612_frontier_dp import build_face_system, face_bfs_order, dp_count

PROBE_LIMIT = 2 * 10 ** 6


def main():
    for L in (2, 3):
        (face_eqs, n_cls, n_free, N, dim, all_eq,
         face_nbrs) = build_face_system(L, L, rank_check=False)
        n_faces = len(face_eqs)
        natural = list(range(n_faces))
        bfs0 = face_bfs_order(face_nbrs, n_faces, 0)
        orders = {
            'natural': natural,
            'natural_rev': natural[::-1],
            'bfs0': bfs0,
            'bfs0_rev': bfs0[::-1],
            'bfs1': face_bfs_order(face_nbrs, n_faces, 1 % n_faces),
            'bfs2': face_bfs_order(face_nbrs, n_faces, 2 % n_faces),
        }
        counts = {}
        for name, order in orders.items():
            t0 = time.time()
            dp, nf, N, tw, peak, dt = dp_count(
                L, L, vals=(0, 1), face_order=order,
                probe_limit=PROBE_LIMIT, rank_check=False)
            tag = 'ABORT' if dp is None else f'DP={dp}'
            if dp is not None:
                if name in counts and counts[name] != dp:
                    print(f'    ORDER MISMATCH {name}: {dp} vs '
                          f'{counts[name]}', flush=True)
                counts[name] = dp
            print(f'L={L} order={name:12s} tree_w={tw:3d} peak={peak:>12d} '
                  f'{tag} [{dt:.0f}s]', flush=True)
    print('PROBE DONE', flush=True)


if __name__ == '__main__':
    main()
