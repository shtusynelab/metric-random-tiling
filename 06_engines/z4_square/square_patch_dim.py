# -*- coding: utf-8 -*-
"""square_patch_dim.py -- kernel dimension sequence of square patches
z=4 square lattice: isogonal rectangle closure = opposite sides equal
(propagating type); kernel dimension = W+L (exactly linear numerically).
Same O(perimeter) scaling as the kagome patch 2(W+L) => the z=4 family is
confirmed sub-extensive in R on both members."""
import numpy as np

def square_patch_dim(W, L):
    hid = lambda i, j: i * (L + 1) + j
    vid = lambda i, j: W * (L + 1) + i * L + j
    n = W * (L + 1) + (W + 1) * L
    rows = []
    for i in range(W):
        for j in range(L):
            r = [0.0] * n
            r[hid(i, j)] += 1.0; r[hid(i, j + 1)] -= 1.0
            rows.append(r)
            r = [0.0] * n
            r[vid(i, j)] += 1.0; r[vid(i + 1, j)] -= 1.0
            rows.append(r)
    M = np.array(rows)
    return M.shape[1] - np.linalg.matrix_rank(M)

if __name__ == "__main__":
    print(f"{'W,L':>8} {'n_vars':>7} {'dim_R':>6} {'W+L':>5}")
    for (W, L) in [(4, 4), (6, 6), (8, 8), (12, 12)]:
        d = square_patch_dim(W, L)
        print(f"{W}x{L:>5} {2*W*L+W+L:>7} {d:>6} {W+L:>5}")
