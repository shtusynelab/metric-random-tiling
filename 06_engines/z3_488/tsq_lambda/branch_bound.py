# -*- coding: utf-8 -*-
"""Branch-count upper bound for the row transfer matrix (step 1 of gate N1).

Row transfer T: state = (upper-row cut-value segment up, current H-column
values hv) (dimensions 3^{2W} x 3^W).
Transition: enumerate the new row segment nr (3^{2W}); the weight is
w = number of legal V-row-walk starting points (0 if the (P) parity
condition fails or the V walk hits a dead branch); the transition is alive
iff the H update is legal (each column's new value in LIB, wrap-around step
hst).
lambda_max(T) <= max_out ~= E_states[sup of the outgoing weight sum]
(nonnegative matrix, Perron-Frobenius: lambda_max <= max row sum; the mean
row sum over random states gives a typical upper bound).
s_asymptotic = ln lambda_max / (2W) <= ln(row-sum bound) / (2W).

Output: branch-sum distributions (Monte Carlo) and s upper bounds for
W = 2,3,4, compared against the measured s(W).
"""
import sys, random, math
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from itertools import product
from tsq_lambda_lib import M_EXP

LIB = [(1, 0), (0, 1), (1, 1)]
rng = random.Random(424242)


def row_transfer_branch(up, hv, W):
    """Given the upper-row cut-value segment up ([(alpha,beta)] x W) and the
    H values hv (x W), enumerate all new row segments nr and return
    (weight sum, number of alive transitions)."""
    tot = 0
    alive = 0
    for nr in product(LIB, repeat=W):     # new row segment = [(alpha,beta) x W]
        # per cell (i): up[i] = (alpha_u, beta_u), nr[i] = (alpha_n, beta_n);
        # c1 = n_alpha_u[i] - n_alpha_n[i+1], c2 = m_alpha_u[i] - m_alpha_n[i+1];
        # per the (D) chain indices, c3 is defined on octagon (i,j):
        # n_beta(i+1,j) - n_beta(i,j+1) = n_beta_nr(i+1) - n_beta_up(i),
        # i.e. the beta-chain segment connects new-row i+1 with upper-row i.
        w_ok = True
        hstep = []
        vst = []
        for i in range(W):
            i1 = (i + 1) % W
            c1 = up[i][1] - nr[i1][1]
            c2 = up[i][0] - nr[i1][0]
            c3 = nr[i1][1] - up[i][1]
            c4 = nr[i1][0] - up[i][0]
            if (c1 + c3) % 2:
                w_ok = False
                break
            hstep.append(((c1 + c3) // 2, c2 + c4))
            vst.append(((c1 - c3) // 2, c2 - c4))
        if not w_ok:
            continue
        # H update legal: hv[i] + hstep[i] in LIB columnwise (wrap-around
        # step, lands directly on library values)
        hnew_ok = all((hv[i][0] + hstep[i][0], hv[i][1] + hstep[i][1]) in
                      {(1, 0), (0, 1), (1, 1)} for i in range(W))
        if not hnew_ok:
            continue
        # V row walk: free starting point, wrap-around closure, all
        # intermediate states in LIB
        if sum(s[0] for s in vst) != 0 or sum(s[1] for s in vst) != 0:
            continue
        cnt = 0
        for start in LIB:
            cur = start
            ok = True
            for d in vst:
                cur = (cur[0] + d[0], cur[1] + d[1])
                if cur not in {(1, 0), (0, 1), (1, 1)}:
                    ok = False
                    break
            if ok:
                cnt += 1
        tot += cnt
        if cnt:
            alive += 1
    return tot, alive


# exact row-sum distribution (W=2: all states 3^6 x 3^2 = 6561 x transitions 3^4 - enumerable)
for W, n_sample in ((2, None), (3, 20000), (4, 20000)):
    dims = (3 ** (2 * W), 3 ** W)
    if n_sample is None:
        states = [(up, hv) for up in product(LIB, repeat=W)
                  for hv in product(LIB, repeat=W)]
        sums = [row_transfer_branch(up, hv, W)[0] for up, hv in states]
        tag = "exact all states"
    else:
        sums = []
        states = None
        for _ in range(n_sample):
            up = tuple(LIB[rng.randrange(3)] for _ in range(W))
            hv = tuple(LIB[rng.randrange(3)] for _ in range(W))
            sums.append(row_transfer_branch(up, hv, W)[0])
        tag = f"sampled {n_sample}"
    mx = max(sums)
    mean = sum(sums) / len(sums)
    sb = math.log(mx) / (2 * W) if mx > 0 else float("-inf")
    sm = math.log(mean) / (2 * W) if mean > 0 else float("-inf")
    print(f"W={W} ({tag}): state count {dims[0]*dims[1]}, row sum max={mx} "
          f"mean={mean:.2f}, s_bound(max)={sb:.4f}, s_bound(mean)={sm:.4f}",
          flush=True)
print("reference measured values: s(2)=1.1001, s(3)=0.7357; "
      "if ln(row-sum bound)/(2W) decreases toward 0 with increasing W, "
      "that rules out the asymptotically positive branch")
