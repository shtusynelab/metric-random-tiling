# -*- coding: utf-8 -*-
"""Step 10 (binary honeycomb phase-resolved spectral refinement): eigenvalue
structure decomposition of T and finalization of the s convention.

Background: the binary honeycomb {0,1} is the same system as the 3.12.12
binary mid band; its perimeter-law closure Omega = 2^{Lx+Ly+1} is a
transient-sector reading. The step8 power iteration fails to converge at
even W (tail_lo != tail_hi). This script uses the full/peripheral spectrum
to answer three questions and finalize the s values:
  (a) the true spectral radius rho_true(W) (even-W correction) and s fits in
      two conventions (Perron convention s_gen = ln rho / W with the
      s_inf + c/W form; limsup convention);
  (b) the degeneracy structure of the largest-modulus eigenvalues ("+/- pairs"
      / complex-conjugate families -- the source of the even-W power-iteration
      oscillation);
  (c) spectral anatomy of the exact 2-legion sub-law sector: the multiplicity
      m_2(W) of eigenvalues exactly equal to 2, and the spectral mechanism of
      "exactness within a window" (the power-sum identity
      sum_{lam != 2} lam^L = 0 for L <= L*(W), verified directly from the
      dense spectrum).

Method: Ly <= 6 dense full spectrum (numpy eigvals, n = 16..4096); Ly = 7,8
scipy eigs peripheral spectrum with k = 80. Cross gates:
  - rho_true >= power-iteration tail_hi (measured in step8) and
    <= tail_hi * (1 + 1e-6);
  - dense spectrum vs exact integer DP Tr(T^L) point-by-point (L = 1..8,
    Ly <= 5);
  - Tr growth rate lim_{L->inf} ln Tr(T^L)/L -> ln rho_true (Ly <= 5, DP up
    to L = 200).
Memory: dense peak Ly=6 ~0.5 GB (4096^2 complex128); eigs < 200 MB; fits
locally.
Output: TSDHC_binary_spectral.csv + stdout fit summary.
"""
import csv
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from P0_PM_hc_trace import trace_power  # noqa: E402
from step8_hc_tm import build_states_pool, build_sparse, perron  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'TSDHC_binary_spectral.csv')
ALPHA = (0, 1)
K_EIGS = 80
# dense full-spectrum cap by alphabet: n <= ~1e4 (binary full space 4^Ly,
# nonarith 9^Ly)
DENSE_MAX_LY = {0: 6, 1: 4}          # keyed by alpha[0]: binary -> Ly6 (4096), nonarith -> Ly4 (6561)
K_EIGS = 80
# step8 measured power-iteration tails (when unconverged, tail_hi is a lower
# bound on rho)
POWER_TAILS = {2: (2.4, 2.5), 3: (2.62892111, 2.62892111),
               4: (3.35925350, 3.57222222), 5: (3.76037045, 3.76037045),
               6: (4.83144140, 4.99675142), 7: (5.67545949, 5.67545949),
               8: (7.31338544, 7.40191651)}


def dense_spectrum(n, F, Tt, C):
    D = np.zeros((n, n), dtype=np.float64)
    np.add.at(D, (F, Tt), C)
    ev = np.linalg.eigvals(D)
    return ev


def eigs_spectrum(n, F, Tt, C, k=K_EIGS):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    A = sp.csr_matrix((C, (F, Tt)), shape=(n, n))
    ev, _ = spl.eigs(A, k=k, which='LM', maxiter=100000, tol=1e-12)
    return ev


def spec_summary(ev, tag, print_all=False):
    mods = np.abs(ev)
    order = np.argsort(-mods)
    rho = mods[order[0]]
    near = int(np.sum(mods > rho * (1 - 1e-8)))
    m2p = int(np.sum(np.abs(ev.real - 2.0) < 1e-8)) if len(ev) <= 4096 else -1
    m2m = int(np.sum(np.abs(ev.real + 2.0) < 1e-8)) if len(ev) <= 4096 else -1
    head = [(ev[i].real, ev[i].imag) for i in order[:12]]
    print(f'{tag}: rho={rho:.8f} max_mod_degeneracy={near} m_2(+2)={m2p} '
          f'm_2(-2)={m2m}', flush=True)
    if print_all:
        print('  full spectrum:', flush=True)
        for i in order:
            re, im = ev[i].real, ev[i].imag
            print(f'    lam = {re:+.6f} {im:+.6f}i  |lam|={math.hypot(re, im):.6f}',
                  flush=True)
    else:
        for re, im in head:
            print(f'    lam = {re:+.6f} {im:+.6f}i  |lam|={math.hypot(re, im):.6f}',
                  flush=True)
    return rho, near, (m2p, m2m), head


def bucket_decomposition(ev, Ly, Lmax):
    """Tr(T^L) decomposition bucketed by |lam|: spectral language of the
    crossover mechanism. Buckets: lam = +2 (sub-law legion), lam = -2,
    |lam| > 2 + eps (generic phase), the rest (sub-leading)."""
    lam2p = [l for l in ev if abs(l.real - 2.0) < 1e-8 and abs(l.imag) < 1e-8]
    lam2m = [l for l in ev if abs(l.real + 2.0) < 1e-8 and abs(l.imag) < 1e-8]
    gen = [l for l in ev if abs(l) > 2.0 + 1e-8]
    print(f'  trace decomposition Ly={Ly} (L=1..{Lmax}): gen_phase '
          f'n={len(gen)}, +2 legion n={len(lam2p)}, -2 n={len(lam2m)}',
          flush=True)
    print(f'  {"L":>3} {"Tr_total":>14} {"gen_phase":>14} {"legion2(+2)":>14} '
          f'{"ratio gen/legion2":>14}', flush=True)
    for L in range(1, Lmax + 1):
        t_gen = float(np.sum(np.array(gen) ** L).real) if gen else 0.0
        t_2 = float(sum(l ** L for l in lam2p)) if lam2p else 0.0
        ratio = (t_gen / t_2) if abs(t_2) > 1e-9 else float('inf')
        print(f'  {L:>3} {"":>14} {t_gen:>14.4f} {t_2:>14.4f} {ratio:>14.4g}',
              flush=True)


def tr_crosscheck(t0graph, ev, Ly, Lmax=8):
    """Dense full spectrum vs exact integer Tr(T^L), point-by-point
    (L = 1..Lmax)."""
    worst = 0.0
    for L in range(1, Lmax + 1):
        tr_exact = trace_power(t0graph, L)
        tr_spec = float(np.sum(ev ** L).real)
        rel = abs(tr_spec - tr_exact) / max(1.0, abs(tr_exact))
        worst = max(worst, rel)
        assert rel < 1e-6, f'Tr mismatch Ly={Ly} L={L}: {tr_spec} vs {tr_exact}'
    print(f'  gate Tr cross-check Ly={Ly} L=1..{Lmax}: worst rel={worst:.2e} OK',
          flush=True)


def tr_growth(t0graph, rho, Ly, Lmax):
    """Growth-rate convergence: ln Tr(T^L)/L -> ln rho (monotone approach in
    the final segment; last 3 points reported)."""
    rates = []
    for L in range(Lmax - 5, Lmax + 1, 2):
        tr = trace_power(t0graph, L)
        rates.append((L, math.log(tr) / L))
    for L, r in rates:
        print(f'  growth Ly={Ly} L={L}: lnTr/L={r:.6f} (ln rho='
              f'{math.log(rho):.6f}, gap={r - math.log(rho):+.5f})',
              flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    alpha = tuple(int(c) for c in sys.argv[1]) if len(sys.argv) > 1 else ALPHA
    maxLy = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tag = 'binary' if alpha == (0, 1) else 'nonarith'
    out_csv = OUT_CSV if tag == 'binary' else \
        os.path.join(HERE, 'TSDHC_nonarith_spectral.csv')
    workers = int(os.environ.get('TM_WORKERS', 0)) or \
        min(64 if os.environ.get('ON_SERVER') else 6, os.cpu_count() or 8)
    exists = os.path.exists(out_csv)
    f = open(out_csv, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not exists:
        w.writerow(['Ly', 'n_states', 'method', 'rho_true', 'ln_rho',
                    'mult_rhomod', 'm2_exact2', 'head_spectrum', 'time_s'])
    for Ly in range(2, maxLy + 1):
        t0 = time.time()
        ctx, t0graph, n = build_states_pool(alpha, Ly, max(4, Ly), workers)
        F, Tt, C = build_sparse(t0graph)
        method = 'dense' if Ly <= DENSE_MAX_LY[alpha[0]] else 'eigs'
        if method == 'dense':
            ev = dense_spectrum(n, F, Tt, C)
        else:
            ev = eigs_spectrum(n, F, Tt, C)
        rho, near, (m2p, m2m), head = spec_summary(
            ev, f'{tag} Ly={Ly}', print_all=(Ly <= 4 and len(alpha) == 2))
        plo, phi = POWER_TAILS.get(Ly, (0.0, 0.0)) if tag == 'binary' \
            else (rho * 0.999, rho)
        # the power-iteration s sequence is a Rayleigh-quotient-type quantity
        # in [rho, max row sum]; rho only needs to be no smaller than the
        # tail minimum (+/- pair oscillation can push tail_hi above rho, so
        # tail_hi is not used as a lower bound; nonarith has no archived
        # tail, so that gate is skipped)
        if tag == 'binary':
            assert rho >= plo * (1 - 1e-6), \
                f'rho_true < power tail_lo: {rho} < {plo}'
        if method == 'dense':
            tr_crosscheck(t0graph, ev, Ly)
            if Ly <= 5:
                tr_growth(t0graph, rho, Ly, 200)
            if Ly <= 6:
                bucket_decomposition(ev, Ly, Lmax={2: 10, 3: 16, 4: 28,
                                                   5: 28, 6: 24}[Ly])
        w.writerow([Ly, n, method, f'{rho:.10f}', f'{math.log(rho):.8f}',
                    near, f'{m2p}/{m2m}',
                    ' '.join(f'{re:.6f}{im:+.6f}i' for re, im in head),
                    f'{time.time()-t0:.0f}'])
        f.flush()
    f.close()
    print('done', flush=True)


if __name__ == '__main__':
    main()
