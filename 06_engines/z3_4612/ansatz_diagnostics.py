# -*- coding: utf-8 -*-
"""4.6.12 ansatz diagnostics for the ternary Knuth four-point series
(question: are there power-law terms beyond 1/L?).

Data (full-orbit calibre):
  ternary {0,1,2}: RTH4612_knuth_estimate_series.csv
    L=1: 2349 / L=2: 29 723 517 / L=3: 6.56623e12 / L=4: 8.351e19
  binary {0,1}: registered  112 / 22 912 / 11 261 320 (L=1..3)
  binary W=2 strips: lambda(2) ~ 2.5 per column -> 2.5/12 ~ 0.208 per hole.
"""
import numpy as np

Om3 = [2349.0, 29723517.0, 6.56623e12, 8.351e19]
s3 = np.array([np.log(o) / (6 * L * L) for L, o in enumerate(Om3, 1)])
Om2 = [112.0, 22912.0, 11261320.0]
s2 = np.array([np.log(o) / (6 * L * L) for L, o in enumerate(Om2, 1)])


def fit_1overL(L, s, mask):
    x = 1.0 / L[mask]
    a, b = np.polyfit(x, s[mask], 1)
    return b, a                      # s_inf, coef of 1/L


def a_local(s):
    """Effective 1/L coefficient: Delta(L)*L*(L+1); constant for a pure
    1/L law (independent of s_inf)."""
    d = -np.diff(s)
    L = np.arange(1, len(s))
    return d * L * (L + 1), d, L


def fit_ab_differences(al, L):
    """Difference-structure test: a_local(L) = a + b*(2L+1)/(L*(L+1))
    (difference image of s = s_inf + a/L + b/L^2).
    3 differences vs 2 parameters = 1 dof."""
    u = (2 * L + 1.0) / (L * (L + 1.0))
    b, a = np.polyfit(u, al, 1)
    resid = al - (a + b * u)
    return a, b, resid


def solve_exact_3pt(Ls3, s3v, form):
    """Exact 3-parameter solve of an ansatz on 3 points (zero dof).
    form='1/L+1/L^2': linear system; form='beta': s = s_inf + a/L^beta
    with a free exponent."""
    if form == '1/L+1/L^2':
        A = np.vstack([np.ones(3), 1.0 / Ls3, 1.0 / Ls3**2]).T
        return np.linalg.solve(A, s3v)
    from scipy.optimize import brentq

    def g(beta, r):
        p = Ls3 ** (-beta)
        return (p[0] - p[1]) / (p[1] - p[2]) - r
    r = (s3v[0] - s3v[1]) / (s3v[1] - s3v[2])
    beta = brentq(g, 0.2, 4.0, args=(r,))
    p = Ls3 ** (-beta)
    a = (s3v[0] - s3v[1]) / (p[0] - p[1])
    return np.array([s3v[0] - a * p[0], a, beta])


L = np.arange(1, 5, dtype=float)

print('=== ternary {0,1,2} four points ===')
print('s =', np.round(s3, 6))
al, d, Ld = a_local(s3)
print('Delta(L)      =', np.round(d, 6))
print('a_local=DL(L+1) =', np.round(al, 4), '  (constant for pure 1/L)')
print('difference ratio D(L)/D(L+1) =', np.round(d[:-1] / d[1:], 3), ' (pure 1/L gives 3, 2)')

for name, mask in (('L=1-3', L <= 3), ('L=2-4', L >= 2), ('L=1-4', np.ones(4, bool))):
    si, a = fit_1overL(L, s3, mask)
    pred = si + a / L[mask]
    print(f'1/L fit {name}: s_inf={si:.4f} a={a:.4f} max|resid|={np.abs(s3[mask]-pred).max():.4f}')

a, b, resid = fit_ab_differences(al, Ld)
print(f'difference-structure test s=s_inf+a/L+b/L^2 (1 dof): a={a:.3f} b={b:.3f} '
      f'resids={np.round(resid,4)} RMS={np.sqrt((resid**2).mean()):.4f}  (data error ~1e-4)')

sol24 = solve_exact_3pt(L[1:], s3[1:], '1/L+1/L^2')
sinf, a, b = sol24
print(f'exact fit L=2-4, s_inf+a/L+b/L^2 (0 dof): s_inf={sinf:.4f} a={a:.4f} b={b:.4f}'
      f'  ->  s(5) pred={sinf+a/5+b/25:.4f}')
sol24b = solve_exact_3pt(L[1:], s3[1:], 'beta')
sinf, a, beta = sol24b
print(f'exact fit L=2-4, s_inf+a/L^beta   (0 dof): s_inf={sinf:.4f} a={a:.4f} beta={beta:.3f}'
      f'  ->  s(5) pred={sinf+a/5**beta:.4f}')
sol13 = solve_exact_3pt(L[:3], s3[:3], 'beta')
sinf, a, beta = sol13
print(f'exact fit L=1-3, s_inf+a/L^beta   (0 dof): s_inf={sinf:.4f} beta={beta:.3f}'
      f'  ->  s(4) pred={sinf+a/4**beta:.4f}  (measured 0.4778, offset {0.4778-sinf-a/4**beta:+.4f})')

print()
print('=== binary {0,1} channel (registered values) ===')
L2 = np.arange(1, 4, dtype=float)
print('s_01 =', np.round(s2, 6))
al2, d2, Ld2 = a_local(s2)
print('a_local =', np.round(al2, 4), ' (looks like clean 1/L)')
si, a = fit_1overL(L2, s2, np.ones(3, bool))
print(f'1/L fit L=1-3: s_inf={si:.4f}')
print(f'W=2 strip lower bound lambda(2)/12 ~ {2.5/12:.3f}  ->  bounds the 1/L '
      f'extrapolation from an independent channel (gap ~{2.5/12-si:.2f})')
