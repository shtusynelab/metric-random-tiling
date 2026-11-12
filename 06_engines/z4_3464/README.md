# z4_3464/ -- 3.4.6.4 (rhombitrihexagonal) engine

## Final conclusions

3.4.6.4 is the second positive-entropy carrier of the paper. Three results,
all delivered in this directory:

1. **Triangle locking (theorem).** In every solution -- real or integer -- all
   triangle edges take one global value: face closure forces the three edges
   of each triangle equal, square pairing propagates the equality, and the
   triangle-adjacency graph (the honeycomb graph of hexagon centers) is
   connected. Half of all edges therefore collapse onto a single coordinate.
   Proof and numerical acceptance: `RTH_triangle_locking_theorem_analytic_proof.md`,
   regression part of `RTH_numerical_verification.py`.

2. **Exact sparse-trace area law.** Torus counts Omega_tor(L x L) = Tr(T^L)
   on the single-column transfer operator (exact integers, sparse CSR),
   L = 2..10: Omega = 128 / 1,036 / 8,800 / 86,956 / 1,114,700 / 23,325,388 /
   796,865,888 / 44,571,806,560 / 4,002,790,772,868, i.e. ln Omega =
   4.852 / 6.943 / 9.083 / 11.373 / 13.924 / 16.965 / 20.496 / 24.520 / 29.018.
   Delivered in `RTH_torus_area_law_8pts.csv`; counter
   `rth_torus_trace_sparse.py` (Hutchinson cross-checks in `hutch_rth.py`).

3. **Positive metric entropy.** The per-face entropy s = ln Omega / (6 L^2)
   is still rising at L = 10 (s*L = 0.404 -> 0.484 over L = 2..10), and the
   asymptotic fit pins the leak at s_infinity ~ 0.040 per face. The
   perimeter-law reading (s = 0, Omega ~ k^(3.3 L)) is falsified: the
   three-parameter fit ln Omega = a*L + b*L^2 + c (a = 1.109, b = 0.1456,
   c = 2.181, RMSE 0.121, crossover L* = sqrt(c/b) ~ 3.9 at the observed
   s*L minimum) predicted ln Omega(9) = 24.52 and ln Omega(10) = 29.03 in
   advance, and both landed (24.520 / 29.018).

Constraint structure (exact, `RTH_entropy_density_verification.csv`): the
torus L x L has F = 6 L^2 faces (L^2 hexagons) and dim_R = dim_Q = L^2 + 3 =
F_hex + 3 exactly -- L = 2..6 gives 7 / 12 / 19 / 28 / 39 (V = 24/54/96/150/216,
E = 48/108/192/300/432). Exact k = 2 counts Omega(L) = 128 / 1,036 / 8,800 /
86,956 for L = 2..5 (s = 0.1286 / 0.0946 per face at L = 3 / 4). Alphabet
translation invariance holds (alpha0 = 0 and alpha0 = 1 give identical
counts). Entropy upper bound: s <= (L^2 + 3) ln k / (6 L^2) -> ln k / 6 per
face. All faces lie in {3, 4, 6} (Niven criterion), so the constraint system
is rationally equivalent and no row splitting occurs -- the entropy survives
through combinatorial thickness of the integer solution lattice, not through
arithmetic splitting. Layer decomposition: any finite real library A (rational
rank r) is bounded by integer counts on the same lattice, Omega(A) <= the
product over layers of the integer counts on the restricted value sets; measured, {0, 1, sqrt(2)} gives Omega =
2,187 / 59,121 (L = 2/3), within the |S|^2 bound (14.00 / 20.03); AP
{0, 1, 2}: Omega(L = 3) = 68,301 (log_3 = 10.13 vs log_2(k = 2) = 10.02) and
Omega(L = 4, k = 3) = 3,837,303 (log_3 = 13.80 vs 13.10 -- bits per value
alpha(L) approximately k-independent to a few percent). Greedy disjoint
supported-subbox certificates stay linear in both carriers (honeycomb
|S| = 2L + 2 = 6/8/10/12/14 for L = 2..6; 3.4.6.4 |S| = 2L + 3 = 7/9/11/13
for L = 2..5, k = 2/3 alike): the certificate route cannot separate the two
carriers; the area law above is the separation.

Also regression-hosted here (rows of `RTH_entropy_density_verification.csv`):
(3,3,3,4,4) with dim = 2L + 1 sub-extensive (5 / 7 / 9 / 11 for L = 2..5),
3.12.12 and 4.8.8 with dim_R = F + 2 and dim_Q linear (4L, 11/19/29),
kagome anchors dim = Lq + Lr + gcd(Lq, Lr) - 1, triangle-locking projection
rank = 1 at L = 2/3, and snub hexagonal (3,3,3,3,6) rigidity dim_R = dim_Q = 1
for k = 1..5 (`snub_lib.py`).

The historical verification sequence that led to these conclusions is
preserved in the archived adjudication records; the deliverables above
supersede any intermediate readings in this file.

## Files

- `RTH_numerical_verification.py` -- all verifications (anchors, counterexample
  construction, counting and certificates, triangle locking, snub rigidity);
  32 assertion groups.
- `RTH_entropy_density_verification.csv` -- delivered dimension / count table.
- `RTH_torus_area_law_8pts.csv` -- delivered L = 2..10 torus trace series plus
  the registered pre-measurement predictions for L = 10.
- `RTH_triangle_locking_theorem_analytic_proof.md` -- triangle locking theorem:
  proof, numerics, English SI draft.
- `rth_torus_trace_sparse.py` -- sparse CSR trace counter (Tr(T^L), adaptive
  blocking).
- `hutch_rth.py` -- Hutchinson estimator cross-checks.
- `rth_cyl_tm.py` / `rth_strip_tm.py` -- cylinder / strip transfer-matrix
  counters.
- `rth_k3_L4.py`, `diag_rth_binary.py`, `rth_zint_demo.py`,
  `thick_crit_numerics.py` -- diagnostics and demos (integer-alphabet random
  tiling, thinness diagnostics).
- `snub_lib.py` -- snub hexagonal torus construction and nullity counts.
- `RTH_area_law_9pts_fit.png/.pdf` -- ln Omega nine-point fit figure with the
  s*L curve (out-of-sample hits marked).
- `RTH_integer_alphabet_random_tiling_demo.png/.pdf` -- integer-alphabet
  random tiling demo.
