# z3_488/ -- 4.8.8 counting engine

Counts edge-length assignments on the 4.8.8 network (octagons + squares,
number field Q(sqrt(2))). Delivered results: the real-alphabet solution space
is extensive (dim_R = F + 2), while integer alphabets have zero entropy -- 
irrational-coefficient rows split the solution into rigid strips
(Omega = 3^(4W) form). The Lambda library Lambda = Z + (sqrt(2)/2) Z revives
the dimension (dim_Lambda = 2 dim_R), but its counting entropy stays
sub-extensive: `TSQ_Lambda_DP_exp.csv` shows ln Omega / F decreasing with
window size across the delivered width series. Determination record:
`TSQ_Lambda_positive_entropy_adjudication.md`.

## Key files

- `TSQ_zero_entropy_theorem_analytic_proof.md` -- theorem and proof (incl.
  English SI draft); parent document of the row-splitting mechanism used by
  the same family member 3.12.12 (`06_engines/z3_31212/`).
- `tsq_lib.py` / `tsq_lambda_lib.py` -- model libraries (torus construction,
  constraint matrix, exact linear algebra, kernel enumeration, strip TM).
- `TSQ_numerical_verification.py` -- numerical verification suite;
  `TSQ_entropy_density_verification.csv` -- delivered dimension table.
- `TSQ_Lambda_DP_exp.csv` -- delivered Lambda-library width series (columns
  W, F, Omega, ln Omega, ln Omega / F); `TSQ_Lambda_DP.csv`,
  `TSQ_Lambda_SNF_series.csv`, `TSQ_Lambda_Knuth_W2_crosscheck_trends.csv` -- 
  supporting series.
- `tsq_lambda/` -- transfer-matrix machinery for the Lambda counts
  (`T_builder_v2.py`, exact trace counters, Hutchinson cross-checks,
  reduced-component and branch-bound utilities).
- `roundE_488_knuth.py` -- Knuth estimator cross-check.
- Vendored demo: `tsq_488_real_demo.py` rendering
  `TSQ_real_metric_random_solution_demo.png/.pdf` (real-metric random
  solution sampling on the 4.8.8 network).
