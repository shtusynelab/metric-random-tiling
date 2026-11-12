# z3_31212/ -- 3.12.12 (truncated honeycomb) archive

3.12.12 is one of the eleven Archimedean uniform vertex types; second member
of the z = 3 mixed-face family (4.8.8 / 3.12.12 / 4.6.12).

## System

Vertex type (3,12,12): faces = equilateral triangles + regular dodecagons in
ratio 2:1 (torus L x L: F = 3 L^2, with 2 L^2 triangles and L^2 dodecagons).
Direction system = the 12 half-steps of 30 degrees, number field Q(sqrt(3)).
Combinatorially the standard truncation of the honeycomb.

## Conclusions

| Layer | Result | Status |
|---|---|---|
| R (real edge lengths) | dim_R = F + 2 = 3 L^2 + 2 (extensive) | [T]+[N] |
| Z (integer alphabets) | sqrt(3)/2 row splitting: dim_Q = 11 / 19 / 29 (L = 2, 3, 4; differences 8 / 10, linear) -> sub-extensive -> s(infinity) = 0 | [N] |
| Mechanism | row splitting (same family as 4.8.8: irrational coefficient sqrt(2)/2 -> sqrt(3)/2) | -- |

Second clean sample (after 4.8.8) where the R and Z layers disagree: the
continuous degrees of freedom live on irrational directions, and integer
alphabets cannot reach them.

## Files

- `TSD_31212split_row_rigidity_theorem.md` -- theorem statement and mechanism
  proof skeleton.
- `TSQ31212_real_metric_random_solution_demo.png/.pdf` -- metric random
  solution demo (R-layer solution-space sampling; closure residual ~1e-16;
  demo rendering script not distributed with this release).
- Numerical verification (housed in `z4_3464/`):
  `z4_3464/RTH_numerical_verification.py`, part 6 (`build_31212`: asserts
  dim_R = 3 L^2 + 2, dim_Q linearity, vertex-type census).
- Mechanism proof parent document:
  `z3_488/TSQ_zero_entropy_theorem_analytic_proof.md` (4.8.8, same family:
  section 2, splitting lemma; section 5, English SI draft).

## Related

- Third family member 4.6.12: see `z3_4612/` (same Q(sqrt(3)) number field).
