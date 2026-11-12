# z5_33344/ -- (3,3,3,4,4) elongated triangular archive

(3,3,3,4,4) is one of the eleven Archimedean uniform vertex types; one of the
two z = 5 all-rational face family {3, 4} members (the other, 3.3.4.3.4, is in
`z5_snubsquare/` -- same alphabet, different cyclic order).

## System

Vertex type (3,3,3,4,4); faces triangles : rectangles = 2 : 1. Direction
system on the 30-degree grid (number field Q(sqrt(3))), but the closure rows
are all-rational (face types in {3, 4}; Niven criterion -> no splitting,
dim_Q = dim_R). Combinatorially: alternating square rows and triangle rows
(each double-row cell contains 2 square rows + 2 triangle rows).

## Conclusion

| Layer | Result | Status |
|---|---|---|
| Dimension | dim = 2L + 1 (L x L double-row-cell torus; sub-extensive) -> s(infinity) = 0 | [N] |
| Mechanism | deficiency (net degrees of freedom -V/2) + strip-type residual degrees of freedom | -- |

SI Table S2 status row: [N] (numerically verified). An analytic proof is not
yet written as a separate document (open item); numerical verification =
`z4_3464/RTH_numerical_verification.py`, `build_33344` (part 6 regression:
dimension series + vertex-type census).

## Files

- `ELT_real_metric_random_solution_demo.png/.pdf` -- metric random solution
  demo (R-layer solution-space sampling; patch kernel dimension 6, consistent
  with the sub-extensive count; closure residual ~1e-16; demo rendering script
  not distributed with this release). The visible degrees of freedom are the
  independent heights of the square rows plus one global width (triangle rows
  lock the width; the rectangles' top and bottom edges belong to two different
  triangle rows, transmitting the width globally) -- a direct visualization of
  the sub-extensive "strip residual".

## Related

- The other cyclic order of the same alphabet {3,3,3,4,4}, 3.3.4.3.4: dim = 2
  = O(1) (rigid), see `z5_snubsquare/SNB_3343_metric_rigidity_theorem.md` -- 
  the cyclic order decides the fate (2L + 1 vs 2).
- Third z = 5 member (3,3,3,3,6) snub hexagonal: dim = 1 (global scale), see
  `z5_snubhex/`.
