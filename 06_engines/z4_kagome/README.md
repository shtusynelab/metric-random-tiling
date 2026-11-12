# z4_kagome/ -- kagome (3.6.3.6), z = 4: zero metric entropy (closed)

## System

- Vertex degree 4; faces = equilateral triangles + regular hexagons (2:1);
  vertex angles 60 x 2 + 120 x 2 = 360.
- Edge directions: the isometric constraint locks them into 3 fixed families;
  at each vertex the 4 edges form two collinear pairs.
- Metric constraints: each triangle binds its three edges to a single variable
  t (exact equilateral reduction); hexagons inherit an additive closure
  constraint (t1 + t2 = t4 + t5 form, same type as the honeycomb).
- Constraint count: net degrees of freedom 0 (critical); the constraint matrix
  is exactly 2F x 2F square.

## Theorem (zero entropy)

- Exact symbolic factorization: det M(k) = 3 (a-1)(b-1)(a-b) / (a^2 b) with
  a = e^(i k1), b = e^(i k2); the zero set is the three root-wraparound lines
  {k1 = 0} union {k2 = 0} union {k1 = k2} on the k-torus.
- dim ker M(Lq, Lr) = Lq + Lr + gcd(Lq, Lr) - 1 (exact), sub-extensive
  -> Omega <= k^dim -> s(infinity) = 0.
- Kernel structure: three families of strip modes (row / column / anti-diagonal
  bipartite alternating modulation), dimension Lq + Lr + g - 1.
- Mechanism: spectral type (zero lines of a nowhere-vanishing symbol), in
  contrast to the square lattice's propagation type.

## Files

- `KAG_zero_entropy_theorem_analytic_proof.md` -- theorem and proof (incl.
  English SI draft).
- `kagome_lib.py` -- model library: torus / tiling construction, constraint
  matrix, exact linear algebra, kernel enumeration, strip TM (state-merged).
- `KAG_numerical_verification.py` + `KAG_entropy_density_verification.csv` -- 
  four-layer numerical verification (brute-force anchors / nullity table +
  symbol factorization / exact Omega series / strip series).
- `kagome_real_demo.py` + `KAG_real_metric_random_solution_demo.png/.pdf` -- 
  real-metric random solution demo.
