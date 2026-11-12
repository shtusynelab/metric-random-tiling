# penrose/ -- Penrose rhombus engine

Counts edge-length assignments on Penrose rhombus tilings. Delivered result -- 
the strip theorem: Omega = k^(number of strips) exactly, for any edge-to-edge
rhombus equilateral tiling (Penrose included). The strips correspond one-to-one
to the lines of the five de Bruijn grids, so a radius-R patch has O(R) strips
against O(R^2) faces -- the counting entropy is s = 0: Penrose cannot carry
metric disorder.

## Key files

- `penrose_lib.py` -- model library (rhombus construction, strip decomposition,
  counting).
- `PEN_strip_verification.csv` -- numerical verification in three layers:
  `A_struct` (structural consistency), `B_scale` (strand-count scaling,
  s_per_face -> 0), `C_omega` (exact Omega = k^#strands, e.g. 531441 = 3^12
  and 4782969 = 3^14, plus the brute-force check 16384 = 2^14).
- `PEN_strip_theorem_analytic_proof.md` -- theorem and proof.
- `PEN_numerical_verification.py` -- verification driver.
