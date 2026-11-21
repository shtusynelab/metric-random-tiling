# z5_snubsquare/ -- 3.3.4.3.4 (snub square) archive

3.3.4.3.4 is one of the eleven Archimedean uniform vertex types; z = 5
all-rational face family {3, 4}, same alphabet as (3,3,3,4,4) but a different
cyclic order.

## Conclusion

dim_R = dim_Q = 2 for every legal torus (N, M both even, exact) ->
Omega = k^2, s(infinity) = 0. The kernel consists of two classes of
rhombus-constant vectors: triangles pair perfectly along T-T edges into
60/120 rhombi, rectangular faces into a x b rectangles. Proof chain: TT
pairing lemma -> rhombus-pair graph locally-constant reduction ->
checkerboard construction G' has exactly two components. Details in
`SNB_3343_metric_rigidity_theorem.md` (incl. English SI draft and physical
reading).

## Files

- `SNB_3343_metric_rigidity_theorem.md` -- theorem + proof + numerical
  acceptance (7 torus sizes, all pass; checker `snub_square_rigidity_check.py`)
  + English SI draft.
- `snub_holonomy_check.py` -- holonomy consistency check.

## Comparison (the three z = 5 cyclic orders)

| Vertex type | Cyclic-order feature | dim | Archive |
|---|---|---|---|
| (3,3,3,4,4) | the two 4s adjacent | 2L+1 (sub-extensive) | `z5_33344/` |
| **(3,3,4,3,4)** | the two 4s non-adjacent | **2 = O(1)** | this directory |
| (3,3,3,3,6) | -- (contains a 6) | 1 (global scale) | `z5_snubhex/` |
