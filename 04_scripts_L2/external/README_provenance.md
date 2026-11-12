# external/ -- cross-project acquisitions (byte-for-byte copies)

Origin: `MOFtronics/pseudocrystal transport/code/` (2026-10-11 state, including h-field biased-ensemble support).
- sampler2.py: two-stage exact uniform sampler (delta-difference coordinates + sparse combination proposals + parallel tempering; h=0 uniform / h!=0 Boltzmann).
- metric_hc.py: honeycomb torus construction `build_torus`, closure rows `closure_rows`, saturated Z-kernel basis `kernel_zbasis`/`reduce_basis`.
Correctness gates (a reuser must re-run them, see README_sampler2.md sect. 6): h=0 counting regressions / small-size full-enumeration cross-checks / reflection criterion / counting identities.
Use in the L2 paper: exact uniform sampling of the value-range ensemble (replacing the DFS production sampler) + the knobs for the biased-sampling / h-field metric-pressure topics.

Both vendored modules are author-owned and are distributed under the same MIT license as the rest of this repository.
