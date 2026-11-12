# z3_hexagonal_P0/ -- honeycomb metric-counting engine

Exact counting of edge-length assignments on hexagonal (3.6.3.6) tori and
strips: integer values on every edge, the two per-face closure equations per
hexagon (eqA: L1+L2 = L4+L5; eqB: 2*L0+L1+L5 = 2*L3+L2+L4), counted exactly by
DFS, transfer matrices (Tr(T^Lx)), and Knuth tree sampling. The honeycomb is
the positive-entropy carrier of the paper -- its counting entropy is carried by
winding sectors. This engine delivers the transfer-matrix spectra (per-width
lambda(W) = ln rho(W)), the exact vs resonant torus shape classes, the
resonant families, and the growth-form series behind the paper's honeycomb
figures.

## Key files

- `pseudocrystal_lib.py` -- shared library: geometry, solvers, counting (used
  across the paper-level suite).
- `edge_alphabet_affine_check.py` -- official small-torus anchor: affine
  invariance of arithmetic-progression alphabets (Omega({4,5,6}) pointwise
  equal to all shifts and rescalings, e.g. {3,4,5}, {8,10,12}, {3,5,7};
  non-AP alphabets strictly smaller). DFS assigns edge by edge, peak < 10 MB.
- `edge_alphabet_binary_check.py` -- official small-torus anchor for the
  two-value class: Omega({0,1}) = Omega({4,6}) = Omega({0,2}) pointwise,
  plus a brute-force (2,2) cross-check.
- `P0_PM_*` -- torus/strip traces, shape classes, growth forms
  (`P0_PM_hc_trace*.py`, `P0_PM_strip_growth*.py`, `P0_PM_torus_growth.py`,
  `P0_PM_impl_vs_metric.py`), resonant-family counts
  (`P0_PM_resonant_families_Ly9.csv` + anchor-gate log).
- `step8_hc_tm.py` ... `step14_hc_processing.py` -- transfer-matrix spectra and
  fits: Perron roots rho(W) per width (binary, ternary), binary spectral
  series, s-fits, long traces, and the finalization of the asymptotic reading.
- `TSDHC_*.csv` -- delivered TM series (`TSDHC_tm_rho.csv`,
  `TSDHC_tm_rho_nonarith.csv`, `TSDHC_binary_spectral.csv`,
  `TSDHC_nonarith_spectral.csv`, `TSDHC_long_trace*.csv`,
  `TSDHC_s_fits_summary.csv`, `TSDHC_processing.csv`).
- `P0_PM_torus_*.csv` -- delivered exact-count series (torus traces, trace
  extensions, shape classes, growth forms).
- `P0_A2*/P0_A2b_*` -- entropy-density tables and Knuth tree-sampling
  calibration / verification tables (`P0_A2_entropy_density_*.csv`,
  `P0_A2b_Knuth_*.csv`, `P0_A2b_TM_*.csv`).
- `P0_L1_halo_anisotropy.py` and `P0_M0_PDF_three_reference.py` -- modules used
  by the paper-level suite in `04_scripts_L2/` (halo anisotropy; pair-
  distribution reference curves).
- `_archive_deprecated_routes/` -- superseded counting routes kept for
  reference.

## Quickstart

```bash
python edge_alphabet_affine_check.py
python edge_alphabet_binary_check.py
```

Both are assert-based self-checks; all assertions pass. They resolve their
inputs relative to the script directory and run from any working directory
(from the repository root: `python 06_engines/z3_hexagonal_P0/<script>.py`).
