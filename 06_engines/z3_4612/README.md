# z3_4612/ -- 4.6.12 counting engine

Counts edge-length assignments on the 4.6.12 network (squares, hexagons,
dodecagons; number field Q(sqrt(3))), on tori and strips, by exact DP /
transfer matrix and by Knuth tree sampling. Delivered result: the naive binary
perimeter law on this network is falsified, and strict positive width rates
are established -- exact transfer-matrix lower bounds against Knuth-sampling
upper bounds pin the width-rate constants (asserted as constants in
`04_scripts_L2/figS7_4612_spectral.py`, which renders Fig. S7).

## Key files

- `RTH4612_binary_anchors.csv` -- exact binary anchors, dual independent
  engines; `RTH4612_binary_strips.csv` -- W = 2 frontier-DP strip series;
  `RTH4612_binary_L4x3_summary.csv` (+ `RTH4612_binary_L4x3_blocks.csv`) -- 
  depth-split 4x3 exact settlement; `RTH4612_binary_frontier.csv`,
  `RTH4612_binary_pinned_t2p.csv` -- frontier and pinned tables.
- `RTH4612_knuth_estimate_series.csv` -- ternary Knuth tree-sampling series;
  the file head records column semantics, estimator caliber, seed convention,
  and the L = 1 / L = 2 anchor-validation runs (pre-fix rows marked
  `superseded`, kept for audit). See also `03_data/README.md`.
- `RTH4612_knuth_strips2.csv` -- strip-Knuth series (extracted from
  `run_knuth_strips2.log`).
- `RTH4612_three_rank_dimR_dimQs3_dimSplit.csv` -- rank data (dim_R, dim_Q,
  dimension split).
- Counters: `roundD_4612_numeric.py` (exact counts), `roundE_*` (anchors,
  frontier DP, pinned DP, Knuth estimators, order probes),
  `roundF_server_blocks_binary.py` / `roundF_split_slow.py` (blocked server
  drivers), `rth4612_tm.py` (transfer-matrix spectra),
  `ansatz_diagnostics.py`, `plot_sZ_1overL.py`.
- `dev_calibrations/` -- Hutchinson estimator calibration JSONs
  (`hutch_locW2_result.json` ... `hutch_locW8_result.json`).
- `TTH4612_lnOmega.png/.pdf` -- ln Omega vs N_F overview plot.
- Dated determination / audit records (`RTH4612_*_adjudication*.md`,
  `RTH4612_constraint_matrix_layer_results_2026-10-01.md`,
  `RTH4612_pinning_splice_theorem_derivation.md`, task logs) are the historical
  provenance for the conclusions above.
