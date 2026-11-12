# 03_data/ -- data dictionary

## Naming conventions

- Alphabet tags in filenames: `a123` = alphabet {1,2,3}, `a456` = {4,5,6},
  `b12` = {1,2} -- the tag lists the allowed edge-length values of the ensemble
  (integer alphabets, modulo translation/rescaling of the value set).
- Torus tags: `tor21_*` = 21x21 honeycomb torus ensembles; `torus13_*` =
  13x13 torus ensembles (raw per-configuration files under `raw/`).

## uniform/

### spots_*.csv (`spots_a123`, `spots_a234`, `spots_a345`, `spots_a456`, `spots_a567`, `spots_b12`)

Per-spot diffraction readouts, one row per patch in the ensemble. Columns:
`spot` = patch index; `system` = alphabet tag; `f_coh`, `f_inc`, `f_coh_al`,
`f_pk` = coherent fraction, incoherent fraction, centroid-aligned coherent
fraction, peak height; `f_coh_ref`, `f_coh_al_ref`, `f_pk_ref` =
ideal-patch-referenced variants of the same quantities; `fcoh_lo` / `fcoh_hi`
and `fcoh_lo_ref` / `fcoh_hi_ref` = configuration-bootstrap intervals for the
coherent fractions (and their referenced variants); `r_lo` / `r_hi` =
configuration-bootstrap interval endpoints accompanying the spot's radial
measure; `H_ens`, `core`, `halo`, `aniso`, `hc_lo`, `hc_hi` = auxiliary
per-spot quantities (definitions: see RS Table R2).

### exact_size_rule_grid.csv

Grid of alphabet-size points: 2,232 rows, 30 alphabets with k = 2..8, windows
Lx = 3..150 x Ly = 2..6. Columns: `alphabet` (quoted list of allowed values),
`k`, `sidon` (alphabet is a Sidon set: all pairwise sums distinct), `has3ap`
(alphabet contains a nontrivial three-term arithmetic progression), `Lx`,
`Ly`, `cls` (classification: `exact` / `resonant` / `anomalous`), `law` (the
perimeter-law prediction, k^(Lx+Ly+1) in every row), `Omega` (exact count),
`ratio` (= Omega / law), `reuse` (runtime bookkeeping flag, `True` throughout),
`n_states`, `secs` (runtime metadata). The `cls` prediction is re-derivable
from the alphabet flags together with the window dimensions (their gcd); see
RS Table R2 for the rule statement. All 40 `anomalous` rows occur at
gcd(Lx, Ly) = 2.

### si_entropy_split.csv

One row per (alphabet, torus size): alphabet tags `12`, `123`, `124` (i.e.
{1,2}, {1,2,3}, {1,2,4}) over nine (Lx, Ly) torus sizes from 2x2 to 6x3
(20 rows). Columns: `alpha`, `Lx`, `Ly`, `F`, `E` (E = 3F asserted),
`omega` (exact count), `ln_Omega`, `H_comp`, `H_cond` with
ln Omega = H_comp + H_cond (the Fig. S11 entropy split), per-face variants
`H_comp_per_F` / `H_cond_per_F`, and `n_comp`, `secs` run metadata.

### disk_counts_exact.csv

Exact planar-disk counts, delivered as TWO data blocks sharing one header row,
separated by `#` comment lines: the binary {1,2} block (R = 0..7) followed by
the arithmetic {1,2,3} block (R = 0..4). Read with
`pandas.read_csv(..., comment='#')` and split the rows by the `alphabet`
column. Columns: `R` (disk radius), `alphabet`, `N_faces`, `E_edges`,
`boundary_edges`, `Omega_exact`, `ln_Omega`, `paths_agreed` (which independent
engines reproduced the value: Python DFS, blocked DFS, column DP, C
scatter/gather modes). The `#` lines record the geometry convention (cube
coordinates, max norm <= R), the two closure equations, engine abbreviations,
and log-linear fit summaries (binary slope a = 0.2303..0.2313 across
sub-windows; arithmetic a = 0.5574..0.5619).

### 06_engines/z3_4612/RTH4612_knuth_estimate_series.csv

Knuth tree-sampling series for the 4.6.12 ternary counts. The `#` header
records the column semantics and estimator caliber: `Omega_gauge_full_orbit`
holds the FULL translation-orbit count Omega_full = 3 x E[X] (the column name
is retained from an earlier mislabeled version for column-order stability);
`Delta_ln` = naive s.d. of ln Omega_full (a total, not per-hole error:
per-hole error = Delta_ln / N_F; heavy-tail caveat -- the naive s.d.
underestimates the true uncertainty of this estimator family); `hit_rate` =
fraction of samples reaching a verified complete leaf; seeds 20261001 +
7919*c per chunk, pooled over chunks. Data columns: `L`, `samples`,
`Omega_gauge_full_orbit`, `Delta_ln`, `hit_rate`, `seconds`, `status`
(`anchor` / `production` / `superseded`), `note`. The L = 1 / L = 2 `anchor`
rows validate the final estimator against exact totals (2,346 vs 2,349 and
2.971e7 vs 29,723,517); pre-fix rows are marked `superseded` and kept for
audit.

### meanF2_*.npy

Structure-factor arrays <F(q)^2> per value domain (601 x 601 float64 grids):
`meanF2_a123.npy`, `meanF2_a234.npy`, `meanF2_a345.npy`, `meanF2_a456.npy`,
`meanF2_a567.npy`, `meanF2_b12.npy`, plus the displacement-field scan family
`meanF2_a123_h+0.15.npy`, `meanF2_a123_h-0.15.npy`, `meanF2_a123_h+0.30.npy`,
`meanF2_a123_h-0.30.npy`. These are the maps used by Fig. 4 (value-domain
halos).

### Other files in uniform/

`ens_*.npz` (ensemble identifiers and sampler metadata), `raw/` (raw
per-configuration ensembles: `tor21_*.npz`, `torus13_*.npz`, including the
h-scan variants `torus13_a123_h+/-0.15` / `+/-0.30`, with sampler metadata),
`displacement_spectra.csv`, `hscan_summary.csv`, `wrms_curve.csv`,
`wrms_curve_tor21.csv`, `ladder_ext.csv`, `size_scaling_tor21.csv`,
`pdf_decay_suite.csv`, `pdf_curves_suite.npz`, `slq_suite.csv`,
`slq_profiles_suite.npz`, `halo_orient_suite.csv`,
`halo_orient_lock_suite.csv` -- value-domain suite series; see RS Table R2
(Table S7 row).

## lower_bounds/

JSON records of the constructive lower-bound certificates (`flip_*_result.json`,
`h2l_*.json`, `mode_*.json`); see RS Table R2.
