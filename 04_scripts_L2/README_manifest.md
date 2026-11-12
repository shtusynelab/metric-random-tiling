# Paper-level scripts (`04_scripts_L2/`)

Canonical copies of every script that generates a paper figure, SI figure, or
delivered data table. The authoritative per-item map (which script makes
which figure, and where its data lives) is **Table R2 of the Reproducibility
Supplement** (published with the paper); this manifest
is the per-folder quick reference.

## Path resolution

All scripts import `paths_common.py` (same folder) and resolve every
location through it — `PC.DATA` (delivered data, `03_data/uniform/`),
`PC.FIGURES` (`02_figures/`), `PC.HC` (honeycomb engine), `PC.engine(name)`
(other engines under `06_engines/`), `PC.fname()` (ASCII filename mapping).
No machine-specific paths; scripts run from any working directory. The same
sources also run in the authors' development tree (the resolver detects the
layout); `dev_name_map.json` maps repository ASCII filenames to the
development originals.

## Main figures

| Script | Role |
|---|---|
| `fig1_model_classification.py` | Fig. 1 (six panels: classic disorder / metric honeycomb / three carriers / readout) |
| `fig1c_four_carriers.py` | historical standalone generator of the old Fig. 1c (superseded by the composite above) |
| `fig2_decay_mechanisms.py` | Fig. 2 (four entropy-decay mechanisms; pattern assets in `02_figures/assets_decay/`) |
| `fig2cd_symmetric_forms.py` | canonical panel functions for Fig. 3 c/d (called 1:1 by the Fig. 3 script) |
| `fig3_honeycomb_metric_entropy.py` | Fig. 3 (honeycomb geometric limit; second row, three columns) |
| `fig4_value_domain_halos.py` | Fig. 4 (seven panels: patches/diffraction, ladder, anisotropy, w_rms, tension) |
| `l1_style.py` | shared figure style (canvas/sizes for the four main figures) |

## SI figures

| Script | Role |
|---|---|
| `gen_SI_figs.py` | Figs. S1–S5 (closure schematic, mechanisms, Lambda saturation, tile orbits) |
| `figSI_area_law.py` | Fig. S6 (3.4.6.4 area law) |
| `figS7_4612_spectral.py` | Fig. S7 (4.6.12 ternary series + spectral determination; rates asserted as constants) |
| `figS8_lower_bound_certs.py` | Fig. S8 (four-panel constructive-certificate composite from `02_figures/assets_certificates/`) |
| `fig_si_entropy_split.py` | Fig. S11 (entropy split; enumerator `si_data.py`; data `03_data/uniform/si_entropy_split.csv`) |
| `si_data.py` | exact enumerator behind Fig. S11 (asserts E = 3F and the split identity at every point) |
| `vd_si_sigs.py` | Figs. S9/S10 data and renderings (PDF decay, S_L(q), halo orientation) |

## Constructive certificates (SI S8.3)

| Script | Role |
|---|---|
| `flip_hc_ternary.py` | honeycomb flower certificate (kernel anchors, square-lattice negative control) |
| `flip_3464_31212.py` | 3.4.6.4 and 3.12.12 patch certificates |
| `flip_488lam_4612.py` | 4.6.12 patch certificate (its 4.8.8 section is not used by the paper) |
| `h2l_strip.py` / `h2l_construct.py` | binary stripe construction: dual-algorithm strip counting, interval criterion, three site sets with subset/injectivity/end-to-end tests |
| `flip_certificate_fig.py` / `a4_cert_figs_all.py` | source renderings for Fig. S8 (real-metric, edge length = edge value) |
| Records | `03_data/lower_bounds/*.json` (five result records + two mode tables) |

## Value-domain ensemble suite

| Script | Role |
|---|---|
| `vd_suite3.py` | suite v3.2: six-domain metrics + displacement spectra + h-scan (f_coh against H_ref; data-driven spot positions) |
| `vd_patch.py` | torus-to-open-patch adapter (raised-hexagon disk; ideal/real configuration gates) |
| `vd_torus_run.py` / `vd_torus_run_v2.py` / `vd2_gate.py` | sampler drivers (server-only; see their docstrings) |
| `vd_wrms.py` | w_rms patch-radius ladder (Fig. 4d data source) |
| `vd_ladder_ext.py` / `vd_tension_v2.py` | ladder-extension and tension drivers |
| `vd_suite2.py` / `vd_ideal_ref.py` / `sampler_audit.py` / `vd_suite.py` | historical (superseded/archive; kept for provenance) |
| `external/sampler2.py` + `external/metric_hc.py` | vendored exact-uniform sampler and honeycomb library (provenance in `external/README_provenance.md`) |

## Asset generators (in this folder)

`gen_S9_torus_construction.py` (Fig. 3a counting-domain schematic),
`gen_decay_assets.py` (Fig. 2 pattern assets), `gen_fig1_carrier_assets.py`
and `gen_metric_patches.py` (Fig. 1 carrier patterns). The six shared
pattern-asset generators (`gen_G_*`, `gen_K3_*`, `gen_R4_*`) live in
`05_predecessor_assets/`.

## disk/ (planar open-boundary counting, SI S6.3)

`disk_count.c` (production engine, C with `__int128` and OpenMP;
`gcc -O3 -std=gnu11 -fopenmp`), `disk_lib.py` (reference DFS/frontier-DP/
column-DP), `blocked_multi.py` (blocked-DFS cross-check); exact counts in
`03_data/uniform/disk_counts_exact.csv` (see `disk/README_manifest.md`).
