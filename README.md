# Metric random tilings: counted geometric disorder and its diffraction

Code, data and figures for the Letter by Xinyan Wu and Qingqing Ji. The [manuscript index](01_paper/README.md) links the current Letter, Supplemental Material and Reproducibility Supplement (v6.2.5). The [figure index](02_figures/README_manifest.md) maps Figs. 1–4 and S1–S5 to the released artwork.

Archived on Zenodo: concept DOI [10.5281/zenodo.24156133](https://doi.org/10.5281/zenodo.24156133); version-specific DOIs are assigned per Zenodo release.

The current manuscript numbers the constructive certificates, 3.4.6.4 counts and 4.6.12 rates as Figs. S1, S2 and S3, respectively. In the fixed Zenodo v6.1.9 snapshot these figures are S3, S1 and S2.

## Layout

| Folder | Contents |
|---|---|
| `01_paper/` | the Letter, SI, and Reproducibility Supplement as PDFs |
| `02_figures/` | all delivered figures (PDF + PNG, Figs. 1–4 and S1–S5) with rendering assets |
| `03_data/` | delivered data: `uniform/` (ensembles, structure factors, exact counts, NPZ raw configurations), `lower_bounds/` (certificate records) |
| `04_scripts_L2/` | paper-level scripts: one script per figure, the value-domain suite (`vd_*`), SI data generators, constructive-certificate checkers (`flip_*`, `h2l_*`), shared style, and `paths_common.py` (see below) |
| `05_predecessor_assets/` | shared pattern-asset generators (Fig. 1/2 input art) |
| `06_engines/` | per-network counting engines (one folder per tiling network), kept beside their data |

## Quick start

```bash
pip install -r requirements.txt
# official small-torus anchors (affine invariance, brute-force cross-checks):
python 06_engines/z3_hexagonal_P0/edge_alphabet_affine_check.py
python 06_engines/z3_hexagonal_P0/edge_alphabet_binary_check.py
```

Everything resolves paths through `04_scripts_L2/paths_common.py`, from any
working directory, so no script carries machine-specific paths. The same
sources also run in the authors' development tree (the resolver detects the
layout); `dev_name_map.json` next to it maps the repository's ASCII filenames
to the development originals (byte-identical, only names changed).

## Counting engines (`06_engines/`)

| Folder | Network | Character |
|---|---|---|
| `z3_hexagonal_P0` | honeycomb (6.6.6) | the positive-entropy carrier; exact traces, TM spectra, shape classes |
| `z3_31212` | 3.12.12 | mid-system positive entropy, TM ladder |
| `z3_4612` | 4.6.12 | perimeter-law falsification, strict width rates |
| `z3_488` | 4.8.8 | ℤ zero entropy / ℝ extensivity, Λ-library counts |
| `z4_3464` | 3.4.6.4 | triangle locking, sparse-trace area law |
| `z4_kagome` / `z4_square` / `z6_triangular` | kagome / square / triangular | zero-entropy mechanisms, closed forms |
| `penrose` | Penrose rhombi | strip theorem Ω = k^#strips |
| `z5_snubhex` / `z5_snubsquare` / `z5_33344` | snub hexagon / snub square / 3.3.3.4.4 | rigidity and sub-extensive cases |

## Reproducing the paper figures

Each main/SI figure has one generating script in `04_scripts_L2/` (map in RS
Table R2). Figures are rendered from delivered data by default; regenerating
raw ensembles needs the sampler drivers (`vd_torus_run*.py`, server-scale)
and the exact-uniform suite (`vd_suite3.py`, `vd_patch.py`). The constructive
certificates of SI S2 re-verify in seconds-to-minutes via `flip_*.py` /
`h2l_*.py` (records land in `03_data/lower_bounds/`, matching the delivered
JSONs bit-for-bit).

## License

Code and datasets: MIT — see `LICENSE`.  If you use this code or data,
please cite the paper.
