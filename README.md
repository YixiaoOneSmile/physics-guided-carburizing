# Physics-guided residual learning for gas carburizing

Reference implementation for **Physics-Guided Residual Learning for Transient
Carbon Fields in Gas Carburizing** by Yixiao Sun, Wenyu Zhang, Xusheng Li and
Dongying Ju. The study predicts numerical carbon-concentration fields under
prescribed, time-varying carburizing schedules.

The physical reference uses the **time-averaged carbon potential**, while
retaining temperature history, mass-transfer conditions, material and geometry.
The network predicts a dynamic correction; the final prediction is the reference
plus that correction. Temporal encoding and a surface-distance gate supply the
spatial predictor with process-history features.

## What is included

| Study | Data | Published comparisons |
|---|---|---|
| `one_d` | 600 cases, 300 pairs; 420/90/90 split; 61 times and 256 depth nodes | 10 model variants, 3 seeds; depth-time prediction |
| `small3d` | 100 cases, 66/12/22 split; six geometry families; 21 times, 32³ nodes, 2 mm domain | MLP, ResNet3D and FNO3D; 3 target forms; 3 seeds |
| `large3d` | Same 100 case conditions and split; 21 times, 256³ cells, 50 mm domain | Separately trained FNO3D; 3 target forms; seed 2027 |

The six geometry families are slab, cylinder, ring, stepped shaft, notched block,
and gear ring. Both 3D tests contain known geometry families. The 50 mm experiment
uses **scale-specific retraining**, not zero-shot transfer of 2 mm weights.
These are simulated diffusion benchmarks, not measured furnace trials.

Only the final experiment sources and validation-selected `best.pt` weights are
released. Intermediate checkpoints, discarded runs, draft manuscripts, logs
containing server paths, and unrelated experiments are excluded. Sources in
`code_snapshot/` retain their historical names so checkpoint provenance remains
traceable; a `v3` or `geosurface` identifier alone is not a dataset-version label.

## 1. Install

Recommended platform: Linux, Python **3.11**. Python 3.11–3.13 is accepted by the
installer. CPU is sufficient for environment checks and the 1D/small-3D examples.
Full 256³ training/evaluation requires a large-memory CUDA GPU; the reference run
used an **RTX PRO 6000 Blackwell, 96 GB**. No lower GPU memory minimum is claimed.

```bash
git clone https://github.com/YixiaoOneSmile/physics-guided-carburizing.git
cd physics-guided-carburizing
bash install.sh cpu
source .venv/bin/activate
```

For the reference CUDA 13.0 backend, on a compatible NVIDIA driver:

```bash
bash install.sh cu130
source .venv/bin/activate
```

Set `PYTHON_BIN=/path/to/python3.11` before the command if needed. Installation
only uses this repository's `.venv`; it does not modify another environment.
The script installs dependencies and runs an actual forward pass for all three
3D backbones. If this final check fails, installation is **not** considered complete.

The reference environment used PyTorch 2.12.0+cu130, NumPy 2.4.6, SciPy 1.17.1,
HDF5/h5py 3.16.0, and an untagged PhysicsNeMo 2.2.0a0 runtime. The exact runtime
source is included under `vendor/physicsnemo_runtime`, with NVIDIA's Apache-2.0
license and file hashes, to avoid silently substituting a different FNO version.
It contains no training data or example projects. See [NOTICE](NOTICE).

## 2. Download data and trained weights

Data and weights: [Google Drive release folder](https://drive.google.com/drive/folders/1RULe4kRjiY6tGU-RaTANu3U9-VIIz07b).
Archive names, sizes and SHA-256 checksums are listed in [assets_manifest.json](assets_manifest.json).

**Access status (15 September 2026):** all 65 transport parts have been uploaded.
The folder owner still needs to enable public-link viewing. Until that setting
is enabled, anonymous/automated Drive downloads will request permission. The
code, manifest, and offline loading instructions are public independently.

Start with the small 3D benchmark:

```bash
python scripts/download.py --study small3d
# Other options:
python scripts/download.py --study one_d
python scripts/download.py --study large3d
```

The 50 mm data are split into ten archives, and transport files are split into
parts no larger than 96 MiB. The downloader joins and verifies them automatically.
Download only the study you need. Leave
space for parts, assembled archives, and extracted files (at least 20 GB for all
assets, plus dependencies and new training outputs).

If Google Drive throttles automated downloads, download all archive parts manually
to `downloads/`, then run:

```bash
python scripts/download.py --study small3d --offline
```

The downloader verifies archive checksums, rejects unsafe archive paths and
refuses to overwrite an existing file unless it already matches the expected
hash. Use only trusted checkpoints; the original evaluators load PyTorch
checkpoint dictionaries, which must not be replaced with untrusted files.

## 3. Reproduce inference

```bash
# 22 test cases, all 21 output times, published FNO residual checkpoint
python run.py evaluate --study small3d --backbone fno --mode residual --seed 2027

# Same-input direct prediction comparator
python run.py evaluate --study small3d --backbone fno --mode prior_direct --seed 2027

# 90 test cases, all stored depth/time points
python run.py evaluate --study one_d --seed 2027

# Full 50 mm test set; large-memory CUDA GPU required
python run.py evaluate --study large3d --mode residual
```

Every command prints a new `work/...` output directory. Metrics and per-case
outputs are written there. No downloaded checkpoint or original dataset is
modified. To choose your own output path, add `--output work/my_new_run`; the
launcher refuses an already-existing directory.

For `small3d`, choose `--backbone mlp|resnet|fno`, `--mode direct|prior_direct|residual`,
and `--seed 2027|2028|2029`. For `one_d`, seeds are 2027, 3407 and 7919; use
`--variant` with a name from `one_d/configs/` (without `_seedXXXX.json`). The
default is the adopted `depth_blend_descriptors_tcn` model. `large3d` supports the
three target forms but only the published FNO backbone and seed 2027.

## 4. Train without overwriting released weights

First test two optimizer updates, using a fresh run directory:

```bash
python run.py train --study small3d --backbone fno --mode residual --epochs 1 --steps 2
```

For the complete published schedule, omit the short-run overrides:

```bash
python run.py train --study small3d --backbone fno --mode residual --seed 2027
python run.py train --study one_d --seed 2027
python run.py train --study large3d --mode residual --hours 24
```

The 3D runs use 40 epochs and up to 160 updates per epoch. One-dimensional
settings, loss weights and stopping rules come from the released per-run JSON.
The original 50 mm run took approximately **4.5 hours per target form** on the
96 GB reference GPU; this is not a runtime promise for other hardware.
Large runs retain the original wall-time guard, initialized afresh by the
launcher. A short-run result is a software check, not a replacement for paper
metrics. Checkpoints from new runs stay under `work/`.

To evaluate a newly trained small-3D run, pass its checkpoint parent:

```bash
python run.py evaluate --study small3d --backbone fno --mode residual \
  --checkpoint-dir work/YOUR_TRAIN_RUN/checkpoints
```

For 1D, the original trainer writes `best.pt` directly into the printed run's
`checkpoints/`; invoke its evaluator directly if evaluating that new run:

```bash
PYTHONPATH=one_d/code_snapshot python one_d/code_snapshot/scripts/evaluate_publication_residual_tcn.py \
  --dataset-root assets/one_d/data \
  --checkpoint work/YOUR_TRAIN_RUN/checkpoints/best.pt \
  --output-dir work/YOUR_NEW_EVALUATION
```

## 5. Regenerate the large-scale diffusion data

```bash
python run.py generate --study large3d --hours 24
```

This uses the recorded case definitions and original GPU diffusion solver, with
outputs in a **new** directory. It does not overwrite the released labels. It
requires the same high-memory CUDA environment as the large experiment. Most
users should download the frozen labels to compare methods on identical inputs.
The 1D and small-3D physical solver sources are included; their released data and
fixed splits are the reference inputs for model training/evaluation.

## Target definitions and evaluation

| Target | Physical-reference input | Model output |
|---|---|---|
| `direct` | Reference channel zeroed | Full carbon field |
| `prior_direct` | Average-Cp field | Full carbon field |
| `residual` | Same Average-Cp field | Correction added to the reference |

All three 3D forms keep the same channel count. `residual` vs `prior_direct` is
the matched comparison of output target formulation, not a comparison between
different input information. Normalization statistics are fitted on training
data only. Do not recompute them on the test set or shuffle individual cases
across the provided splits.

Carbon content and its MAE are in **wt%**; relative L2 error is dimensionless
and is not classification accuracy. A 75.2% error reduction means
`100 * (baseline_error - residual_error) / baseline_error`. Report model-forward
timing separately from physical-reference construction.

| Published comparison | Error reduction |
|---|---:|
| 1D vs Average-Cp: full field / surface history / final profile MAE | 43.8% / 98.1% / 58.3% |
| Small 3D residual vs prior-direct: MLP / ResNet3D / FNO relative L2 | 19.0% / 38.2% / 57.9% |
| Large 3D residual vs prior-direct: FNO relative L2 | 75.2% |

See [data documentation](docs/DATA.md), [verification](docs/VERIFICATION.md) and
the accompanying manuscript for definitions and study-specific assumptions.

## License and citation

Code: Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Research data and
trained weights: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
Credit the authors and cite the repository and manuscript when using them.
The manuscript is a submission, not an accepted publication; no DOI is assigned
in this release.
