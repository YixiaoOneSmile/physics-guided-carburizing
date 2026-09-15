# Release verification

Verification date: 15 September 2026.

The release is a portable copy of the completed experiments. Numerical solvers, model definitions, training losses, and evaluation functions were not rewritten. Launchers adapt paths and create isolated output folders. Training starts a new run; evaluation loads an existing checkpoint without updating it.

## Source preservation

728 Python files from the selected experiment code and its frozen PhysicsNeMo runtime were compared byte-for-byte with their server originals. All matched. `frozen_source_sha256.json` records the hashes. The runtime is retained because replacing its FNO implementation could change behavior or checkpoint compatibility. Configuration changes are limited to portable paths and new run-output locations.

All 100 locally archived large-study HDF5 files, all 60 selected trained checkpoints, and all nine transferred 1-D/small-3-D HDF5 split files were checked against server SHA-256 hashes and matched. The download manifest extends integrity checks to the release archives and their contents.

## Original-checkpoint reproduction

The public launchers were tested using the original checkpoints, with seed 2027:

| Study and model | Test cases | Reproduced concentration MAE (wt%) | Reproduced relative L2 |
|---|---:|---:|---:|
| 1-D depth-gated descriptor TCN | 90 | 0.001348511059121746 | 0.014885794066099658 |
| 2 mm FNO3D residual | 22 | 0.0029497031937353313 | 0.005725101070393893 |
| 50 mm FNO3D residual | 22 | 0.0003709031661529548 | 0.0032536815986865556 |

All 46 checked non-timing numeric entries matched the corresponding archived evaluations exactly in the original software/hardware environment. The machine-readable comparisons are in `reproduction_checks.json`. Timing is intentionally excluded because it varies with machine load and hardware. These checks cover representative original checkpoints, not a fresh retraining of every published run.

## Executability checks

- CPU forward passes through MLP, ResNet3D, and FNO3D passed shape and finite-value assertions.
- A separate FNO3D training smoke test (one epoch, two training steps) completed, evaluated validation data, and wrote a new checkpoint. Its weights are not included as research results.
- The 1-D and both 3-D release evaluation launchers completed their full test sets as reported above.
- The installer creates a repository-local virtual environment and does not install into the original experiment environment.
- A clean Python 3.11 CPU virtual environment was installed with `bash install.sh cpu`. All three forward checks passed. A full 22-case FNO3D residual evaluation in this environment returned MAE 0.002949737276966599 wt% and relative L2 0.005725836455398662. The MAE differs from the original GPU result by about 3.41e-8 wt%; CPU/GPU results need not be bit-identical. The installed package versions are recorded in `environment_cpu_verified.txt`.
- The 1-D transport parts were joined, archive checksums verified, and both data and checkpoint archives successfully extracted using the offline downloader.

The published numerical results use the original trained checkpoints. Small floating-point differences can occur on different hardware or library versions. CPU installation and forward-pass checks do not imply that the 50 mm training experiment is practical on a CPU.
