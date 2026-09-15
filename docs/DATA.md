# Data and checkpoint guide

All concentration fields are synthetic diffusion solutions, not furnace measurements. Carbon concentration and absolute concentration errors use wt% (mass percentage points); relative L2 errors are dimensionless. Grid coordinates, process inputs, and normalizations are consumed by the original loaders included in each study.

## Studies

| Study | Train / validation / test | Released inputs | Released trained models |
|---|---|---|---|
| `one_d` | 420 / 90 / 90 | Three HDF5 splits and dataset metadata; 300 equal-mean process pairs remain in the same split | 10 variants, each with seeds 2027, 3407, 7919 |
| `small3d` | 66 / 12 / 22 | Three HDF5 splits and three matching strict-Average-Cp sidecars | MLP, ResNet3D, FNO3D × direct, prior_direct, residual × seeds 2027, 2028, 2029 |
| `large3d` | 66 / 12 / 22 | 100 case HDF5 files, split across ten download archives | FNO3D direct, prior_direct, residual, each with seed 2027 |

The 1-D grid has 256 depth points over 0–6 mm and 61 stored times over 0–6 h. The small 3-D study uses a 32³ grid over a 2 mm domain. The large study uses a 256³ grid over a 50 mm domain. These are different datasets with separately trained models, not a zero-shot resolution-transfer experiment. Each 3-D case has 21 output times. Large-study HDF5 files retain the original stored representation used by its loader; inspect `large3d/code_snapshot/large_fno.py` for reconstruction and normalization.

The six 3-D geometry families are slab, cylinder, ring, stepped shaft, notched block, and gear ring. All six occur in training and testing. Geometry masks identify material cells; diffusion and error evaluation are restricted to the appropriate material/surface regions. The supplied geometry is the numerical geometry, not a CAD-derived industrial part.

## On-disk layout

After running the downloader:

```text
assets/
  one_d/
    data/                 # original train/validation/test HDF5 and JSON
    checkpoints/<variant>_seed<seed>/best.pt
  small3d/
    data/                 # original full-field HDF5 splits
    prior/                # matching strict-Average-Cp HDF5 splits
    checkpoints/<backbone>_<mode>_seed_<seed>/best.pt
  large3d/
    data/<case>.h5
    checkpoints/<mode>/best.pt
```

Keep original filenames and pair each study with its own checkpoints and normalization. Configurations under `one_d/configs`, `small3d/configs`, and `large3d/configs` preserve the corresponding training settings. The small and large studies use prescribed synthetic kinetics; their labels should not be substituted for the 1-D labels based on the Ågren diffusivity relation.

## Integrity and licenses

`assets_manifest.json` lists archive sizes, SHA-256 hashes, and individual-file hashes. For transport, archives are split into parts no larger than 96 MiB. The downloader verifies each part, joins it into the original archive, verifies the archive, and refuses to replace a different existing asset. Download all parts for the requested study when using offline mode. Keep downloaded assets unchanged and send every new run to a separate output directory.

The project-generated datasets and trained checkpoints are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribute the paper and this repository, and indicate any modifications. Code is under Apache-2.0; retained NVIDIA code keeps its original notices. Load only trusted checkpoints: PyTorch checkpoint files are not a safe interchange format for arbitrary untrusted content.
