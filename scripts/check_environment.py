"""Check the actual frozen FNO import and model forward, not just pip exit status."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'vendor/physicsnemo_runtime'), str(ROOT/'small3d/code_snapshot')]
import torch
import numpy
import h5py
import scipy
from matrix_models3d import MatrixPredictor3D
torch.set_num_threads(2)
for backbone in ('mlp', 'resnet', 'fno'):
    net = MatrixPredictor3D(backbone=backbone, grid_channels=17).eval()
    with torch.no_grad():
        y = net(torch.zeros(1,17,16,16,16), torch.zeros(1,4,21), torch.tensor([10]))
    assert y.shape == (1,1,16,16,16) and torch.isfinite(y).all()
    print(backbone, 'forward OK')
print({'torch':torch.__version__, 'numpy':numpy.__version__, 'h5py':h5py.__version__,
       'scipy':scipy.__version__, 'cuda_available':torch.cuda.is_available()})

