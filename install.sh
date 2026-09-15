#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
backend="${1:-cpu}"
case "$backend" in cpu|cu130) ;; *) echo 'Usage: bash install.sh [cpu|cu130]'; exit 2;; esac
py="${PYTHON_BIN:-python3}"
"$py" -c 'import sys; assert (3,11)<=sys.version_info[:2]<(3,14), "Python 3.11-3.13 required"'
if [[ ! -d .venv ]]; then "$py" -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url "https://download.pytorch.org/whl/$backend"
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py
echo 'Installed in .venv. Activate with: source .venv/bin/activate'
echo 'Next: python scripts/download.py --study small3d'

