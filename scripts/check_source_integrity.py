"""Verify that the numerical/model source snapshot has not changed."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'docs/frozen_source_sha256.json').read_text())
failed=[]
for name,expected in manifest.items():
    path=ROOT/name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        failed.append(name)
if failed:raise SystemExit('Source integrity check failed: '+', '.join(failed))
print(f'All {len(manifest)} frozen source files match the release snapshot.')
