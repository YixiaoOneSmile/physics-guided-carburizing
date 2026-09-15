"""Relocate output globals without modifying archived numerical/model code."""
import sys
import runpy
from pathlib import Path
code, output, action, mode = sys.argv[1:]
sys.path.insert(0,code)
import build_large_data
build_large_data.ROOT=Path(output)
import large_fno
large_fno.ROOT=Path(output)
script={'train':'train_three_fno.py','evaluate':'evaluate_three_fno.py','generate':'build_large_data.py'}[action]
sys.argv=[script]+(['--mode',mode] if action!='generate' else [])
if action == 'generate':
    build_large_data.main()
else:
    runpy.run_path(str(Path(code)/script),run_name='__main__')
