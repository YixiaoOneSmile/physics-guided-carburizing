"""Portable launchers around unmodified experiment sources.

Downloaded assets are read-only inputs. Each invocation uses a fresh output
directory; no original experiment or published checkpoint is overwritten.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train', 'evaluate', 'generate'])
    p.add_argument('--study', choices=['one_d','small3d','large3d'], required=True)
    p.add_argument('--backbone', choices=['mlp','resnet','fno'], default='fno')
    p.add_argument('--mode', choices=['direct','prior_direct','residual'], default='residual')
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--variant', default='depth_blend_descriptors_tcn')
    p.add_argument('--assets', type=Path, default=ROOT/'assets')
    p.add_argument('--output', type=Path)
    p.add_argument('--epochs', type=int)
    p.add_argument('--steps', type=int)
    p.add_argument('--checkpoint-dir', type=Path, help='Alternative checkpoint tree from a training output')
    p.add_argument('--hours', type=float, default=24)
    a = p.parse_args()
    if a.hours <= 0: p.error('--hours must be positive')
    asset = a.assets.resolve()/a.study
    if not (asset/'data').is_dir() and a.action != 'generate':
        p.error(f'Missing {asset}/data; download this study first')
    tag = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = (a.output or ROOT/'work'/f'{a.study}_{a.action}_{tag}').resolve()
    if out.exists(): p.error(f'Output already exists: {out}. Use a new directory.')
    out.mkdir(parents=True)
    for folder in ('configs','results','logs','checkpoints'): (out/folder).mkdir()
    code = ROOT/a.study/'code_snapshot'
    (out/'code_snapshot').symlink_to(code, target_is_directory=True)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT/'vendor/physicsnemo_runtime'),str(code)])
    (out/'invocation.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},indent=2))
    ckroot = a.checkpoint_dir.resolve() if a.checkpoint_dir else asset/'checkpoints'
    if a.study == 'one_d':
        if a.action == 'generate': p.error('Use the released 1-D dataset; this launcher covers training and evaluation.')
        ck = ckroot/f'{a.variant}_seed{a.seed}'
        if a.action == 'evaluate':
            cmd=[sys.executable,str(code/'scripts/evaluate_publication_residual_tcn.py'),
                 '--dataset-root',str(asset/'data'),'--checkpoint',str(ck/'best.pt'),'--output-dir',str(out/'results')]
        else:
            cfg = json.loads((ROOT/'one_d/configs'/f'{a.variant}_seed{a.seed}.json').read_text())
            excluded={'dataset_root','output_dir','train_sha256','validation_sha256','device','model_type','prediction_task','query_channels'}
            if a.epochs: cfg['epochs']=a.epochs
            cmd=[sys.executable,str(code/'train_publication_residual_tcn.py'),'--dataset-root',str(asset/'data'),'--output-dir',str(out/'checkpoints')]
            for k,v in cfg.items():
                if k in excluded: continue
                key='--'+k.replace('_','-')
                cmd += [key if v else '--no-'+k.replace('_','-')] if isinstance(v,bool) else [key,str(v)]
    elif a.study == 'small3d':
        if a.action == 'generate': p.error('Use the released small3d dataset; solver source is included for inspection.')
        cfg = json.loads((ROOT/'small3d/configs/matrix.json').read_text())
        cfg.update(source_root=str(asset/'data'),sidecar_root=str(asset/'prior'),experiment_root=str(out))
        if a.epochs: cfg['epochs']=a.epochs
        if a.steps: cfg['steps_per_epoch']=a.steps
        config=out/'configs/matrix.json';config.write_text(json.dumps(cfg,indent=2))
        if a.action == 'evaluate':
            name=f'{a.backbone}_{a.mode}_seed_{a.seed}'
            (out/'checkpoints'/name).symlink_to(ckroot/name,target_is_directory=True)
        filename='train_matrix3d.py' if a.action=='train' else 'evaluate_matrix3d.py'
        cmd=[sys.executable,str(code/filename),'--config',str(config),'--backbone',a.backbone,'--mode',a.mode,'--seed',str(a.seed)]
    else:
        if a.backbone!='fno' or a.seed!=2027: p.error('The published large3d experiment is FNO, seed 2027.')
        cfg=json.loads((ROOT/'large3d/configs/run.json').read_text())
        if a.epochs: cfg['epochs']=a.epochs
        if a.steps: cfg['steps_per_epoch']=a.steps
        (out/'configs/run.json').write_text(json.dumps(cfg,indent=2))
        for name in ('cases.json','stats.json'):
            (out/'configs'/name).write_bytes((ROOT/'large3d/configs'/name).read_bytes())
        (out/'configs/deadline.json').write_text(json.dumps({'deadline_epoch':time.time()+a.hours*3600}))
        if a.action=='generate': (out/'data').mkdir()
        else: (out/'data').symlink_to(asset/'data',target_is_directory=True)
        if a.action=='evaluate': (out/'checkpoints'/a.mode).symlink_to(ckroot/a.mode,target_is_directory=True)
        cmd=[sys.executable,str(ROOT/'scripts/launch_large.py'),str(code),str(out),a.action,a.mode]
    print('Output:',out,flush=True)
    subprocess.run(cmd,env=env,check=True,cwd=ROOT)

if __name__=='__main__': main()

