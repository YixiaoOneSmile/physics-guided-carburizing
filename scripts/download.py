"""Download verified release archives and safely extract without overwriting files."""
from pathlib import Path
import argparse,hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',choices=['one_d','small3d','large3d','all'],required=True)
    p.add_argument('--archive-dir',type=Path,default=ROOT/'downloads')
    p.add_argument('--assets',type=Path,default=ROOT/'assets')
    p.add_argument('--offline',action='store_true',help='Use manually downloaded archives only')
    a=p.parse_args();manifest=json.loads((ROOT/'assets_manifest.json').read_text())
    a.archive_dir.mkdir(parents=True,exist_ok=True)
    for item in manifest['archives']:
        if a.study!='all' and item['study']!=a.study:continue
        archive=a.archive_dir/item['name']
        if not archive.exists():
            if item.get('parts'):
                parts=[]
                for part in item['parts']:
                    local=a.archive_dir/part['name']
                    if not local.exists():
                        if a.offline:raise FileNotFoundError(local)
                        if not part.get('drive_id'):raise RuntimeError('Part not yet published: '+part['name'])
                        import gdown
                        gdown.download(id=part['drive_id'],output=str(local),quiet=False,resume=True)
                    if digest(local)!=part['sha256']:raise RuntimeError('Checksum mismatch: '+str(local))
                    parts.append(local)
                partial=archive.with_suffix(archive.suffix+'.assembling')
                with partial.open('wb') as dst:
                    for part in parts:
                        with part.open('rb') as src:shutil.copyfileobj(src,dst)
                if digest(partial)!=item['sha256']:raise RuntimeError('Assembled archive checksum mismatch')
                partial.replace(archive)
            else:
                if a.offline:raise FileNotFoundError(archive)
                if not item.get('drive_id'):raise RuntimeError('Archive not yet published: '+item['name'])
                import gdown
                gdown.download(id=item['drive_id'],output=str(archive),quiet=False,resume=True)
        if digest(archive)!=item['sha256']:raise RuntimeError('Checksum mismatch: '+str(archive))
        with tarfile.open(archive,'r:gz') as tf:
            for member in tf:
                target=(a.assets/member.name).resolve()
                if not target.is_relative_to(a.assets.resolve()) or not member.isfile():
                    raise RuntimeError('Unsafe archive entry: '+member.name)
                if target.exists():
                    expected=manifest['files'].get(member.name)
                    if expected and digest(target)==expected['sha256']:continue
                    raise FileExistsError('Refusing to overwrite: '+str(target))
                target.parent.mkdir(parents=True,exist_ok=True)
                with tf.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        print('Verified and extracted',item['name'])
if __name__=='__main__':main()
