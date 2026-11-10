"""Repair documented invalid JPEG payloads in a private copy using upstream ZIP.

Run prepare_kaggle first. Original downloaded files remain unchanged.
"""
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=Path('results/kaggle_seed42/dataset.json'))
    a=p.parse_args()
    previous=json.loads(a.audit.read_text())
    if a.destination.exists(): raise FileExistsError('Preserve existing private data copy')
    shutil.copytree(a.source,a.destination)
    repairs=[]
    with zipfile.ZipFile(a.archive) as z:
        members={name.split('/',1)[1]:name for name in z.namelist() if '/' in name}
        for row in previous['corrupt']:
            data=z.read(members[row['file']])
            Image.open(io.BytesIO(data)).verify()
            target=a.destination/row['file']
            repairs.append(dict(file=row['file'],original_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                repaired_sha256=hashlib.sha256(data).hexdigest(),source='dilkushsingh/Facial_Emotion_Dataset'))
            target.write_bytes(data)
    a.audit.with_name('original_download_audit.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
    a.audit.with_name('repair_provenance.json').write_text(json.dumps(dict(
        upstream_archive_sha256=hashlib.sha256(a.archive.read_bytes()).hexdigest(),repairs=repairs),indent=2),encoding='utf-8')
    print(f'Repaired {len(repairs)} invalid training images in private copy',flush=True)

if __name__=='__main__': main()
