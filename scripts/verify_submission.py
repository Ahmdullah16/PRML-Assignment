"""Verify an extracted submission without the private training photographs.

From the ZIP's Code directory: python -m scripts.verify_submission --root ..
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
from prml import CLASSES
from prml.metrics import classification_metrics

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args(); root=a.root.resolve()
    hashes=json.loads((root/'Documentation/file_sha256.json').read_text())
    for file,digest in hashes.items():
        assert hashlib.sha256((root/file).read_bytes()).hexdigest()==digest,file
    results=root/'Results'
    manifest=json.loads((results/'split_manifest.json').read_text())
    summary=json.loads((results/'summary.json').read_text())
    digest=hashlib.sha256((results/'split_manifest.json').read_bytes()).hexdigest()
    assert tuple(manifest['labels'])==CLASSES
    assert digest==summary['split_manifest_sha256']
    parts=manifest['partitions']
    for key in ('file','pixel_sha256','duplicate_group'):
        sets=[{r[key] for r in parts[name]} for name in ('train','validation','test')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),key
    with (results/'Test_Predictions.csv').open(newline='') as f: predictions=list(csv.DictReader(f))
    checks=[]
    import torch
    from prml.inference import load_checkpoint,predict
    from prml.models import Generator
    torch.set_num_threads(2)
    image=io.BytesIO(); Image.new('RGB',(48,48),(128,128,128)).save(image,format='PNG')
    for kind in ('cnn','gan'):
        rows=[r for r in predictions if r['model']==kind]
        assert [r['file'] for r in rows]==[r['file'] for r in parts['test']]
        metric=classification_metrics([CLASSES.index(r['actual']) for r in rows],
                                      [CLASSES.index(r['predicted']) for r in rows])
        for key in ('accuracy','macro_precision','macro_recall','macro_f1','balanced_accuracy'):
            assert np.isclose(metric[key],summary['models'][kind]['test'][key]),(kind,key)
        assert metric['confusion_matrix']==summary['models'][kind]['test']['confusion_matrix']
        assert sum(map(sum,metric['confusion_matrix']))==len(parts['test'])
        probabilities=np.array([[float(r[f'p_{c}']) for c in CLASSES] for r in rows])
        assert np.isfinite(probabilities).all() and (probabilities>=0).all()
        assert np.allclose(probabilities.sum(1),1,atol=1e-6)
        assert np.array_equal(probabilities.argmax(1),[CLASSES.index(r['predicted']) for r in rows])
        net,meta=load_checkpoint(root/'Checkpoints'/f'{kind}.pt')
        assert meta['experiment']==summary['experiment'] and meta['split_manifest_sha256']==digest
        assert meta['best_epoch']==summary['models'][kind]['best_epoch']
        assert sum(p.numel() for p in net.parameters())==summary['models'][kind]['parameters']
        image.seek(0); scores=predict(net,kind,image)
        assert tuple(scores)==CLASSES and np.isclose(sum(scores.values()),1)
        checks.append(dict(model=kind,predictions=len(rows),metrics_verified=True,checkpoint_reload=True,four_probability_outputs=True))
    gen=torch.load(root/'Checkpoints/generator.pt',map_location='cpu',weights_only=True)
    disc=torch.load(root/'Checkpoints/gan.pt',map_location='cpu',weights_only=True)
    assert gen['best_epoch']==disc['best_epoch'] and gen['split_manifest_sha256']==digest
    g=Generator(); g.load_state_dict(gen['model']); g.eval()
    with torch.inference_mode(): generated=g(torch.randn(4,64),torch.arange(4))
    assert generated.shape==(4,3,64,64) and torch.isfinite(generated).all()
    assert (root/'RESULTS_REPORT.pdf').read_bytes().startswith(b'%PDF-')
    print(json.dumps(dict(status='passed',files_hash_verified=len(hashes),checks=checks,
                          paired_generator_verified=True,no_private_photos_required=True),indent=2))

if __name__=='__main__': main()
