import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scripts.prepare_kaggle import prepare
from prml import CLASSES
from prml.metrics import classification_metrics

ROOT=Path(__file__).resolve().parents[1]

def test_kaggle_duplicates_preserve_test_and_split_groups(tmp_path):
    rng=np.random.default_rng(1234)
    for split in ('train_dir','test_dir'):
        for name in ('happy','sad','fear','neutral','angry'):
            folder=tmp_path/split/name; folder.mkdir(parents=True)
            for i in range(12 if split=='train_dir' else 2):
                Image.fromarray(rng.integers(0,256,(48,48,3),dtype=np.uint8)).save(folder/f'{i}.png')
    duplicate=tmp_path/'train_dir/happy/duplicate.png'
    duplicate.write_bytes((tmp_path/'test_dir/happy/0.png').read_bytes())
    (tmp_path/'train_dir/sad/corrupt.jpg').write_text('429: error text')
    parts,stats=prepare(tmp_path,tmp_path/'audit')
    assert len(parts['test'])==8 and len(stats['corrupt'])==1
    assert len(stats['excluded_training'])==1
    assert all({r['label'] for r in rows}==set(range(4)) for rows in parts.values())
    sets=[{r['duplicate_group'] for r in rows} for rows in parts.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    assert all('angry' not in r['file'] for rows in parts.values() for r in rows)
    assert not any(r['file'].endswith('duplicate.png') for rows in parts.values() for r in rows)

def test_measured_kaggle_manifest_and_metrics():
    out=ROOT/'results/kaggle_seed42'
    manifest=json.loads((out/'split_manifest.json').read_text())
    assert tuple(manifest['labels'])==CLASSES
    parts=manifest['partitions']
    for key in ('file','pixel_sha256','duplicate_group'):
        sets=[{r[key] for r in parts[p]} for p in ('train','validation','test')]
        assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    stats=json.loads((out/'dataset.json').read_text())
    assert not stats['corrupt']
    assert stats['totals']=={k:len(v) for k,v in parts.items()}
    if not (out/'summary.json').exists():
        return # Preparation can be verified while the first training run is ongoing.
    summary=json.loads((out/'summary.json').read_text())
    assert summary['split_manifest_sha256']==hashlib.sha256((out/'split_manifest.json').read_bytes()).hexdigest()
    with (out/'Test_Predictions.csv').open(newline='') as f: predictions=list(csv.DictReader(f))
    for kind in ('cnn','gan'):
        rows=[r for r in predictions if r['model']==kind]
        assert [r['file'] for r in rows]==[r['file'] for r in parts['test']]
        metric=classification_metrics([CLASSES.index(r['actual']) for r in rows],[CLASSES.index(r['predicted']) for r in rows])
        for key in ('accuracy','macro_precision','macro_recall','macro_f1','balanced_accuracy'):
            assert np.isclose(metric[key],summary['models'][kind]['test'][key])
        assert metric['confusion_matrix']==summary['models'][kind]['test']['confusion_matrix']
        assert sum(map(sum,metric['confusion_matrix']))==len(parts['test'])
        for r in rows:
            probs=[float(r[f'p_{c}']) for c in CLASSES]
            assert np.isfinite(probs).all() and np.isclose(sum(probs),1)
            assert CLASSES[np.argmax(probs)]==r['predicted']
