"""Provision only verified, validation-selected classifiers for public inference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from prml.inference import load_checkpoint
from prml.train import write_json
from scripts.train_kaggle import EXPERIMENT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('results/runs/kaggle_seed42'))
    p.add_argument('--results',type=Path,default=Path('results/kaggle_seed42'))
    a=p.parse_args()
    s=json.loads((a.results/'summary.json').read_text())
    d=json.loads((a.results/'dataset.json').read_text())
    verification=json.loads((a.results/'verification.json').read_text())
    assert s['experiment']==EXPERIMENT and s['status']=='complete'
    assert verification['saved_predictions_recomputed'] and verification['sklearn_metrics_crosschecked']
    files={}
    for kind in ('cnn','gan'):
        source=a.run/f'{kind}.pt'; digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest==s['models'][kind]['checkpoint_sha256']
        model,meta=load_checkpoint(source)
        assert meta['experiment']==EXPERIMENT and meta['split_manifest_sha256']==s['split_manifest_sha256']
        target=Path('deployment')/source.name; target.parent.mkdir(exist_ok=True)
        shutil.copy2(source,target)
        files[kind]=dict(file=target.name,bytes=target.stat().st_size,sha256=digest,best_epoch=meta['best_epoch'])
    write_json('deployment/manifest.json',dict(experiment=EXPERIMENT,seed=42,dataset=d['dataset'],
        split_manifest_sha256=s['split_manifest_sha256'],files=files,
        note='New Kaggle experiment; paired generator stays private; classifier-only public inference; no source photographs'))
    history=Path('README_48_IMAGE_HISTORY.md')
    if not history.exists():
        history.write_text('# Archived 48-image project README\n\nThis is the pre-Kaggle README, preserved as historical documentation. Statements about current deployment below describe that earlier version.\n\n'+Path('README.md').read_text(encoding='utf-8'),encoding='utf-8')
    rows=[]
    for label,key in [('Accuracy','accuracy'),('Precision (Macro)','macro_precision'),('Recall (Macro)','macro_recall'),('F1-score (Macro)','macro_f1'),('Balanced accuracy','balanced_accuracy')]:
        rows.append('| '+label+' | '+' | '.join(f"{s['models'][k]['test'][key]:.4f}" for k in ('cnn','gan'))+' |')
    table='| Metric | CNN | AC-GAN classifier |\n|---|---:|---:|\n'+'\n'.join(rows)
    counts='| Expression | Train | Validation | Test |\n|---|---:|---:|---:|\n'+'\n'.join(
        '| '+c.title()+' | '+' | '.join(str(d['counts'][part][c]) for part in ('train','validation','test'))+' |'
        for c in ('happiness','sadness','fear','neutral'))
    readme=f'''# PRML Expression Lab - CNN versus AC-GAN

**Ahmdullah | Roll 264102307 | IIT Guwahati | M.Tech, Signal Processing and Machine Learning | Electronics and Electrical Engineering**

# Classification of Facial Expressions into Happiness, Sadness, Fear and Neutral Using CNN and GAN: A Comparative Study

The current experiment retrains both models from scratch on **dilkushsingh/facial-emotion-dataset**, using the same four-class partitions and validation-only checkpoint selection. GAN classification uses the discriminator auxiliary head; the generator synthesizes class-conditioned images. [Open Streamlit](https://facial-expression.streamlit.app/).

## Newly measured held-out test results

{table}

All **{d['totals']['test']:,} eligible supplied test images** are evaluated by each model. Precision, recall and F1 above are macro averages with undefined divisions assigned zero. Weighted and per-class metrics, raw probabilities and predictions are in [results/kaggle_seed42](results/kaggle_seed42/). These results belong to the new Kaggle experiment, separately from every historical 48-image study.

Read the [PDF report](reports/kaggle_seed42/RESULTS_REPORT.pdf), [Markdown report](reports/kaggle_seed42/RESULTS_REPORT.md), and [audit/provenance notes](reports/KAGGLE_AUDIT.md). [Independent arithmetic checks](results/kaggle_seed42/verification.json) cross-check scikit-learn and saved prediction CSVs.

## Dataset and splitting

{counts}

Totals: **{d['totals']['train']:,} training / {d['totals']['validation']:,} validation / {d['totals']['test']:,} test**. Keep original test membership; derive approximately 15% validation from eligible training data, stratified by class, seed 42, grouping duplicate candidates. Exclude training groups touching test, redundant exact training duplicates, and conflicting-label training groups. Audit: 604 exact-duplicate groups, 540 conservative near-duplicate pairs, 868 excluded training files. Within-test duplicates remain. No audited file/pixel-hash/duplicate group overlaps partitions. Near-duplicate heuristics are imperfect. **Subject IDs are unavailable; this is an image-level experiment, not subject-independent evaluation.**

The original local download contained 393 invalid training `.jpg` files holding HTTP 429 error text. They were recovered in a separate private copy from the dataset README's documented [upstream](https://github.com/dilkushsingh/Facial_Emotion_Dataset). All 25,053 selected file hashes now match that upstream archive byte-for-byte. Original files remain unchanged. The exact snapshot, repair history, upstream comparison and label mapping are recorded under results/kaggle_seed42. The observed current Kaggle API version is 17; the local download's version cannot be proven. Input SHA256s define the measured snapshot. Raw photographs are excluded from public GitHub and the assignment ZIP.

Preprocessing: EXIF orientation -> RGB -> bilinear 64x64 -> float32 [-1,1]. Random horizontal flips occur only during training for both models. Numeric order is sadness=0, happiness=1, neutral=2, fear=3; plots reorder both axes to Happiness, Sadness, Fear, Neutral.

## Architecture and training

The original architectures are reused unchanged. CNN: Conv/BN/ReLU/Pool blocks 16/32/64, dense 4096->64, ReLU, dropout .3 and four logits. AC-GAN: Gaussian noise 64 + class embedding 16, generator dense to 256x4x4 followed by 128/64/32/3-channel upsampling; discriminator four strided-convolution stages 32/64/128/256 with source and four-class heads.

CNN uses AdamW lr .001, decay .0001, ReduceLROnPlateau. GAN D uses AdamW with zero weight decay and G uses Adam, lr .0002, betas (.5,.999). D optimizes real/fake BCE and real/fake class CE; G optimizes non-saturating adversarial BCE plus class CE. Gradients are checked and norm-clipped at 10. Batch size 64. Budgets: 20 CNN / 15 GAN epochs, early stopping patience 5. Lowest validation classification CE selects checkpoints, pairing G and D at the same epoch. No test-driven selection.

CNN executed {s['models']['cnn']['epochs']} epochs, selected epoch {s['models']['cnn']['best_epoch']}, duration {s['models']['cnn']['training_seconds']:.1f}s. GAN executed {s['models']['gan']['epochs']} epochs, selected epoch {s['models']['gan']['best_epoch']}, duration {s['models']['gan']['training_seconds']:.1f}s. Hardware: Intel i5-1135G7 CPU, four PyTorch threads; no CUDA GPU. Full configuration and source hashes are in [Experiment_Configuration.json](results/kaggle_seed42/Experiment_Configuration.json).

## Run or reproduce

Use Python 3.12. The submission ZIP and `requirements-kaggle.txt` record the direct versions used in the executed environment; GPU is optional. Public inference retains the existing CPU-only cloud requirements under app/requirements.txt.

```bash
python -m pip install -r requirements-kaggle.txt
python -m scripts.prepare_kaggle --data "PATH/TO/DATA SET"
python -m scripts.train_kaggle --data "PATH/TO/DATA SET"
python -m scripts.evaluate_kaggle --data "PATH/TO/DATA SET"
python -m pytest -q
python -m streamlit run app/streamlit_app.py
python -m prml.predict --checkpoint deployment/cnn.pt --image "PATH/TO/IMAGE.jpg"
python -m prml.predict --checkpoint deployment/gan.pt --image "PATH/TO/IMAGE.jpg"
```

Use the exact hash-verified repaired snapshot and manifest for measured-run reproduction. scripts/repair_kaggle.py documents recovery into a private copy, using the original audit and upstream archive. New input snapshots must use separately named results/run directories. Training resumes per-epoch state and preserves completed model runs. Evaluation requires completion flags for both models before final test access.

The seven dashboard pages default to the new experiment. Historical studies are separately available through the evidence selector. Prediction Studio uses the new packaged classifiers, matching training preprocessing and labels, and visibly disables inference if weights are missing. Uploads stay in memory; no models train or data download at startup. Checkpoint loading uses CPU mapping, weights_only=True and strict metadata/state checks. Public classifier hashes are in deployment/manifest.json. Generator weights, optimizer resume states and generated sample grids remain private.

## Historical 48-image studies

The [archived README](README_48_IMAGE_HISTORY.md) preserves earlier descriptions. Original [LOSO results](results/corrected/aggregate.json), [per-class 6/3/3 reports](reports/class_6_3_3_seed42/EXPERIMENT_REPORT.md), [subject-holdout reports](reports/seed42/EXPERIMENT_REPORT.md), notebooks and recovered baselines remain unchanged. Their scores are not used as new Kaggle results. Earlier public weights remain in Git history.

## Interpretation and limitations

Predicted facial-expression labels do not establish a person's actual emotional or mental state. Softmax scores are uncalibrated. A single seed, unknown subject overlap, within-test duplicates, heuristic deduplication, class imbalance, annotation noise, different architectures and unequal adversarial compute limit the comparison. Neither generative realism, memorization risk, demographic fairness nor real-world deployment fitness is established. No accuracy change relative to the 48-image studies can be attributed to a single factor.

Dataset attribution: Dilkush Singh, [Facial Emotion Dataset](https://www.kaggle.com/datasets/dilkushsingh/facial-emotion-dataset), local metadata CC BY 4.0. CNN, GAN, AC-GAN and PyTorch references appear in the report. Raw faces are not redistributed.
'''
    Path('README.md').write_text(readme,encoding='utf-8')
    print('Verified Kaggle classifiers provisioned; current README updated; historical README preserved',flush=True)

if __name__=='__main__': main()
