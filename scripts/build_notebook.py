"""Build a reproducible, executable course notebook with measured split-specific outputs."""
from pathlib import Path
import nbformat as nbf


def main(strategy="subject"):
    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md("# IIT Guwahati — PRML Facial Expression Recognition\n## CNN versus AC-GAN: seed-42 shuffled train / validation / test study\n"
           "**Ahmdullah · M.Tech, Signal Processing and Machine Learning · Electronics and Electrical Engineering**\n\n"
           "This executable notebook uses the supplied private ZIP, audits the images, saves a subject-independent split, "
           "trains both models, selects checkpoints using validation only, and reports all three partitions separately. "
           "Original photographs and generated faces are not embedded in notebook outputs."),
        md("## 1. Design and reproducibility\nThe requested ratio is **50:30:20**. Six identities cannot be divided exactly in that ratio. "
           "Largest-remainder allocation gives **3 train / 2 validation / 1 test subjects**, or **24/16/8 images** "
           "(**50%/33.33%/16.67%**). Identity separation takes priority over an image-level ratio that would leak people. "
           "Seed **42** shuffles subjects and rows; training minibatches shuffle each epoch. Validation and test are fixed.\n\n"
           "This is a separate holdout experiment from the original six-fold LOSO study. These same images have been examined "
           "in earlier experiments, so the holdout is not a new external dataset. No parameter or epoch budget is selected on test labels."),
        md("## 2. Environment\nRun this notebook from the downloaded project repository (or its `notebooks/` folder). "
           "A notebook downloaded alone requires the repository modules. Use Python **3.12**.\n\n```bash\n"
           "git clone https://github.com/Ahmdullah16/PRML-Assignment.git\ncd PRML-Assignment\n"
           "python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu\n"
           "python -m pip install -r requirements-dev.txt\npython -m pip install nbformat nbclient ipykernel\n```\n"
           "Training is CPU compatible. The public Streamlit app performs inference only; it never runs this training notebook at startup."),
        code("%matplotlib inline\nimport os, sys, json, hashlib, platform\nfrom pathlib import Path\n"
             "candidates = [Path.cwd(), Path.cwd().parent, Path.cwd() / 'PRML-Assignment']\n"
             "ROOT = next((p.resolve() for p in candidates if (p / 'prml/holdout.py').exists()), None)\n"
             "if ROOT is None:\n    raise FileNotFoundError('Open this notebook inside the PRML-Assignment repository; see setup instructions above.')\n"
             "os.chdir(ROOT)\nsys.path.insert(0, str(ROOT))\n"
             "import numpy as np\nimport pandas as pd\nimport torch\nimport matplotlib.pyplot as plt\n"
             "from IPython.display import display, Markdown, Image as DisplayImage\n"
             "from prml import CLASSES, SIZE, LATENT\nfrom prml.holdout import prepare_dataset, run_experiment, export_experiment, verify_checkpoints, SPLIT_NAMES\n"
             "from prml.models import CNN, Generator, Discriminator\nfrom prml.train import seed_everything, discriminator_loss, generator_loss\n"
             "from prml.inference import load_checkpoint, predict\nfrom scripts.report_holdout import build_reports\n"
             "torch.set_num_threads(4)\nseed_everything(42)\n"
             "display(pd.DataFrame([{'Python': platform.python_version(), 'PyTorch': str(torch.__version__), 'NumPy': np.__version__, 'seed': 42, 'device': 'cpu'}]))"),
        md("## 3. Private dataset ZIP\nThe default below is the exact local path provided for this assignment. "
           "Set `PRML_DATA_ZIP` for another machine. Extraction admits only recognized class images and rejects path traversal "
           "and duplicate archive members. Existing different files are never silently overwritten. The raw images stay in ignored `data/`."),
        code("DATA_ZIP = Path(os.environ.get('PRML_DATA_ZIP', r'C:\\Users\\DELL\\Downloads\\ML LAB MINI PROJECT\\PRML\\PRML-Assignment\\data\\raw\\PRML Assignment Data.zip'))\n"
             "DATA_DIR = ROOT / 'data/seed42_zip'\nRUN_DIR = ROOT / 'results/runs/holdout_seed42'\n"
             "PUBLIC_RESULTS = ROOT / 'results/holdout_seed42'\n"
             "if not DATA_ZIP.is_file():\n    raise FileNotFoundError('Private ZIP not found; edit DATA_ZIP or set PRML_DATA_ZIP.')\n"
             "rows, parts, split_manifest = prepare_dataset(DATA_ZIP, DATA_DIR, seed=42)\n"
             "print('Validated images:', len(rows), '| Subjects:', len({r['subject'] for r in rows}))\n"
             "print('ZIP SHA256:', hashlib.sha256(DATA_ZIP.read_bytes()).hexdigest())\n"
             "display(pd.DataFrame([{'label': i, 'expression': c, 'images': sum(r['class'] == c for r in rows)} for i, c in enumerate(CLASSES)]))"),
        md("## 4. Dataset audit\nEvery image is decoded and checked against filename/folder labels. "
           "Exact decoded duplicates fail validation. SHA256 records bind the input files. Perceptual hashes flag review candidates, "
           "not proof of duplicate identity. Previously reviewed cross-subject matches are accepted only if their file hashes match."),
        code("from prml.data import audit\naudit_result = audit(rows)\n"
             "display(pd.DataFrame([{'original_dimensions': k, 'images': v} for k, v in audit_result['dimensions'].items()]))\n"
             "print('Near-duplicate candidates:', len(audit_result['near_duplicates']))\n"
             "print('Cross-subject candidates:', sum(not p['same_subject'] for p in audit_result['near_duplicates']))\n"
             "fig, ax = plt.subplots(figsize=(6, 3))\nax.bar(CLASSES, [audit_result['counts'][c] for c in CLASSES], color='#167E87')\n"
             "ax.set(title='Audited original class balance', ylabel='Images'); display(fig); plt.close(fig)"),
        md("## 5. Seed-42 shuffle and subject-independent partitions\nThe shuffled subject order and complete records are saved "
           "in `split_manifest.json`. All four classes occur in every partition. Both models use exactly the same membership. "
           "Training uses random horizontal flips; final train/validation/test metrics all use unaugmented images."),
        code("print('Shuffled subject order:', split_manifest['shuffled_subject_order'])\n"
             "display(pd.DataFrame([{'partition': name, 'subjects': ', '.join(p['subjects']), 'images': p['images'], 'actual_ratio': p['actual_ratio'], **p['class_counts']}\n"
             "                      for name, p in split_manifest['partitions'].items()]))\n"
             "subject_sets = [{r['subject'] for r in parts[name]} for name in SPLIT_NAMES]\n"
             "assert not (subject_sets[0] & subject_sets[1] or subject_sets[0] & subject_sets[2] or subject_sets[1] & subject_sets[2])\n"
             "assert [len(parts[n]) for n in SPLIT_NAMES] == [24, 16, 8]\nprint('No subject overlap. Total records:', sum(map(len, parts.values())))"),
        md("## 6. Preprocessing and CNN\nEXIF orientation → RGB → bilinear 64×64 resize → pixels in [−1,1]. "
           "The checkpoint-compatible CNN has conv/BN/ReLU/max-pool blocks at 16,32,64 channels; a 4096→64→4 dense head; dropout 0.3. "
           "The last layer emits logits; softmax is used for inference.\n\n"
           r"$$Y_{i,j,k}=\sum_{m,n,c} W_{m,n,c,k}X_{i+m,j+n,c}+b_k$$" "\n\n"
           r"$$p(y=k|x)=\frac{e^{z_k}}{\sum_{j=1}^{4}e^{z_j}},\qquad L_{CNN}=-\log p(y|x).$$"),
        code("cnn = CNN()\ndiscriminator = Discriminator()\ngenerator = Generator()\n"
             "display(pd.DataFrame([{'network': name, 'parameters': sum(p.numel() for p in net.parameters())}\n"
             "                      for name, net in [('CNN', cnn), ('Discriminator', discriminator), ('Generator', generator)]]))\n"
             "x = torch.zeros(2, 3, SIZE, SIZE)\nassert cnn(x).shape == (2, 4)\n"
             "fake = generator(torch.randn(2, LATENT), torch.tensor([0, 3]))\nsource_logits, class_logits = discriminator(fake)\n"
             "assert fake.shape == (2, 3, 64, 64)\nassert source_logits.shape == (2,) and class_logits.shape == (2, 4)\n"
             "print('CNN, generator and discriminator tensor dimensions verified.')\ndel cnn, discriminator, generator, fake"),
        md("## 7. AC-GAN objective and correct gradient flow\nThe class-conditioned generator takes Gaussian noise and a label embedding. "
           "The discriminator emits source and expression logits; **its expression head performs classification**.\n\n"
           r"$$L_D=BCE(D_s(x),1)+BCE(D_s(G(z,c)),0)+CE(D_c(x),c)+CE(D_c(G(z,c)),c)$$" "\n\n"
           r"$$L_G=BCE(D_s(G(z,c)),1)+CE(D_c(G(z,c)),c).$$" "\n\n"
           "All terms have unit weights. The generator source term is the explicitly non-saturating variant. "
           "Fake tensors detach for discriminator updates; discriminator weights freeze while retaining input gradients for generator updates. "
           "The generator checkpoint is saved at the same validation-selected epoch as the discriminator."),
        md("## 8. Fixed training and validation protocol\nCNN: **25 epochs**. AC-GAN: **40 epochs**. Batch **8**, Adam lr **0.0002**. "
           "CNN betas=(0.9,0.999); GAN betas=(0.5,0.999). Seed **42** resets before each method. "
           "The best checkpoint minimizes validation classification cross-entropy. Test images do not enter training, "
           "augmentation, preprocessing fitting or model selection. No search is performed against test performance.\n\n"
           "The next cell trains when no completed run exists. A matching completed run is replay-verified and reused, "
           "rather than silently retrained. To intentionally repeat training, use a new `RUN_DIR`; keep the split and budgets fixed."),
        code("reusing = (RUN_DIR / 'summary.json').exists()\n"
             "summary = run_experiment(DATA_ZIP, DATA_DIR, RUN_DIR, ROOT / 'results/duplicate_review.json',\n"
             "                         seed=42, cnn_epochs=25, gan_epochs=40, device='cpu')\n"
             "print('Replayed and reused matching completed experiment.' if reusing else 'Executed new experiment and replay-verified saved checkpoints.')\n"
             "display(pd.DataFrame([{'model': k, 'selected_epoch': v['best_epoch'], 'training_seconds': v['training_seconds']} for k, v in summary['models'].items()]))"),
        md("## 9. Separate final training report\nTraining metrics below are measured on unaugmented training images using "
           "the frozen selected checkpoint. They describe fit to seen subjects, not out-of-sample generalization."),
        code("metric_keys = ['loss', 'accuracy', 'balanced_accuracy', 'macro_precision', 'macro_recall', 'macro_f1']\n"
             "def partition_table(name):\n    return pd.DataFrame({k.upper(): {m: v['partitions'][name][m] for m in metric_keys} for k, v in summary['models'].items()})\n"
             "display(partition_table('train'))"),
        md("## 10. Separate final validation report\nThese subjects select the checkpoint. Validation performance is "
           "selection-biased and must not be labeled test accuracy."),
        code("display(partition_table('validation'))"),
        md("## 11. Separate frozen-model test report\nThis is one held-out subject and eight images. "
           "A single changed prediction moves accuracy by **12.5 percentage points**. The test set stays fixed and is not "
           "used to choose a seed, epoch budget, learning rate or public classifier."),
        code("display(partition_table('test'))\n"
             "display(pd.DataFrame([{'model': kind.upper(), 'class': cls, **{m: model['partitions']['test'][m + '_by_class'][cls] for m in ['precision', 'recall', 'f1']}}\n"
             "                      for kind, model in summary['models'].items() for cls in CLASSES]))"),
        md("## 12. Learning curves and confusion matrices\nValidation CE selects the checkpoint. "
           "Loss magnitudes differ across objectives, so compare classification metrics rather than raw CNN/GAN losses."),
        code("export_experiment(RUN_DIR, PUBLIC_RESULTS, ROOT / 'deployment')\n"
             "build_reports(PUBLIC_RESULTS, ROOT / 'reports/seed42')\n"
             "display(DisplayImage(filename=str(PUBLIC_RESULTS / 'figures/split_accuracy.png')))\n"
             "for kind in ['cnn', 'gan']:\n    display(DisplayImage(filename=str(PUBLIC_RESULTS / f'figures/{kind}_history.png')))"),
        code("for split in SPLIT_NAMES:\n    for kind in ['cnn', 'gan']:\n"
             "        display(DisplayImage(filename=str(PUBLIC_RESULTS / f'figures/{split}_{kind}_confusion.png')))"),
        md("## 13. Per-image prediction evidence and checkpoint replay\nThe following table contains codes and probabilities, "
           "not source photographs. Both saved classifiers are replayed on all three partitions. Replaying verifies saved outputs; "
           "it does not change the checkpoint or tune the model."),
        code("prediction_rows = []\nfor kind, model in summary['models'].items():\n"
             "    for split, measured in model['partitions'].items():\n        for p in measured['predictions']:\n"
             "            prediction_rows.append({'model': kind, 'partition': split, **p})\n"
             "prediction_frame = pd.DataFrame(prediction_rows)\ndisplay(prediction_frame[prediction_frame['partition'] == 'test'])\n"
             "verify_checkpoints(parts, RUN_DIR, summary)\ndisplay(pd.DataFrame(json.loads((RUN_DIR / 'verification.json').read_text())['checks']))"),
        md("## 14. Inference example without publishing a face\nUse a locally supplied authorized image for a real demonstration. "
           "This notebook uses an in-memory gray test image to verify the inference API only; the result is not a face-recognition claim. "
           "No image is sent to a server from this cell. Softmax scores are uncalibrated."),
        code("import io\nfrom PIL import Image\nbuffer = io.BytesIO()\nImage.new('RGB', (64, 64), (128, 128, 128)).save(buffer, format='PNG')\n"
             "for kind in ['cnn', 'gan']:\n    model, metadata = load_checkpoint(RUN_DIR / f'{kind}.pt')\n    buffer.seek(0)\n"
             "    scores = predict(model, kind, buffer)\n    assert abs(sum(scores.values()) - 1) < 1e-6\n    print(kind, scores)"),
        md("## 15. Interpretation and limitations\nRead the executed results above: high training fit does not establish generalization. "
           "Compare validation and test separately, and preserve negative results. Chance-level expected accuracy is 25% for four balanced "
           "classes, but a constant predictor also obtains 25%. One test identity cannot support a reliable population confidence interval. "
           "Repeated seeds, external subjects, larger licensed datasets, calibration and architecture-matched ablations are future work. "
           "No synthetic realism/diversity claim is made. Classification describes apparent expression labels, not actual emotions or mental state."),
        md("## 16. Deliverables, privacy and deployment\nSeparate Markdown reports are under `reports/seed42/`; metrics, split manifests, "
           "histories and statistical figures are under `results/holdout_seed42/`. The two validation-selected classifier checkpoints "
           "are packaged for the requested public inference. Original photographs, ZIP and generator weights remain ignored/private. "
           "The public application never retrains on startup and handles uploads in memory.\n\n"
           "```bash\npython -m pytest -q\npython -m streamlit run app/streamlit_app.py\n```"),
        md("## 17. References\n1. LeCun et al. (1998), [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h).\n"
           "2. Goodfellow et al. (2014), [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).\n"
           "3. Odena, Olah and Shlens (2017), [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585).\n"
           "4. [Project repository](https://github.com/Ahmdullah16/PRML-Assignment).")
    ]
    if strategy == "class_6_3_3":
        nb.cells[0] = md("# IIT Guwahati — PRML Facial Expression Recognition\n## CNN versus AC-GAN: per-class 6/3/3 split, seed 42\n"
            "**Ahmdullah · M.Tech, Signal Processing and Machine Learning · Electronics and Electrical Engineering**\n\n"
            "This notebook implements the corrected request: shuffle each class's 12 images separately; select 6 for training, "
            "3 for validation, 3 for test; then shuffle the three combined partitions separately. Original photographs and generated faces are not embedded.")
        nb.cells[1] = md("## 1. Exact requested allocation\n| Class | Train | Validation | Test |\n|---|---:|---:|---:|\n"
            "| Happiness | 6 | 3 | 3 |\n| Sadness | 6 | 3 | 3 |\n| Fear | 6 | 3 | 3 |\n| Neutral | 6 | 3 | 3 |\n"
            "| **Total** | **24** | **12** | **12** |\n\n"
            "This is **50%/25%/25%**, exactly the corrected 6/3/3 instruction. Each class is shuffled before allocation. "
            "Combined train, validation and test image orders are then shuffled independently. Seed 42 controls every stream. "
            "There is no image overlap, but people can appear in more than one partition. Therefore this is image-level evaluation, "
            "not unseen-subject evaluation. The earlier subject-independent studies remain separate baselines.")
        nb.cells[5].source = nb.cells[5].source.replace("results/runs/holdout_seed42", "results/runs/class_6_3_3_seed42").replace(
            "results/holdout_seed42", "results/class_6_3_3_seed42").replace(
            "prepare_dataset(DATA_ZIP, DATA_DIR, seed=42)", "prepare_dataset(DATA_ZIP, DATA_DIR, seed=42, strategy='class_6_3_3')")
        nb.cells[8] = md("## 5. Shuffle each class, split 6/3/3, then mix each partition\n"
            "`SeedSequence(42)` spawns four independent RNG streams for the four classes and three more streams for the combined partitions. "
            "The exact per-class shuffled order, final partition order and subject overlap audit are saved. "
            "Training batches additionally reshuffle each epoch. Validation/test images receive no random augmentation.")
        nb.cells[9] = code("display(pd.DataFrame(split_manifest['per_class_shuffled_order']))\n"
            "display(pd.DataFrame([{'partition': name, 'images': p['images'], 'share': p['actual_ratio'], **p['class_counts']}\n"
            "                      for name, p in split_manifest['partitions'].items()]))\n"
            "image_sets = [{r['filename'] for r in parts[name]} for name in SPLIT_NAMES]\n"
            "assert not (image_sets[0] & image_sets[1] or image_sets[0] & image_sets[2] or image_sets[1] & image_sets[2])\n"
            "assert len(set.union(*image_sets)) == 48\n"
            "for name, count in zip(SPLIT_NAMES, [6, 3, 3]):\n    assert all(sum(r['class'] == cls for r in parts[name]) == count for cls in CLASSES)\n"
            "print('No repeated images across splits; exact 6/3/3 per class verified.')\n"
            "print('Subject overlap audit:', split_manifest['subject_overlap'])")
        nb.cells[14].source = nb.cells[14].source.replace("gan_epochs=40, device='cpu')", "gan_epochs=40, device='cpu', strategy='class_6_3_3')")
        nb.cells[19] = md("## 11. Separate frozen-model test report\nThe test contains **12 images, exactly 3 per class**. "
            "Each changed prediction moves accuracy by **8.33 percentage points**. No test image enters training, augmentation or checkpoint selection. "
            "However, shared subject identities and similar a/b photographs mean the scores cannot establish unseen-person recognition. "
            "The test set does not choose the seed, epoch budget, learning rate or public model.")
        nb.cells[22].source = nb.cells[22].source.replace("reports/seed42", "reports/class_6_3_3_seed42")
        nb.cells[28] = md("## 15. Interpretation and limitations\nTraining fit, validation selection and held-out-image test performance "
            "are reported separately. This exact class-stratified protocol allows subject overlap, which can make scores optimistic for unseen people. "
            "The earlier LOSO results provide the separate subject-independent comparison. Uniform guessing has expected accuracy 25%; a constant "
            "predictor also gets 25% on balanced classes. Only six people are represented, so the 48 images are not 48 independent people. "
            "No population confidence interval or universal CNN-versus-GAN conclusion is claimed. Outputs describe apparent expression categories, "
            "not actual emotions or mental state. A larger licensed dataset, external subjects, multiple seeds and calibration remain future work.")
        nb.cells[29] = md("## 16. Deliverables and deployment\nSeparate reports are in `reports/class_6_3_3_seed42/`; split records, probabilities, "
            "histories and statistical figures are in `results/class_6_3_3_seed42/`. Only the two validation-selected classifiers are packaged for "
            "the authorized public prediction. The source ZIP, original photographs and generator remain private. The public app never trains at startup.\n\n"
            "```bash\npython -m pytest -q\npython -m streamlit run app/streamlit_app.py\n```")
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (PRML)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}}
    path = Path("notebooks/PRML_Class_6_3_3_Seed42.ipynb" if strategy == "class_6_3_3" else "notebooks/PRML_Seed42_CNN_ACGAN.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"Built {path}: {len(nb.cells)} cells")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["subject", "class_6_3_3"], default="class_6_3_3")
    main(p.parse_args().strategy)
