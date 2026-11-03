# PRML Expression Lab — CNN versus AC-GAN

**Ahmdullah · IIT Guwahati · M.Tech, Signal Processing and Machine Learning · Electronics and Electrical Engineering**

A tested PyTorch study of **happiness, sadness, fear and neutral** using 48 original photographs from six subjects. A supervised CNN is compared with the **expression head of an AC-GAN discriminator**, using identical subject-disjoint test/validation partitions. The Streamlit research dashboard works without private images or checkpoints; prediction is enabled only when trusted weights are available locally.

**[Open the research dashboard](https://facial-expression.streamlit.app/)** · seed-42 classifier inference and research reports.

## Current: shuffle each class and split 6 / 3 / 3

The corrected request uses all **12 images in each class**: shuffle that class with seed 42, assign **6 training / 3 validation / 3 test**, then independently shuffle each combined partition. Totals: **24 train / 12 validation / 12 test (50% / 25% / 25%)**. Training minibatches shuffle again each epoch. Exact per-class permutations and final file orders are saved in `results/class_6_3_3_seed42/split_manifest.json`.

| Class | Training | Validation | Test |
|---|---:|---:|---:|
| Happiness | 6 | 3 | 3 |
| Sadness | 6 | 3 | 3 |
| Fear | 6 | 3 | 3 |
| Neutral | 6 | 3 | 3 |
| **Total** | **24** | **12** | **12** |

Open the [executed 6/3/3 notebook](notebooks/PRML_Class_6_3_3_Seed42.ipynb). Separate [training](reports/class_6_3_3_seed42/TRAIN_REPORT.md), [validation](reports/class_6_3_3_seed42/VALIDATION_REPORT.md), [test](reports/class_6_3_3_seed42/TEST_REPORT.md) and [experiment](reports/class_6_3_3_seed42/EXPERIMENT_REPORT.md) reports document this exact protocol.

| Corrected partition | CNN accuracy | AC-GAN accuracy |
|---|---:|---:|
| Training | 100.00% | 50.00% |
| Validation | 41.67% | 25.00% |
| Test | 41.67% | 25.00% |

Images do not overlap across sets. **Subject identities do overlap**, as recorded in the manifest. These are held-out-image scores, not unseen-person scores. Previous subject-independent LOSO and subject-holdout results are preserved separately. Models select epochs using validation CE only, with fixed 25 CNN / 40 GAN epoch budgets and seed 42.

```bash
python -m scripts.run_holdout --zip "PATH/TO/PRML Assignment Data.zip" --strategy class_6_3_3 --output results/runs/class_6_3_3_seed42 --public-output results/class_6_3_3_seed42 --export
python -m scripts.report_class_split
```

The public classifiers in `deployment/` use this corrected split. Their source ZIP and generator remain private. No training is performed during app startup.

## Previous: seed-42 subject-holdout notebook and reports

This section records the earlier protocol. Its notebook and reports remain available for comparison; current public prediction uses the corrected 6/3/3 image split above.

Open [the executed project notebook](notebooks/PRML_Seed42_CNN_ACGAN.ipynb). Its **13 code cells executed without errors** using the supplied private ZIP. Install `requirements-notebook.txt` for the notebook environment. The default ZIP path is the user-provided Windows path; use `PRML_DATA_ZIP` to override it elsewhere.

The requested **50:30:20** ratio is rounded across six complete subjects to **3:2:1 subjects**, i.e. **24/16/8 images (50%/33.33%/16.67%)**. Seed 42 produces subject order `116, 079, 168, 140, 066, 004`: the first three train, the next two validate, and `004` tests. Rows and training minibatches also shuffle reproducibly. There is no subject overlap and no validation/test augmentation.

| Seed-42 partition | CNN accuracy | AC-GAN accuracy |
|---|---:|---:|
| Training (24 images, seen subjects) | 100.00% | 25.00% |
| Validation (16 images, checkpoint selection) | 81.25% | 25.00% |
| Test (8 images, held-out subject) | 37.50% | 25.00% |

Separate [training](reports/seed42/TRAIN_REPORT.md), [validation](reports/seed42/VALIDATION_REPORT.md), [test](reports/seed42/TEST_REPORT.md) and [experiment](reports/seed42/EXPERIMENT_REPORT.md) reports include class-wise precision/recall/F1, loss and confusion matrices. Test macro F1 is **0.2429 CNN** and **0.1000 AC-GAN**. The train–test gap limits generalization; validation scores are selection-biased. These images were examined in the prior LOSO study, so this split is not a new external test dataset.

```bash
python -m pip install -r requirements-notebook.txt
python -m scripts.run_holdout --zip "PATH/TO/PRML Assignment Data.zip" --export
python -m scripts.report_holdout
```

Completed matching runs are verified and reused, not retrained silently. All six model/partition evaluations replay from saved checkpoints. Only the two validation-selected **classifier** checkpoints are packaged under `deployment/` for the requested public prediction; the GAN generator, ZIP and original photographs remain excluded. The deployed app never trains at startup. Statistical evidence from this new split is kept separate from the historical and corrected LOSO results below.

![Corrected experiment comparison](results/figures/corrected/comparison.png)

## Measured evidence

| Experiment / aggregation | CNN | AC-GAN classifier |
|---|---:|---:|
| Corrected pooled accuracy | 39.58% (19/48) | 33.33% (16/48) |
| Corrected pooled balanced accuracy | 39.58% | 33.33% |
| Corrected pooled macro precision | 0.3957 | 0.3304 |
| Corrected pooled macro recall | 0.3958 | 0.3333 |
| Corrected pooled macro F1 | 0.3927 | 0.3288 |
| Corrected mean **fold** macro F1 | 0.2784 | 0.1917 |
| Historical replay-verified accuracy | 33.33% | 25.00% |
| Historical mean fold macro F1 | 0.2011 | 0.1000 |

Six people are the sampling units; 48 photographs are not 48 independent people. Uniform guessing has expected accuracy 25%. The CNN's three-image advantage is descriptive, not a statistically established or universal result. These models estimate appearance labels, **not actual emotions or mental state**. Scores are uncalibrated and the models have limited generalization evidence.

## Run locally

Use **Python 3.12**. CPU training/inference is supported; CUDA is optional. Direct dependency versions are pinned; transitive versions can vary. The measured run used PyTorch 2.14.1+cpu, NumPy 2.2.6 and four CPU threads. Hardware/software and source hashes are in `results/corrected/run.json`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m streamlit run app/streamlit_app.py
```

The app never retrains during startup. Its seven pages are Overview, Dataset Explorer, CNN Explorer, GAN Explorer, Model Comparison, Prediction Studio and Research Report. Prediction Studio loads the two packaged seed-42 classifiers. The Research Report selector includes separate training, validation and test reports. Missing weights disable upload inference visibly; predictions are never filled with random outputs.

## Private dataset setup

Extract the supplied private archive locally:

```text
data/raw/
  sadness/    004_o_m_s_a.jpg ...
  happiness/  004_o_m_h_a.jpg ...
  neutral/    004_o_m_n_a.jpg ...
  fear/       004_o_m_f_a.jpg ...
```

The validator also discovers folders named `Happiness PRML`, etc. A nested root must identify exactly one folder per class. Labels follow the recovered mapping sadness=0, happiness=1, neutral=2, fear=3. Numeric filename prefixes define subjects; expression codes must agree with the class folder. Every image is decoded; corrupt files, ambiguous filenames, duplicate pixels, missing classes and incomplete fold class coverage fail clearly. No images are silently skipped.

```bash
python -m scripts.validate_dataset --data data/raw
python -m scripts.verify_results
```

The audited 48 images all have original dimensions 2835×3543. Exact duplicates: zero. Perceptual matches are review candidates; seven cross-subject pairs were visually reviewed as distinct photographs and their SHA256 pairs are recorded in `results/duplicate_review.json`. Modified input hashes invalidate that review. Real photographs, generator/other checkpoints, generated samples and private example grids are ignored by Git. The only weight exceptions are the two packaged public classifiers. No public source-image distribution permission is assumed.

## Reproduce the corrected experiment

Each fold uses four subjects / 32 images for training, the next sorted subject / 8 images for validation, and one held-out subject / 8 images for testing. This is LOSO testing with one subject-disjoint validation holdout per fold. Preprocessing is fixed EXIF→RGB→64×64 bilinear resize→[−1,1]; augmentation is horizontal flipping on training images only for both methods.

```bash
python -m prml.train --data data/raw --output results/runs/new-run --duplicate-review results/duplicate_review.json
python -m scripts.verify_run --data data/raw --run results/runs/new-run
python -m scripts.export_results --source results/runs/new-run
python -m scripts.build_report
python -m scripts.private_diagnostics --data data/raw --run results/runs/new-run
```

Defaults: 25 CNN epochs, 40 GAN epochs, batch eight, Adam lr 0.0002, seed 42+fold. Seeds reset independently before each model; GAN Adam betas (0.5,0.999), CNN (0.9,0.999). Minimum validation classification CE selects the epoch. Test labels never enter training or selection. A smoke check can use `--smoke --cnn-epochs 1 --gan-epochs 1` with a separate output directory; smoke runs cannot be exported as research results. Existing run directories cannot be overwritten accidentally.

The CNN has three conv/BN/ReLU/max-pool blocks, channels 16/32/64; dense 4096→64→4 with dropout 0.3. AC-GAN uses noise 64+class embedding 16, a convolutional generator, and a four-stage discriminator with source and class heads. The corrected discriminator learns classes on both real and fake images. Generator source loss is explicitly non-saturating. Generator and discriminator are checkpointed at the same selected epoch. Full loss weights and equations are in the report.

`results/corrected/` contains public JSON evidence and histories; `results/figures/` contains statistical plots and architecture diagrams. `results/runs/` contains ignored full local runs/weights. `results/private/` contains ignored generated grids, OOF correct/incorrect examples with scores, and nearest-training-image distances. Distances are low-resolution pixel diagnostics, not proof against memorization. No synthetic face images are published by default.

## Inference and local face display

The default public models are `deployment/cnn.pt` and `deployment/gan.pt`, with hashes in `deployment/manifest.json`. They are selected by seed-42 holdout validation CE, never by test performance. Set `PRML_CHECKPOINT_DIR` to override them locally. A flat directory with `cnn.pt`/`gan.pt` works directly; a directory containing `fold_0/cnn.pt`, etc., exposes the local LOSO fold selector. Do not use a fold ensemble to claim unbiased predictions on training identities: other folds trained on that identity.

```bash
python -m prml.predict --checkpoint results/runs/corrected/fold_0/cnn.pt --image private/example.jpg
```

Checkpoint loading uses CPU mapping, `weights_only=True`, strict state shapes and exact class/size metadata. Uploaded images are handled in memory. The optional gallery requires `PRML_PRIVATE_DATA` and `PRML_ALLOW_FACE_DISPLAY=1`; use only in an authorized private session. The public dashboard contains no source faces or weights.

## Historical recovery and verification

The remote initially contained only a README at commit `e10ee954bc521d764d84f9ca6ffd030a1ccdfe6b`. The supplied local data path revealed earlier source, reports, six folds, histories and 18 checkpoints. `legacy/` preserves the old source/report; `results/historical/` preserves JSON evidence. All 12 recovered classifiers reproduced their 48 OOF labels per method; arithmetic agrees with the stored summary. Original training was not rerun. Original environment metadata is unavailable, and generator weights were not paired with selected discriminator epochs.

To replay recovered classifiers against the separate private old checkpoint directory:

```bash
python -m scripts.verify_results --data data/raw --checkpoints PRIVATE_OLD_RESULTS
```

The corrected run changes the fake-label D loss, selection criterion, seed reset and checkpoint pairing. Historical-versus-corrected performance changes cannot be attributed to a single factor without ablations. Pooled macro F1 and mean fold macro F1 are separate. Exploratory uncertainty resamples subjects, with six-cluster and overlapping-training caveats.

## Streamlit Community Cloud deployment

1. Connect an authorized GitHub account at [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app for `Ahmdullah16/PRML-Assignment`, using the reviewed implementation branch or merged `main`.
3. Set main file path **`app/streamlit_app.py`** and Python **3.12** in advanced settings.
4. Cloud uses `app/requirements.txt` with **CPU-only PyTorch** from the official CPU index, plus display dependencies. No CUDA packages, dataset download or training job is required. Python **3.12** is required for these pins; the cloud default may be newer.
5. Verify all seven pages and both public classifiers. If packaged checkpoints are missing, Prediction Studio must show its unavailable state.

The packaged classifiers ship with the repository; weights are not downloaded from an untrusted upload or arbitrary URL. Inference imports are lazy and loading uses `weights_only=True` with strict class/size/state checks. No generator or source images are fetched. Do not put credentials in code or expose the dataset via app storage. Consult the [official deployment guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). Current verification status is recorded in the delivery log.

## Verification and report

CI installs CPU PyTorch, tests data integrity/leakage, architectures, probabilities, adversarial gradients, loss behavior, deterministic inference, strict loading, one-epoch optimization, metric arithmetic and all seven Streamlit pages. It verifies public historical arithmetic without private images and never runs full training.

Read [the academic report](reports/ASSIGNMENT_REPORT.md), [recovery/audit notes](reports/AUDIT.md) and the delivery log for current PR/check/deployment status. The report cites authentic CNN, GAN and AC-GAN literature and separates measured findings from hypotheses.
