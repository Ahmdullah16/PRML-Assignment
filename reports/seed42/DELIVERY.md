# Seed-42 holdout, notebook and public inference delivery

This follow-up uses the exact ZIP supplied by the user. Its 48 images are retained privately; extraction occurs under ignored `data/seed42_zip/`. The experiment is distinct from the earlier LOSO results.

## Experiment

- Seed: **42**, reset per model. NumPy shuffles subject membership and partition row order; a seeded DataLoader shuffles training batches each epoch.
- Requested split: 50:30:20. Whole-subject allocation: **24 training / 16 validation / 8 test images**, or **50% / 33.33% / 16.67%**.
- Training subjects: **116, 079, 168**. Validation subjects: **140, 066**. Test subject: **004**.
- Epoch budgets fixed before test evaluation: CNN 25; AC-GAN 40. Validation classification CE selects epoch **25 CNN** and **8 AC-GAN**.
- CNN train/validation/test accuracy: **100% / 81.25% / 37.5%**. AC-GAN: **25% / 25% / 25%**.
- Held-out test macro F1: **0.2429 CNN**, **0.1000 AC-GAN**. Eight images from one subject do not support strong population claims.
- Both saved classifiers were replayed on all three partitions. Stored probabilities, metrics and label mappings matched.

## Notebook and reports

`notebooks/PRML_Seed42_CNN_ACGAN.ipynb` executed successfully: **13 code cells, zero errors**, ten embedded statistical figures. Matching completed training is reused and replay-verified; a fresh checkout trains using the same fixed settings. The notebook includes the exact private ZIP path as a default and supports a `PRML_DATA_ZIP` override.

Separate reports: `TRAIN_REPORT.md`, `VALIDATION_REPORT.md`, `TEST_REPORT.md`; summary: `EXPERIMENT_REPORT.md`. Each includes split-specific losses, accuracy, balanced accuracy, macro/per-class precision, recall, F1 and confusion matrices. Training metrics describe seen-image fit; validation metrics are selection-biased; the single held-out-subject test is reported separately.

## Deployment payload and verification

Only `deployment/cnn.pt` (1,152,924 bytes) and `deployment/gan.pt` (2,846,775 bytes) are prepared for public inference. These are classifier snapshots selected on validation, not the generator or source photographs. SHA256 hashes are in `deployment/manifest.json`. Publishing these weights makes them downloadable, so publication is gated on the user's explicit checkpoint approval.

The app uses CPU-only PyTorch, caches trusted model loading, handles uploaded images in memory, and never trains on startup. The previous LOSO evidence remains available separately. Local verification: **24 tests passed**, including packaged checkpoint integrity, deterministic inference, enabled upload controls, dataset splitting, ZIP traversal rejection and notebook execution evidence.

Repository branch: `feat/seed42-holdout-public-prediction`. Publication, merge and live-inference status are recorded in the final chat delivery after verification.
