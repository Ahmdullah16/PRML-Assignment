# Kaggle retraining audit

Base repository: `Ahmdullah16/PRML-Assignment`, main commit `5b16c99430d4250680d4708ff14afd9366dbed0a`.
Implementation branch: `experiment/kaggle-four-class-seed42`.

The existing `prml/models.py` CNN, conditional generator and dual-head discriminator are reused unchanged. Existing adversarial loss helpers, strict CPU inference, preprocessing and metric arithmetic are reused. The historical LOSO, subject holdout and per-class 6/3/3 studies remain in their original directories. New scripts and `results/kaggle_seed42` isolate the new experiment. No historical scores are substituted for retrained results.

The original dataset is accessible at the user-provided Windows path. The documented four folders contain 25,053 train/test files combined. Initial decoding found 393 invalid training JPEGs containing HTTP 429 text. `scripts/repair_kaggle.py` creates a private copy and recovers those files from the dataset README's documented upstream GitHub archive. The original download remains unchanged. `repair_provenance.json` records archive and per-file hashes. No test file needed repair. Kaggle's public dataset API identified version 17 as current (last updated 2026-10-17); this does not prove the local download version. The exact experiment snapshot is identified by SHA256 manifests.

The final repaired snapshot has 25,053 decodable 48x48 JPEGs in the selected four classes. Audit found 604 exact pixel-duplicate groups and 540 conservative near-duplicate pairs (dHash Hamming <=3 plus grayscale 32x32 RMSE <=0.06). Whole groups are used for splitting; groups overlapping the original test partition are excluded from training. Redundant exact training duplicates and conflicting-label training groups are excluded. Exclusions total 868 training files. All 5,009 original selected test images remain test images. Within-test duplicates remain and limit effective sample independence. Subject IDs are unavailable; image-level splitting does not establish subject independence. Heuristic near-duplicate search is conservative and not a guarantee against every similar image.

All 25,053 selected file hashes match the documented upstream archive byte-for-byte. Observed upstream commit: `444c3b5871d5404012abec91466ce1817df66af9`. Archive SHA256: `e7b93c455c73c553545c401b7053508419de5290a4c833bf896d38f5301f6732`. Full comparison is recorded in `results/kaggle_seed42/upstream_comparison.json`.

The 5,009 original selected test records contain 4,971 unique decoded pixel hashes: 38 repeated test records. One exact-pixel test group has conflicting source labels. All original test annotations and membership are retained identically for both models; these facts are disclosed as dependence and annotation-quality limitations rather than silently changing the benchmark. `test_duplicate_summary.json` records the audit.

| Expression | Train | Validation | Test |
|---|---:|---:|---:|
| Happiness | 5307 | 936 | 1599 |
| Sadness | 4029 | 711 | 1240 |
| Fear | 2821 | 498 | 911 |
| Neutral | 4143 | 731 | 1259 |
| Total | 16300 | 2876 | 5009 |

Manifest hash: `812f3a7974cf89273bb454df2a2b4298152e635d958e56cbee5b063a08206d60`.

Roll number 264102307 was verified against the locally supplied instructor assignment PDF, matching AHMDULLAH and the CNN/GAN four-class topic. The PDF itself contains other students' personal information and is not redistributed.

Hardware: Intel Core i5-1135G7, 4 cores/8 logical processors, Intel Iris Xe, no CUDA GPU. Training uses four CPU threads. CNN budget: 20 epochs, AdamW lr .001, ReduceLROnPlateau. GAN budget: 15 epochs, D AdamW with zero decay, G Adam, lr .0002, betas .5/.999. Both use batch 64 and five-epoch early-stopping patience. Validation classification CE selects checkpoints. GAN generator and discriminator are saved together at the selected epoch. Source photographs, resume optimizer states, generator weights and generated faces are not committed to public GitHub. Selected generator weights may be included in the private academic submission for reproduction.

Environment setup initially encountered a blocked pandas 3.1.1 native binary under Windows Application Control. Installing the repository's existing pandas 2.3.3 dependency resolved imports; no security policy was changed. Training uses PyTorch 2.15.0+cpu. Public inference keeps the existing pinned cloud runtime and checkpoint-compatible architectures.
