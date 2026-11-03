# Implementation and acceptance record

Evidence date: 3 November 2026. Repository: https://github.com/Ahmdullah16/PRML-Assignment. Implementation branch: `feat/reproducible-cnn-acgan`; follow-up: `fix/lightweight-cloud-dashboard`.

Implementation commit `e06e5aca4f5578a07a773c2a969f41eadefa054f` was merged through [PR #1](https://github.com/Ahmdullah16/PRML-Assignment/pull/1) by Ahmdullah16. Merge commit: `be1f6db5d894fb3d7ed1467da08a5ab0b72c9463`. [Hosted CI passed](https://github.com/Ahmdullah16/PRML-Assignment/actions/runs/41634099413).

The public dashboard was rendered and verified at **https://facial-expression.streamlit.app/**. Its cloud runtime was corrected from Python 3.14 to the tested Python 3.12 after a dependency build failure. The follow-up uses `app/requirements.txt` to keep cloud display independent of PyTorch/CUDA, with lazy model imports for authorized local inference.

## Scientific and engineering verification

- Audited 48 images, four balanced classes, six filename-based subjects; every image decodes; no exact duplicates. Seven cross-subject perceptual matches reviewed and hash-bound as distinct photographs.
- Replayed 12 recovered classifier checkpoints: every saved OOF label and metric matched. Recovered training was not repeated; original execution environment remains unavailable.
- Executed all six corrected folds: 25 CNN epochs and 40 GAN epochs per fold, batch eight, seed 42+fold, CPU, four threads; independent train/validation/test subjects.
- Replayed all 12 corrected classifiers: image hashes, split manifests, every OOF probability, selected epochs and pooled/fold metric aggregates verified.
- Local automated suite: **18 passed**, covering seven app pages, operation without PyTorch, data validation, subject separation, model dimensions, probabilities, adversarial gradients, one-epoch optimization, strict checkpoint loading, deterministic inference and historical arithmetic.
- Local Streamlit server verified at port 8517; Prediction Studio rendered both classifiers with probability charts. Source photographs and all weights remain excluded from Git.
- Statistical figures and all fold histories generated for historical and corrected experiments. Private generated grids, per-image classification examples and nearest-training-image diagnostics generated under ignored `results/private/`.
- Detailed report and README completed. CI runs CPU checks without private images or full training.

## Acceptance status

| Requirement | Status |
|---|---|
| Four classes, dataset inspection and subject-disjoint splitting | Passed |
| CNN and genuine AC-GAN implementation, loss/gradient tests | Passed |
| Controlled comparison, pooled/fold metrics, probabilities, confusion matrices | Passed |
| Existing results verified or qualified | Passed: classifier replay verified; original training provenance qualified |
| Training and comparison graphs, architecture diagrams | Passed |
| Dataset and generated examples | Generated privately; public face display requires distribution permission |
| Streamlit pages and local checkpoint inference | Passed |
| Private images and weights excluded from Git | Passed |
| Academic report, README, tests and CI definition | Passed locally; hosted CI status recorded in PR |
| Branch/commit/PR | Main implementation integrated through PR #1; follow-up PR recorded in chat |
| Merge | PR #1 merged by Ahmdullah16; follow-up status recorded in chat |
| Public deployment | Verified: https://facial-expression.streamlit.app/; public prediction disabled without private weights |

The six-subject experiment is a descriptive course study. It does not establish population superiority, generative realism/diversity, privacy of learned weights or actual emotional-state recognition. Public inference needs separately authorized private checkpoint provisioning. No publication of faces or weights is included in this delivery.
