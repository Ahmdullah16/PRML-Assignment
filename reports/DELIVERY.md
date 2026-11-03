# Implementation and acceptance record

Evidence date: 3 November 2026. Repository: https://github.com/Ahmdullah16/PRML-Assignment. Implementation branch: `feat/reproducible-cnn-acgan`.

## Scientific and engineering verification

- Audited 48 images, four balanced classes, six filename-based subjects; every image decodes; no exact duplicates. Seven cross-subject perceptual matches reviewed and hash-bound as distinct photographs.
- Replayed 12 recovered classifier checkpoints: every saved OOF label and metric matched. Recovered training was not repeated; original execution environment remains unavailable.
- Executed all six corrected folds: 25 CNN epochs and 40 GAN epochs per fold, batch eight, seed 42+fold, CPU, four threads; independent train/validation/test subjects.
- Replayed all 12 corrected classifiers: image hashes, split manifests, every OOF probability, selected epochs and pooled/fold metric aggregates verified.
- Local automated suite: **17 passed**, covering seven app pages, data validation, subject separation, model dimensions, probabilities, adversarial gradients, one-epoch optimization, strict checkpoint loading, deterministic inference and historical arithmetic.
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
| Branch/commit/PR | Branch prepared; final commit and PR recorded in chat/PR |
| Merge | Pending review/check completion; no merge claimed by this file |
| Public deployment | Prepared; verified deployment URL recorded in chat if deployed |

The six-subject experiment is a descriptive course study. It does not establish population superiority, generative realism/diversity, privacy of learned weights or actual emotional-state recognition. Public inference needs separately authorized private checkpoint provisioning. No publication of faces or weights is included in this delivery.
