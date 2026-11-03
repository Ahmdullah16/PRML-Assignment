# Recovery, scientific audit and changes

The public repository was inspected before implementation: one initial README commit, one unprotected main branch and no issues/PRs. No PRML implementation or archive was found in the Codex workspace. The supplied local ZIP path then revealed an existing source tree and six complete result folders next to the archive. Those artifacts, rather than inaccessible Library links, became the recovery source.

## Reused

- CNN, generator and discriminator architectures, including names and tensor shapes, remain compatible with recovered weights.
- Fixed label order, 64×64 RGB preprocessing, cached tensors, training-only horizontal flips and the rotating subject-disjoint split were retained and tested.
- Six historical metrics files, class predictions, histories and aggregate configuration are preserved. Original source/report are in `legacy/`.
- All 12 old classifiers were safely loaded and their held-out predictions replayed. Every saved prediction and confusion matrix matched. Original training was not repeated.

## Corrected or added

- Added generated-label class CE to discriminator loss; documented non-saturating G source loss.
- Detached fake images for D, froze D weights for G, tested gradients and checked finite losses.
- Selected checkpoints on validation CE only and synchronized G/D snapshots at the selected epoch.
- Reset seeds independently and recorded deterministic settings, source/dataset hashes, folds, environment, timing, parameters and latency.
- Added duplicate/corruption/label checks, reviewed hash-bound perceptual matches, blocked silent overwrite and smoke-result publication.
- Added pooled OOF metrics alongside fold means/std, exploratory subject bootstrap intervals, probabilities, strict loading and replay verification.
- Added seven-page inference-only dashboard, statistical/architecture figures, private synthetic and correct/incorrect grids, report, README and CI.

## Interpretation boundaries

Historical arithmetic and predictions are verified; original environment and generator-selection pairing are not. Corrected results come from newly executed six-fold training with saved provenance. Multiple controls changed together; differences are not an isolated loss ablation. Folds share training data. Six subjects support descriptive results, not strong population claims. Generated samples and weights remain private pending distribution permission. The public app supports results-only operation.
