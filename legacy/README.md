# Recovered implementation

These source files and the report are preserved as historical evidence. The current entry point is `python -m prml.train`; do not use this directory as the corrected implementation. Model architectures, class mapping, RGB preprocessing, cached datasets, horizontal-flip augmentation, and LOSO partition logic were reused from `original_train.py`.

The original discriminator's class loss used only real images, while the generator used generated labels. Its selected discriminator snapshot and final generator snapshot came from potentially different epochs. Checkpoint selection used validation accuracy with training-loss tie breaking. Original random state was initialized once per fold and consumed by CNN training before GAN training. New experiments correct these issues and are reported separately. Replay verifies original predictions, not original training reproducibility. See `results/historical_verified.json` and the current report.
