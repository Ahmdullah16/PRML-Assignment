# PRML Assignment Report — Four-Class Facial Expression Classification using CNN and GAN

## Abstract
This project classifies labeled facial photographs into sadness, happiness, neutral, and fear using two approaches: (1) a supervised convolutional neural network and (2) an auxiliary-classifier generative adversarial network, where the discriminator's class head predicts expressions. To address repeated photographs of the same individuals, a leave-one-subject-out test protocol with a separate validation subject is used. The dataset comprises 48 JPEG images (12 per category, six subjects), with just eight test samples per fold. Accuracy, macro-F1, balanced accuracy, and confusion matrices are computed for each model.

## Dataset and pre-processing
Source folders: Happiness PRML, Sadness PRML, Fear PRML, Neutral PRML. Class counts are balanced at 12 images each. Image filenames share person IDs. For every fold: 32 training images from four people, eight validation images from one person, eight test images from another person. Every photograph is EXIF-normalized, resized to 64×64 RGB, and mapped into [-1,1]. Random horizontal flip is used during CNN training only.

## Architectures
**CNN:** feature extractor with three convolutional blocks at 16, 32, 64 channels; max-pool after each block; 64-dimensional fully connected layer, dropout, and four output logits. Supervised cross entropy.

**AC-GAN:** generator concatenates 64-dimensional noise with learned 16-dimensional class embedding and upscales through transposed convolutions into 64×64 RGB. Discriminator uses four downsampling convolutions (32, 64, 128, 256 channels), a real/fake logistic head, and a four-class head. Train both adversarial and auxiliary classification objectives. At test time only the discriminator's four-class head is used.

## Protocol and metrics
A seed of 42 (offset by fold) initializes each run. The main measured experiment uses 25 CNN epochs and 40 GAN epochs with batch size 8 and learning rate 0.0002. The best epoch is selected only using validation accuracy, breaking ties on training loss. Test subjects remain unseen during both model fitting and selection. All six people take turns as test subject. Reported scores reflect a single training seed per fold.

## Observations
Actual numerical outcomes are saved in `results/summary.csv`, `results/aggregate.json`, and individual fold JSON/PNG outputs. Numbers should be copied from those files, not invented or inferred. With six test folds of eight images each, the sample size is too small for broad generalization claims. The GAN training objective is more complex than CNN cross-entropy; training losses cannot be directly compared.

## Conclusion
This is a valid classroom comparison of a discriminative CNN versus a generator-plus-auxiliary-classifier GAN, with explicit subject-leakage safeguards. Results establish behavior on the supplied limited photographs only. This is classification of labeled facial expressions, not a validated measurement of true emotions.

## Actual six-fold experimental results (completed run)

The program ran all six folds successfully with CPU/PyTorch and returned these measured values. The run was seeded once per fold; these are descriptive results, not confidence intervals.

| Test subject ID | CNN accuracy | GAN discriminator accuracy | CNN macro F1 | GAN macro F1 |
|---|---:|---:|---:|---:|
| 004 | 37.5% | 25.0% | 0.250 | 0.100 |
| 066 | 50.0% | 25.0% | 0.343 | 0.100 |
| 079 | 25.0% | 25.0% | 0.100 | 0.100 |
| 116 | 25.0% | 25.0% | 0.111 | 0.100 |
| 140 | 25.0% | 25.0% | 0.125 | 0.100 |
| 168 | 37.5% | 25.0% | 0.278 | 0.100 |

**Aggregate (mean over six subject folds):** CNN accuracy **33.33%**, macro F1 **0.2011**; GAN discriminator accuracy **25.00%**, macro F1 **0.1000**. Subject-fold accuracy standard deviations: CNN **0.0932** and GAN **0.0000** (population standard deviation). Random baseline for four balanced categories is **25%**. Thus the observed CNN result is numerically better on this dataset, while neither method is robust enough for general-purpose prediction.

The result is a **negative finding for the small-data GAN approach**: despite training a conditional generator and adversarial discriminator, the discriminator classification head did not exceed chance accuracy on unseen subjects. Generator samples were **not independently rated for image quality** and should not be described as realistic or high-fidelity.

## Reproducibility evidence
- `python -m pytest -q`: 2 unit tests passed.
- `python -m prml.train --fold all --cnn-epochs 25 --gan-epochs 40 --batch-size 8 --output results`: exited successfully.
- `results/summary.csv`: full per-subject comparisons; `results/fold_*/`: six test confusion matrices and saved checkpoints.
- `results/aggregate.json`: training configuration and descriptive statistics.
