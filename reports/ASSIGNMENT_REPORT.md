# Comparative Analysis of Convolutional Neural Networks and Generative Adversarial Networks for Four-Class Facial Expression Recognition

**Ahmdullah · Indian Institute of Technology Guwahati**
M.Tech, Signal Processing and Machine Learning · Electronics and Electrical Engineering
PRML assignment · evidence assembled 3 November 2026

## 1. Abstract

This study compares a supervised CNN with the expression head of an auxiliary-classifier GAN (AC-GAN) on 48 labeled photographs from six filename-based subjects. Each subject contributes two photographs in each of four categories: happiness, sadness, fear and neutral. Leave-one-subject-out evaluation uses four training subjects, one validation subject and one test subject per fold. The corrected experiment achieves 39.58% CNN accuracy (19/48) and 33.33% AC-GAN classifier accuracy (16/48). Pooled macro F1 is 0.3927 and 0.3288 respectively. A recovered historical run was independently checked by replaying all 12 classifier checkpoints, without repeating its original training. Results are descriptive classroom findings with substantial uncertainty, rather than evidence of deployment readiness or a universal advantage for CNNs.

## 2. Introduction

Facial-expression recognition assigns an appearance label to an image. This task does not establish a person's actual emotional or mental state. CNNs optimize classification directly, whereas an AC-GAN jointly learns conditional synthesis and source discrimination with an auxiliary classifier. A small controlled comparison illustrates the distinction between these objectives. Repeated images of the same person require identity-aware evaluation: random image splits can reward recognition of person-specific features.

## 3. Problem statement

Given an RGB image x, estimate one of exactly four class labels. The fixed implementation order is sadness=0, happiness=1, neutral=2, fear=3. Training and checkpoint loading enforce this mapping. The AC-GAN generator is not a classifier: expression predictions come from the discriminator's auxiliary head.

## 4. Objectives

Audit and preserve existing work; validate the supplied images; prevent subject leakage; implement and test correct adversarial and class losses; compare both models on identical held-out photographs; distinguish historical evidence from new experiments; expose results through an inference-only dashboard; and preserve the privacy of source photographs and potentially memorizing weights.

## 5. Related work

Convolutional networks learn shared local filters through gradient-based optimization; LeCun et al. established this approach for document recognition [1]. Goodfellow et al. introduced adversarial optimization between a generator and discriminator [2]. Odena, Olah and Shlens introduced the AC-GAN formulation with a class-conditioned generator and auxiliary discriminator classifier [3]. These sources motivate the architectures and objectives; they do not establish that this small face dataset will benefit from adversarial learning. No unverified benchmark result is borrowed from these papers.

## 6. Dataset description and audit

The accessible local archive and extracted folders contain 48 valid JPEG files: 12 per class, all 2835×3543 pixels. The six filename prefixes are 004, 066, 079, 116, 140 and 168. Every subject has eight images. Subject membership follows the numeric prefix; expression codes s/h/n/f must match folder labels. This is filename-based identity metadata, not independent identity verification or a demographic audit. Dataset licensing, collection protocol and consent documentation were not supplied.

The validator decodes all images, checks dimensions, records SHA256 file hashes, rejects identical decoded pixels, and raises on corrupt files, missing/empty classes or ambiguous names. It found no exact duplicates. Sixty-nine 64-bit dHash pairs have Hamming distance at most five, including seven cross-subject pairs. All ten images implicated in cross-subject matches were visually reviewed as distinct photographs; approved pairs are bound to file hashes in results/duplicate_review.json. Similar lighting and framing can produce perceptual-hash collisions. Hash checks cannot guarantee that all possible near duplicates are detected.

Original photographs are neither copied into public Git nor embedded in public figures. Authorized private example grids, prediction examples and generated samples are produced locally. The dashboard's dataset gallery is disabled unless an operator explicitly configures private local display.

## 7. Data preprocessing

The reused pipeline applies EXIF orientation, RGB conversion, bilinear resizing to 64×64, then x/127.5−1 normalization to [−1,1]. This fixed transform estimates no dataset-wide statistics and uses no test fitting. Faces are already framed portraits; no face detector or learned alignment is used. Square resizing distorts the original aspect ratio and may remove subtle expression information, a limitation preserved for compatibility with recovered checkpoints. Only training samples receive random horizontal flips with probability 0.5. Both corrected methods use the same transform and augmentation rule; adversarial sampling consumes additional RNG draws, so exact augmented minibatches need not match.

## 8. CNN formulation and architecture

For a convolutional filter W, bias b and input X:

\[
Y_{i,j,k}=\sum_{m,n,c}W_{m,n,c,k}X_{i+m,j+n,c}+b_k.
\]

Three blocks use 3×3 convolution, batch normalization, ReLU and 2×2 max pooling, with channels 16, 32 and 64. Local convolution shares parameters; ReLU provides nonlinearity; pooling reduces spatial dimensions; normalization conditions activations. The 64×8×8 feature map is flattened to 4096, projected to 64 units, followed by ReLU, dropout 0.3 and four logits. The original dense head is retained for checkpoint compatibility; replacing it with global average pooling would define a new architecture and invalidate direct checkpoint reuse. This network remains susceptible to memorization.

\[
p(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{4}e^{z_j}},\qquad
\mathcal L_{CNN}=-\sum_{k=1}^{4} y_k\log p(y=k\mid x).
\]

PyTorch cross-entropy consumes logits, implementing stable log-softmax internally. Softmax is applied for inference, not before the training loss. Scores are uncalibrated.

## 9. GAN and AC-GAN formulation

The generator concatenates 64 Gaussian latent values with a 16-dimensional learned class embedding, projects to 256×4×4, then uses four transposed convolutions with channels 128,64,32,3 and output tanh. Intermediate blocks use batch normalization and ReLU. The discriminator downsamples through four stride-2 convolutions (32,64,128,256 channels) with LeakyReLU 0.2, then branches into one source logit and four expression logits.

Let S denote real/fake source and C the class. Original AC-GAN maximizes:

\[
L_S=\mathbb E_x\log P(S=real\mid x)+\mathbb E_{z,c}\log P(S=fake\mid G(z,c)),
\]
\[
L_C=\mathbb E_{x,c}\log P(C=c\mid x)+\mathbb E_{z,c}\log P(C=c\mid G(z,c)).
\]

The original discriminator maximizes L_S+L_C and generator maximizes L_C−L_S. The corrected implementation minimizes:

\[
L_D=BCE(D_s(x),1)+BCE(D_s(G(z,c)),0)+CE(D_c(x),c)+CE(D_c(G(z,c)),c),
\]
\[
L_G=BCE(D_s(G(z,c)),1)+CE(D_c(G(z,c)),c).
\]

All four discriminator terms have unit weights, each averaged over its minibatch before summation. The generator uses a non-saturating source objective, an explicit stabilization variant rather than literal minimax source loss. BCE-with-logits prevents unstable manual logarithms. Fake images are detached for discriminator updates; discriminator weights are frozen while retaining input gradients for generator updates. Tests verify that fake-class gradients reach the discriminator, generator conditioning receives gradients, and generator updates do not accumulate discriminator gradients. There is one D and one G update per batch. Nonfinite losses terminate the run.

## 10. Experimental methodology

The recovered architectures and original preprocessing are reused. Before new code, the public repository was inspected: one initial README commit, no alternate branches, issues or PRs. The subsequently supplied local path revealed the completed source, report, six folds and 18 checkpoints. Original code/report and JSON histories are preserved separately. All recovered class predictions and metric arithmetic were verified; the original training process was not rerun.

The corrected experiment fixes the missing generated-label discriminator loss, validation-only selection, paired generator snapshots, independent per-model seed initialization, strict metadata loading, deterministic settings, and environment/provenance recording. This is a new experiment, not an ablation isolating one correction. No hyperparameter search or test-label-driven adjustment was performed. Its budgets were inherited before viewing corrected results: 25 CNN epochs and 40 GAN epochs, batch size eight, Adam learning rate 0.0002. CNN Adam betas are (0.9,0.999); GAN betas are (0.5,0.999). Different model sizes, objectives and update counts mean equal epochs would not imply equal computation.

## 11. Training and validation strategy

Each of six folds holds out one subject for testing and the next subject in sorted order for validation; the remaining four subjects train both methods. This is LOSO testing with a rotating single-subject validation holdout, not exhaustive nested cross-validation. Validation uses unaugmented real images. The checkpoint with minimum validation classification cross-entropy is selected over the fixed budget; ties retain the first epoch. The selected generator is saved at the same epoch as its discriminator. Frozen models are evaluated on the fold's test subject only after selection.

Seeds are 42+fold and reset separately for each method. Python, NumPy and PyTorch seeds are set; deterministic algorithms are requested and cuDNN benchmarking disabled. Reproducibility remains conditional on the software/hardware environment [4]. Train/validation sets overlap across outer folds, which constrains statistical inference. Saved split manifests, source hashes and image hashes allow auditing of each run.

## 12. Evaluation metrics and uncertainty

Accuracy is correct predictions divided by test images. Per-class precision is TP/(TP+FP), recall TP/(TP+FN), and F1 the harmonic mean of precision and recall. Zero predicted support gives zero precision/F1. Balanced accuracy is the average class recall. Macro precision, recall and F1 average the four class-specific scores. Confusion matrices use true classes as rows and predictions as columns. Each original image appears once in pooled out-of-fold predictions for each method.

Macro F1 is nonlinear: pooled macro F1 is computed from all OOF counts and differs from the mean of six fold macro F1 values. Both are reported. Each fold has eight test images, so one changed prediction moves its accuracy by 12.5 percentage points. The independent sampling units are at most six people, not 48 independent people. Exploratory percentile intervals resample held-out subject accuracies 10,000 times with seed 2026; they condition on these fitted OOF models and omit retraining uncertainty. With six subjects and overlapping training sets, these intervals are descriptive, not reliable population coverage claims or significance tests.

## 13. Results

| Metric (pooled OOF) | CNN | AC-GAN classifier |
|---|---:|---:|
| accuracy | 0.3958 | 0.3333 |
| balanced_accuracy | 0.3958 | 0.3333 |
| macro_precision | 0.3957 | 0.3304 |
| macro_recall | 0.3958 | 0.3333 |
| macro_f1 | 0.3927 | 0.3288 |


| Test subject | Validation subject | CNN accuracy | AC-GAN accuracy | CNN selected epoch | AC-GAN selected epoch |
|---|---|---:|---:|---:|---:|
| 004 | 066 | 0.375 | 0.250 | 25 | 22 |
| 066 | 079 | 0.500 | 0.250 | 8 | 7 |
| 079 | 116 | 0.250 | 0.250 | 25 | 24 |
| 116 | 140 | 0.500 | 0.500 | 25 | 5 |
| 140 | 168 | 0.250 | 0.250 | 23 | 14 |
| 168 | 004 | 0.500 | 0.500 | 25 | 15 |


| Expression | CNN precision | CNN recall | CNN F1 | AC-GAN precision | AC-GAN recall | AC-GAN F1 |
|---|---:|---:|---:|---:|---:|---:|
| sadness | 0.3333 | 0.2500 | 0.2857 | 0.2500 | 0.1667 | 0.2000 |
| happiness | 0.4615 | 0.5000 | 0.4800 | 0.2857 | 0.3333 | 0.3077 |
| neutral | 0.4545 | 0.4167 | 0.4348 | 0.2857 | 0.3333 | 0.3077 |
| fear | 0.3333 | 0.4167 | 0.3704 | 0.5000 | 0.5000 | 0.5000 |



CNN pooled confusion matrix (rows true, columns predicted; sadness, happiness, neutral, fear):

```text
  3   4   2   3
  0   6   3   3
  2   1   5   4
  4   2   1   5
```

GAN pooled confusion matrix (rows true, columns predicted; sadness, happiness, neutral, fear):

```text
  2   4   4   2
  2   4   4   2
  2   4   4   2
  2   2   2   6
```


- CNN: exploratory subject-accuracy bootstrap interval [0.3125, 0.4792]; mean fold macro F1 0.2784.
- GAN: exploratory subject-accuracy bootstrap interval [0.2500, 0.4167]; mean fold macro F1 0.1917.

The corrected CNN predicts 19/48 correctly and AC-GAN 16/48. AC-GAN yields 25% accuracy in four folds and 50% in two. Inspect per-fold confusion matrices and probabilities before treating pooled nonconstant predictions as useful discrimination: a model can produce different constant classes in different folds.

Historical replay-verified accuracy is 33.33% for CNN and 25.00% for GAN; historical mean fold macro F1 is 0.2011 and 0.1000. These retain the original real-only discriminator class loss and selection rule and must not be relabeled as the corrected AC-GAN results. Original environment metadata and generator-selection pairing are missing.

## 14. CNN versus GAN comparison and computational cost

| Quantity | CNN | AC-GAN |
|---|---:|---:|
| Classifier parameters | 286276.00 | 710629.00 |
| Generator parameters | 0.00 | 1022179.00 |
| Total six-fold training/validation time (s) | 20.34 | 136.20 |
| Mean batch-1 forward latency (ms) | 3.09 | 4.25 |


Training time includes validation within the training function, excludes decoding cached inputs, final test evaluation and plotting. Forward latency averages 30 warmed, synchronized batch-1 calls on already preprocessed inputs; image I/O and preprocessing are excluded. Values depend on system load and are not service-level guarantees. AC-GAN total training parameters include both networks, but inference uses only its discriminator. The two methods are not matched for parameter count or compute.

## 15. Discussion

The corrected CNN has a numerical accuracy advantage of three images (6.25 percentage points) over AC-GAN. This does not prove statistical superiority or causation. A direct supervised objective may be easier to optimize with four training subjects; adversarial competition may introduce instability. These are plausible explanations, not isolated experimental findings. CNN and AC-GAN have different feature extractors, regularization and optimizer momentum. An architecture-matched discriminator-only control and repeated seeds would help separate architecture from adversarial effects.

Uniform independent guessing has expected accuracy 25% for four balanced categories. A constant predictor also achieves 25%; near-chance accuracy therefore cannot establish meaningful recognition. The historical GAN's low fold macro F1 is consistent with constant-class behavior visible in saved predictions. No realism, diversity or generative-quality claim follows from discriminator classification scores. Private synthetic grids and nearest-training-image pixel distances are diagnostic only. Pixel distance after downsampling cannot certify absence of memorization.

## 16. Limitations

Six identities and two repetitions per class are insufficient for broad generalization. Single-seed folds, fixed validation identity, architecture/compute mismatch, square resizing and subjective expression labels limit interpretation. No external validation, calibration, demographic fairness assessment, face-localization study, human generated-image rating, FID/IS claim or clinical inference is provided. Original training environment is unavailable. Licensing and permission for public distribution of face images or learned checkpoints are not established. Public inference is disabled unless approved weights are provisioned privately.

## 17. Conclusion

A working, tested CNN/AC-GAN pipeline supports a reproducible comparison on the supplied photographs with explicit subject separation. Recovered results are verified by checkpoint replay; corrected training is separately recorded and evaluated. The CNN's numerical advantage is dataset- and protocol-specific. Neither model is sufficiently validated for consequential expression assessment or for inference about actual emotions.

## 18. Future improvements

Obtain a larger appropriately licensed and consented dataset; use more independent people and multiple seeds; predeclare tuning budgets; use inner group validation for hyperparameters; compare architecture-matched supervised and adversarial heads; assess calibration; and use controlled face alignment and augmentation studies. Evaluate generative quality and memorization independently before distributing samples or weights. Avoid selecting improvements on these same test subjects without a new external test set.

## 19. References

1. LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. (1998). Gradient-Based Learning Applied to Document Recognition. Proceedings of the IEEE 86(11), 2278–2324. [Author-hosted record](https://bottou.org/papers/lecun-98h).
2. Goodfellow, I. et al. (2014). Generative Adversarial Networks. [Original paper](https://arxiv.org/abs/1406.2661).
3. Odena, A., Olah, C., and Shlens, J. (2017). Conditional Image Synthesis With Auxiliary Classifier GANs. [Original paper](https://arxiv.org/abs/1610.09585).
4. PyTorch documentation. [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html). Consulted 3 November 2026.
5. Streamlit documentation. [Prepare and deploy an app](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). Consulted 3 November 2026.

## 20. Reproducibility instructions

Use Python 3.12 and the explicit direct-dependency pins in requirements.txt. Tests use requirements-dev.txt. See results/corrected/run.json for the actual execution environment, source SHA256 values, device, threads and configuration. The run began from the initial repository commit with local implementation changes; source hashes, rather than that initial commit alone, identify executed code. A later Git commit packages the implementation. Private images and weights are deliberately absent from Git.

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pytest -q
python -m scripts.validate_dataset --data data/raw
python -m scripts.verify_results
python -m prml.train --data data/raw --output results/runs/new-run --duplicate-review results/duplicate_review.json
python -m scripts.verify_run --data data/raw --run results/runs/new-run
python -m scripts.export_results --source results/runs/new-run
python -m scripts.private_diagnostics --data data/raw --run results/runs/new-run
python -m streamlit run app/streamlit_app.py
```

For a different dataset, review perceptual-match candidates and regenerate the hash-bound review record; do not reuse unrelated approvals. Existing output runs cannot be overwritten accidentally. Smoke runs are explicitly marked and refused by the public exporter. Streamlit performs no training or private dataset download during startup. The seven pages present results and methods; upload inference requires a trusted locally configured checkpoint directory. No live public deployment is claimed by this report; deployment status is recorded separately in the delivery log.
