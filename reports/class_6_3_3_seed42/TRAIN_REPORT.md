# Train report — seed 42

Images: **24**; subjects: **004, 066, 079, 116, 140, 168**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These are in-sample fit metrics, not generalization estimates.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 0.4060 | 1.3784 |
| accuracy | 1.0000 | 0.5000 |
| balanced_accuracy | 1.0000 | 0.5000 |
| macro_precision | 1.0000 | 0.4265 |
| macro_recall | 1.0000 | 0.5000 |
| macro_f1 | 1.0000 | 0.4354 |

## CNN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 1.0000 | 1.0000 | 1.0000 |
| happiness | 1.0000 | 1.0000 | 1.0000 |
| neutral | 1.0000 | 1.0000 | 1.0000 |
| fear | 1.0000 | 1.0000 | 1.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  6   0   0   0
  0   6   0   0
  0   0   6   0
  0   0   0   6
```

## GAN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.3846 | 0.8333 | 0.5263 |
| happiness | 0.7500 | 0.5000 | 0.6000 |
| neutral | 0.5714 | 0.6667 | 0.6154 |
| fear | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  5   1   0   0
  2   3   1   0
  2   0   4   0
  4   0   2   0
```