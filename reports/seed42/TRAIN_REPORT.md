# Train report — seed 42

Images: **24**; subjects: **116, 079, 168**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These are in-sample fit metrics, not generalization estimates.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 0.4216 | 1.3475 |
| accuracy | 1.0000 | 0.2500 |
| balanced_accuracy | 1.0000 | 0.2500 |
| macro_precision | 1.0000 | 0.0625 |
| macro_recall | 1.0000 | 0.2500 |
| macro_f1 | 1.0000 | 0.1000 |

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
| sadness | 0.0000 | 0.0000 | 0.0000 |
| happiness | 0.0000 | 0.0000 | 0.0000 |
| neutral | 0.2500 | 1.0000 | 0.4000 |
| fear | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  0   0   6   0
  0   0   6   0
  0   0   6   0
  0   0   6   0
```