# Validation report — seed 42

Images: **12**; subjects: **004, 066, 079, 116, 140, 168**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These images selected the checkpoint; the scores are selection-biased and are not test performance.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 1.2884 | 1.3864 |
| accuracy | 0.4167 | 0.2500 |
| balanced_accuracy | 0.4167 | 0.2500 |
| macro_precision | 0.3750 | 0.1214 |
| macro_recall | 0.4167 | 0.2500 |
| macro_f1 | 0.3778 | 0.1625 |

## CNN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.0000 | 0.0000 | 0.0000 |
| happiness | 0.5000 | 0.3333 | 0.4000 |
| neutral | 0.3333 | 0.6667 | 0.4444 |
| fear | 0.6667 | 0.6667 | 0.6667 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  0   1   2   0
  0   1   2   0
  0   0   2   1
  1   0   0   2
```

## GAN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.2000 | 0.3333 | 0.2500 |
| happiness | 0.0000 | 0.0000 | 0.0000 |
| neutral | 0.2857 | 0.6667 | 0.4000 |
| fear | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  1   0   2   0
  0   0   3   0
  1   0   2   0
  3   0   0   0
```