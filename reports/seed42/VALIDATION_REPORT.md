# Validation report — seed 42

Images: **16**; subjects: **140, 066**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These images selected the checkpoint; the scores are selection-biased and are not test performance.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 1.1154 | 1.3540 |
| accuracy | 0.8125 | 0.2500 |
| balanced_accuracy | 0.8125 | 0.2500 |
| macro_precision | 0.8542 | 0.1250 |
| macro_recall | 0.8125 | 0.2500 |
| macro_f1 | 0.8042 | 0.1667 |

## CNN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.6667 | 1.0000 | 0.8000 |
| happiness | 0.7500 | 0.7500 | 0.7500 |
| neutral | 1.0000 | 0.5000 | 0.6667 |
| fear | 1.0000 | 1.0000 | 1.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  4   0   0   0
  1   3   0   0
  1   1   2   0
  0   0   0   4
```

## GAN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.2500 | 0.5000 | 0.3333 |
| happiness | 0.0000 | 0.0000 | 0.0000 |
| neutral | 0.2500 | 0.5000 | 0.3333 |
| fear | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  2   0   2   0
  2   0   2   0
  2   0   2   0
  2   0   2   0
```