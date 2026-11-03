# Test report — seed 42

Images: **8**; subjects: **004**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These are held-out-subject metrics for one identity. The split/budget were fixed before this evaluation.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 1.1796 | 1.3757 |
| accuracy | 0.3750 | 0.2500 |
| balanced_accuracy | 0.3750 | 0.2500 |
| macro_precision | 0.1833 | 0.0625 |
| macro_recall | 0.3750 | 0.2500 |
| macro_f1 | 0.2429 | 0.1000 |

## CNN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.0000 | 0.0000 | 0.0000 |
| happiness | 0.4000 | 1.0000 | 0.5714 |
| neutral | 0.0000 | 0.0000 | 0.0000 |
| fear | 0.3333 | 0.5000 | 0.4000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  0   2   0   0
  0   2   0   0
  0   0   0   2
  0   1   0   1
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
  0   0   2   0
  0   0   2   0
  0   0   2   0
  0   0   2   0
```