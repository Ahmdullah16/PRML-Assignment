# Test report — seed 42

Images: **12**; subjects: **004, 079, 116, 140, 168**. All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.

These are held-out-image metrics. Subjects can also occur in training/validation; this is not subject-independent testing. The split/budget were fixed before this evaluation.

| Metric | CNN | AC-GAN classifier |
|---|---:|---:|
| loss | 1.2649 | 1.3905 |
| accuracy | 0.4167 | 0.2500 |
| balanced_accuracy | 0.4167 | 0.2500 |
| macro_precision | 0.5417 | 0.1667 |
| macro_recall | 0.4167 | 0.2500 |
| macro_f1 | 0.4194 | 0.1944 |

## CNN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 1.0000 | 0.3333 | 0.5000 |
| happiness | 0.3333 | 0.3333 | 0.3333 |
| neutral | 0.3333 | 0.6667 | 0.4444 |
| fear | 0.5000 | 0.3333 | 0.4000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  1   1   1   0
  0   1   2   0
  0   0   2   1
  0   1   1   1
```

## GAN per-class metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| sadness | 0.0000 | 0.0000 | 0.0000 |
| happiness | 0.3333 | 0.3333 | 0.3333 |
| neutral | 0.3333 | 0.6667 | 0.4444 |
| fear | 0.0000 | 0.0000 | 0.0000 |

Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.

```text
  0   2   1   0
  1   1   1   0
  1   0   2   0
  1   0   2   0
```