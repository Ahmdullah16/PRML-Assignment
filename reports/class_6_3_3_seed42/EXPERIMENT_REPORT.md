# Seed-42 per-class 6/3/3 shuffled image experiment

Each of the four classes is shuffled separately, then split into exactly **6 training, 3 validation, 3 test images**. The combined training, validation and test sets are each shuffled again using independent seed-42 streams. Totals: **24/12/12 images (50%/25%/25%)**. This is the user's corrected image-level protocol.

The split uses `numpy.random.SeedSequence(42)` with one spawned stream per class and one per final partition. Images never overlap across partitions. Training minibatches reshuffle each epoch. Validation/test membership stays fixed and receives no random augmentation.

**Subject overlap is allowed by this image-level design.** The result measures held-out images, not recognition of unseen people. The earlier LOSO and subject-holdout evidence remain separate.

| Partition | Subjects represented | Images | Actual share |
|---|---|---:|---:|
| train | 004, 066, 079, 116, 140, 168 | 24 | 50.00% |
| validation | 004, 066, 079, 116, 140, 168 | 12 | 25.00% |
| test | 004, 079, 116, 140, 168 | 12 | 25.00% |

Both models reuse the existing 64×64 architectures and 25 CNN / 40 AC-GAN epoch budgets. Adam lr=0.0002, batch=8, seed=42 reset per model. Minimum validation classification CE selects the checkpoint. Final partition reports evaluate that selected checkpoint in eval mode without augmentation. The test result is inspected after the budget and selection rule are fixed; it never selects the model.

| Model | Train accuracy | Validation accuracy | Test accuracy | Test macro F1 | Selected epoch |
|---|---:|---:|---:|---:|---:|
| CNN | 100.00% | 41.67% | 41.67% | 0.4194 | 25 |
| GAN | 50.00% | 25.00% | 25.00% | 0.1944 | 2 |

Training accuracy describes fit to seen images; validation metrics are used for model selection and are not unbiased final-test estimates. The test contains 12 images, three per class; one prediction changes accuracy by 8.33 percentage points. Subjects overlap across splits, so image-level scores can be optimistic for unseen-person generalization. These images have appeared in the earlier study, so this is a new split experiment, not a new external test dataset.

The CNN predicts appearance labels directly. The AC-GAN discriminator's auxiliary expression head predicts classes; the generator is not used for public recognition. Results do not measure actual emotions or mental state.

The source ZIP SHA256, file hashes, shuffled order, split records, software environment, seed, model histories, checkpoint epoch and all probabilities are saved in the result JSON. Both checkpoints were replayed across all three partitions. The original LOSO evidence remains separate and unchanged.

Separate reports: [training](TRAIN_REPORT.md), [validation](VALIDATION_REPORT.md), [test](TEST_REPORT.md).

## Exact class allocation

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| sadness | 6 | 3 | 3 |
| happiness | 6 | 3 | 3 |
| neutral | 6 | 3 | 3 |
| fear | 6 | 3 | 3 |

Subject overlap audit:

```json
{
  "train_vs_validation": [
    "004",
    "066",
    "079",
    "116",
    "140",
    "168"
  ],
  "train_vs_test": [
    "004",
    "079",
    "116",
    "140",
    "168"
  ],
  "validation_vs_test": [
    "004",
    "079",
    "116",
    "140",
    "168"
  ]
}
```
