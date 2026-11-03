# Seed-42 shuffled subject holdout

A separate experiment from the six-fold LOSO study. The requested 50:30:20 subject allocation rounds to three training, two validation and one test subject: **24/16/8 images (50%/33.33%/16.67%)**. An exact image-level ratio would split identities across partitions and is not used.

Subject IDs are shuffled once using `numpy.random.default_rng(42)`. Rows inside each partition are also shuffled. Training minibatches shuffle each epoch using a seeded PyTorch generator. Validation/test membership stays fixed; validation and test never receive random augmentation.

| Partition | Subjects | Images | Actual share |
|---|---|---:|---:|
| train | 116, 079, 168 | 24 | 50.00% |
| validation | 140, 066 | 16 | 33.33% |
| test | 004 | 8 | 16.67% |

Both models reuse the existing 64×64 architectures and 25 CNN / 40 AC-GAN epoch budgets. Adam lr=0.0002, batch=8, seed=42 reset per model. Minimum validation classification CE selects the checkpoint. Final partition reports evaluate that selected checkpoint in eval mode without augmentation. The test result is inspected after the budget and selection rule are fixed; it never selects the model.

| Model | Train accuracy | Validation accuracy | Test accuracy | Test macro F1 | Selected epoch |
|---|---:|---:|---:|---:|---:|
| CNN | 100.00% | 81.25% | 37.50% | 0.2429 | 25 |
| GAN | 25.00% | 25.00% | 25.00% | 0.1000 | 8 |

Training accuracy describes fit to seen images; validation metrics are used for model selection and are not unbiased final-test estimates. The test contains one subject and eight images; one prediction changes test accuracy by 12.5 percentage points. No population confidence interval is justified from one test identity. These images have appeared in the earlier study, so this is a new split experiment, not a new external test dataset.

The CNN predicts appearance labels directly. The AC-GAN discriminator's auxiliary expression head predicts classes; the generator is not used for public recognition. Results do not measure actual emotions or mental state.

The source ZIP SHA256, file hashes, shuffled order, split records, software environment, seed, model histories, checkpoint epoch and all probabilities are saved in the result JSON. Both checkpoints were replayed across all three partitions. The original LOSO evidence remains separate and unchanged.

Separate reports: [training](TRAIN_REPORT.md), [validation](VALIDATION_REPORT.md), [test](TEST_REPORT.md).