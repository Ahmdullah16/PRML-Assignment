"""Three separate split reports and statistical figures from the frozen seed-42 run."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from prml import CLASSES
from prml.holdout import SPLIT_NAMES
from prml.visualization import confusion, finish


def build_reports(root=Path("results/holdout_seed42"), report_root=Path("reports/seed42")):
    root, report_root = Path(root), Path(report_root)
    summary = json.loads((root / "summary.json").read_text())
    report_root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    overview = ["# Seed-42 shuffled subject holdout", "",
        "A separate experiment from the six-fold LOSO study. The requested 50:30:20 subject allocation rounds to "
        "three training, two validation and one test subject: **24/16/8 images (50%/33.33%/16.67%)**. "
        "An exact image-level ratio would split identities across partitions and is not used.", "",
        "Subject IDs are shuffled once using `numpy.random.default_rng(42)`. Rows inside each partition are also shuffled. "
        "Training minibatches shuffle each epoch using a seeded PyTorch generator. Validation/test membership stays fixed; "
        "validation and test never receive random augmentation.", "",
        "| Partition | Subjects | Images | Actual share |", "|---|---|---:|---:|"]
    for split in SPLIT_NAMES:
        part = summary["split_manifest"]["partitions"][split]
        overview.append(f'| {split} | {", ".join(part["subjects"])} | {part["images"]} | {part["actual_ratio"]:.2%} |')
    overview += ["", "Both models reuse the existing 64×64 architectures and 25 CNN / 40 AC-GAN epoch budgets. "
        "Adam lr=0.0002, batch=8, seed=42 reset per model. Minimum validation classification CE selects the checkpoint. "
        "Final partition reports evaluate that selected checkpoint in eval mode without augmentation. "
        "The test result is inspected after the budget and selection rule are fixed; it never selects the model.", "",
        "| Model | Train accuracy | Validation accuracy | Test accuracy | Test macro F1 | Selected epoch |",
        "|---|---:|---:|---:|---:|---:|"]
    for kind in ("cnn", "gan"):
        model = summary["models"][kind]
        parts = model["partitions"]
        overview.append(f'| {kind.upper()} | {parts["train"]["accuracy"]:.2%} | {parts["validation"]["accuracy"]:.2%} | '
                        f'{parts["test"]["accuracy"]:.2%} | {parts["test"]["macro_f1"]:.4f} | {model["best_epoch"]} |')
    overview += ["", "Training accuracy describes fit to seen images; validation metrics are used for model selection and "
        "are not unbiased final-test estimates. The test contains one subject and eight images; one prediction changes "
        "test accuracy by 12.5 percentage points. No population confidence interval is justified from one test identity. "
        "These images have appeared in the earlier study, so this is a new split experiment, not a new external test dataset.", "",
        "The CNN predicts appearance labels directly. The AC-GAN discriminator's auxiliary expression head predicts classes; "
        "the generator is not used for public recognition. Results do not measure actual emotions or mental state.", "",
        "The source ZIP SHA256, file hashes, shuffled order, split records, software environment, seed, model histories, "
        "checkpoint epoch and all probabilities are saved in the result JSON. Both checkpoints were replayed across "
        "all three partitions. The original LOSO evidence remains separate and unchanged.", "",
        "Separate reports: [training](TRAIN_REPORT.md), [validation](VALIDATION_REPORT.md), [test](TEST_REPORT.md)."]
    (report_root / "EXPERIMENT_REPORT.md").write_text("\n".join(overview), encoding="utf-8")
    for split in SPLIT_NAMES:
        partition = summary["split_manifest"]["partitions"][split]
        lines = [f"# {split.title()} report — seed 42", "",
            f'Images: **{partition["images"]}**; subjects: **{", ".join(partition["subjects"])}**. '
            "All scores below use the frozen validation-selected checkpoint, eval mode and no augmentation.", "",
            {"train": "These are in-sample fit metrics, not generalization estimates.",
             "validation": "These images selected the checkpoint; the scores are selection-biased and are not test performance.",
             "test": "These are held-out-subject metrics for one identity. The split/budget were fixed before this evaluation."}[split], "",
            "| Metric | CNN | AC-GAN classifier |", "|---|---:|---:|"]
        for metric in ("loss", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"):
            values = [summary["models"][kind]["partitions"][split][metric] for kind in ("cnn", "gan")]
            lines.append(f'| {metric} | {values[0]:.4f} | {values[1]:.4f} |')
        for kind in ("cnn", "gan"):
            values = summary["models"][kind]["partitions"][split]
            lines += ["", f"## {kind.upper()} per-class metrics", "", "| Class | Precision | Recall | F1 |", "|---|---:|---:|---:|"]
            for cls in CLASSES:
                lines.append("| " + cls + " | " + " | ".join(f'{values[key][cls]:.4f}' for key in
                             ("precision_by_class", "recall_by_class", "f1_by_class")) + " |")
            lines += ["", "Confusion matrix: rows true, columns predicted; sadness, happiness, neutral, fear.", "", "```text",
                      *[" ".join(f"{x:3}" for x in row) for row in values["confusion_matrix"]], "```"]
            finish(confusion(values["confusion_matrix"], f"{kind.upper()} · {split} · seed 42\n"
                   f'{partition["images"]} images / {len(partition["subjects"])} subjects'), figures / f"{split}_{kind}_confusion.png")
        (report_root / f"{split.upper()}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(3)
    for i, kind in enumerate(("cnn", "gan")):
        vals = [summary["models"][kind]["partitions"][split]["accuracy"] for split in SPLIT_NAMES]
        ax.bar(x + (i - .5) * .35, vals, .35, label=kind.upper())
    ax.set(xticks=x, xticklabels=["Train (seen)", "Validation (selection)", "Test (held-out)"],
           ylabel="Accuracy (0–1)", ylim=(0, 1), title="Separate partition performance · subject split · seed 42")
    ax.legend()
    finish(fig, figures / "split_accuracy.png")
    for kind in ("cnn", "gan"):
        history = json.loads((root / f"{kind}_history.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for key in ("train_loss", "val_loss", "generator_loss"):
            if key in history[0]:
                axes[0].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        for key in ("train_accuracy", "val_accuracy"):
            axes[1].plot([r["epoch"] for r in history], [r[key] for r in history], label=key)
        axes[0].set(xlabel="Epoch", ylabel="Loss (model-specific units)")
        axes[1].set(xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
        for ax in axes:
            ax.axvline(summary["models"][kind]["best_epoch"], ls=":", color="gray", label="Selected epoch")
            ax.legend(fontsize=8)
        fig.suptitle(f"{kind.upper()} · shuffled subject holdout · seed 42")
        finish(fig, figures / f"{kind}_history.png")
    return summary


if __name__ == "__main__":
    build_reports()
