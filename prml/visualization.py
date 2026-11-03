"""Scientific figures: no private real or generated faces exported by default."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from . import CLASSES

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.dpi": 180})
COLORS = ["#167E87", "#B65B43"]


def finish(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def confusion(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array(cm)
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set(xticks=range(4), yticks=range(4), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted expression", ylabel="True expression", title=title)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "#142B40")
    fig.colorbar(im, ax=ax, label="Images")
    return fig


def architecture(out):
    for kind, labels in {
        "cnn": ["RGB image\n3 × 64 × 64", "3 convolution blocks\n16 → 32 → 64 channels\nBN · ReLU · max-pool",
                "Flatten 4096\nDense 64 · ReLU\nDropout 0.3", "4 class logits\nSoftmax for inference"],
        "acgan": ["Noise z (64) +\nclass embedding (16)", "Generator\nDense → 4 upsamplers\nTanh RGB 64 × 64",
                  "Discriminator\n4 downsamplers\n32 → 64 → 128 → 256", "Source logit: real/fake\n4 class logits: expression"]
    }.items():
        fig, ax = plt.subplots(figsize=(13, 3))
        ax.axis("off")
        for i, label in enumerate(labels):
            x = .12 + i * .25
            ax.text(x, .5, label, ha="center", va="center", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=1", fc="#EAF0F5", ec=COLORS[0]))
            if i < 3:
                ax.annotate("", xy=(x + .16, .5), xytext=(x + .1, .5), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="->", color=COLORS[0]))
        ax.set_title(f"{kind.upper()} architecture · recovered checkpoint-compatible design", pad=18)
        finish(fig, out / f"{kind}_architecture.png")


def render_results(result_file, output):
    data = json.loads(Path(result_file).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    provenance = "Corrected LOSO · seed 42" if "newly" in data.get("provenance", "") else "Historical LOSO · checkpoint replay verified"
    metrics = data["metrics"]
    for name in ("cnn", "gan"):
        finish(confusion(metrics[name]["pooled"]["confusion_matrix"], f"{name.upper()} · pooled 48 images\n{provenance}"), out / f"{name}_confusion.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = np.arange(len(keys))
    for i, name in enumerate(("cnn", "gan")):
        ax.bar(x + (i - .5) * .35, [metrics[name]["pooled"][k] for k in keys], .35, label=name.upper(), color=COLORS[i])
    ax.set(xticks=x, xticklabels=["Accuracy", "Balanced\naccuracy", "Macro\nprecision", "Macro\nrecall", "Pooled\nmacro F1"], ylim=(0, 1), ylabel="Score (0–1)", title=provenance)
    ax.axhline(.25, color="gray", ls=":", label="Uniform-guess accuracy (only)")
    ax.legend(fontsize=8)
    finish(fig, out / "comparison.png")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(4)
    for ax, key in zip(axes, ("precision_by_class", "recall_by_class", "f1_by_class")):
        for i, name in enumerate(("cnn", "gan")):
            ax.bar(x + (i - .5) * .35, [metrics[name]["pooled"][key][c] for c in CLASSES], .35, color=COLORS[i], label=name.upper())
        ax.set(xticks=x, xticklabels=CLASSES, ylim=(0, 1), title=key.replace("_by_class", ""), ylabel="Score (0–1)")
        ax.tick_params(axis="x", rotation=35)
    axes[0].legend()
    fig.suptitle(provenance)
    finish(fig, out / "class_metrics.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(("cnn", "gan")):
        predictions = metrics[name]["predictions"]
        subjects = sorted({r["subject"] for r in predictions})
        accuracy = [np.mean([r["true"] == r["predicted"] for r in predictions if r["subject"] == s]) for s in subjects]
        ax.plot(subjects, accuracy, "o-", label=name.upper(), color=COLORS[i])
    ax.axhline(.25, color="gray", ls=":")
    ax.set(xlabel="Held-out subject ID", ylabel="Accuracy (0–1)", ylim=(0, 1), title=provenance)
    ax.legend()
    finish(fig, out / "fold_accuracy.png")
    architecture(out)
    history_root = Path(result_file).parent if "newly" in data.get("provenance", "") else Path(result_file).parent / "historical"
    for fold in history_root.glob("fold_*"):
        for kind in ("cnn", "gan"):
            history_file = fold / f"{kind}_history.json"
            if not history_file.exists():
                continue
            hist = json.loads(history_file.read_text())
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            for key in ("train_loss", "val_loss", "generator_loss"):
                if key in hist[0]:
                    axes[0].plot([r["epoch"] for r in hist], [r[key] for r in hist], label=key)
            for key in ("train_accuracy", "val_accuracy"):
                if key in hist[0]:
                    axes[1].plot([r["epoch"] for r in hist], [r[key] for r in hist], label=key)
            axes[0].set(xlabel="Epoch", ylabel="Loss (objective units)")
            axes[1].set(xlabel="Epoch", ylabel="Accuracy (0–1)", ylim=(0, 1))
            for ax in axes:
                ax.legend(fontsize=8)
            fig.suptitle(f"{kind.upper()} · {fold.name} · {provenance}")
            finish(fig, out / f"{fold.name}_{kind}_history.png")


def render_dataset(dataset, output):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(CLASSES, [dataset["counts"][c] for c in CLASSES], color=COLORS[0])
    axes[0].set(ylabel="Original images", title="Dataset class balance · audited input")
    axes[1].bar(dataset["dimensions"].keys(), dataset["dimensions"].values(), color=COLORS[1])
    axes[1].set(xlabel="Original width × height (pixels)", ylabel="Images", title="Image dimensions · before 64 × 64 resize")
    finish(fig, out / "dataset.png")
