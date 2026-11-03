"""Private generated samples, nearest training image distances and OOF examples.

Never publish outputs automatically: GANs can memorize identifiable photographs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prml import CLASSES, LATENT
from prml.models import Generator
from prml.data import discover, split_subjects, Faces
from prml.train import write_json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(data, run, out):
    rows = discover(data)
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for fold_path in sorted(run.glob("fold_*")):
        fold = int(fold_path.name.split("_")[1])
        train, _, test = split_subjects(rows, fold)
        saved = torch.load(fold_path / "generator.pt", map_location="cpu", weights_only=True)
        generator = Generator()
        generator.load_state_dict(saved["model"])
        generator.eval()
        rng = torch.Generator().manual_seed(2026 + fold)
        y = torch.arange(4).repeat_interleave(4)
        with torch.inference_mode():
            fake = generator(torch.randn(16, LATENT, generator=rng), y)
        originals = torch.stack(Faces(train).images)
        distances = ((fake[:, None] - originals[None]) ** 2).mean((2, 3, 4)).sqrt()
        nearest = distances.min(1)
        # Pixel distance is a coarse flag; no threshold proves absence of memorization.
        diagnostics.append(dict(fold=fold, selected_epoch=saved["best_epoch"],
             nearest_train_pixel_rmse=nearest.values.tolist(),
             nearest_train_filenames=[train[i]["filename"] for i in nearest.indices.tolist()],
             warning="Low-resolution pixel distance only; not a privacy guarantee or diversity score"))
        fig, axes = plt.subplots(4, 4, figsize=(6, 7))
        for i, ax in enumerate(axes.flat):
            ax.imshow(((fake[i].permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy())
            ax.set_title(CLASSES[y[i]], fontsize=8)
            ax.axis("off")
        fig.suptitle(f"SYNTHETIC · {fold_path.name} · selected epoch {saved['best_epoch']}\nCorrected AC-GAN; no realism claim")
        fig.tight_layout()
        fig.savefig(out / f"{fold_path.name}_synthetic.png", dpi=140)
        plt.close(fig)
        metrics = json.loads((fold_path / "metrics.json").read_text())
        for kind in ("cnn", "gan"):
            preds = {r["filename"]: r for r in metrics["models"][kind]["test_predictions"]}
            fig, axes = plt.subplots(2, 4, figsize=(11, 7))
            for ax, row in zip(axes.flat, test):
                with Image.open(row["path"]) as im:
                    im.thumbnail((200, 250))
                    ax.imshow(im)
                p = preds[row["filename"]]
                ax.set_title(f'True: {p["true"]}\nPred: {p["predicted"]} ({max(p["probabilities"]):.2f})', fontsize=9)
                ax.axis("off")
            fig.suptitle(f"PRIVATE · {kind.upper()} · held-out subject {test[0]['subject']} · corrected experiment")
            fig.tight_layout()
            fig.savefig(out / f"{fold_path.name}_{kind}_examples.png", dpi=120)
            plt.close(fig)
    write_json(out / "memorization_diagnostics.json", diagnostics)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, default=Path("results/runs/corrected"))
    p.add_argument("--output", type=Path, default=Path("results/private"))
    a = p.parse_args()
    main(a.data, a.run, a.output)
