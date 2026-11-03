"""Subject-disjoint 4-class CNN vs auxiliary-classifier GAN (AC-GAN) benchmark."""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

CLASSES = ("sadness", "happiness", "neutral", "fear")
SIZE = 64
LATENT = 64


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover(root: Path) -> list[dict]:
    """Read labels from parent dirs, subject IDs from NNN_ filename prefix."""
    root = Path(root)
    rows = []
    for label, name in enumerate(CLASSES):
        directory = root / name
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing expected class directory: {directory}")
        for path in sorted(directory.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            match = re.match(r"^(\d+)_", path.name)
            if not match:
                raise ValueError(f"Subject ID missing from filename: {path.name}")
            try:
                with Image.open(path) as im:
                    im.verify()
            except Exception as exc:
                raise ValueError(f"Invalid image {path}: {exc}") from exc
            rows.append({"path": str(path), "label": label, "class": name,
                         "subject": match.group(1), "filename": path.name})
    if not rows:
        raise ValueError(f"No images under {root}")
    if len(set(x["subject"] for x in rows)) < 3:
        raise ValueError("At least three different subjects required")
    return rows


def split_subjects(rows: list[dict], fold: int = 0) -> tuple[list[dict], list[dict], list[dict]]:
    subjects = sorted({x["subject"] for x in rows})
    test_id, val_id = subjects[fold % len(subjects)], subjects[(fold + 1) % len(subjects)]
    train = [r for r in rows if r["subject"] not in (test_id, val_id)]
    val = [r for r in rows if r["subject"] == val_id]
    test = [r for r in rows if r["subject"] == test_id]
    assert {r["subject"] for r in train}.isdisjoint({val_id, test_id})
    assert {r["subject"] for r in val}.isdisjoint({test_id})
    if not all({r["label"] for r in part} == set(range(4)) for part in (train, val, test)):
        raise ValueError("Every subject split must contain all four expressions")
    return train, val, test


def image_tensor(path: str) -> torch.Tensor:
    with Image.open(path) as im:
        rgb = ImageOps.exif_transpose(im).convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1)


class Faces(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False):
        self.rows, self.augment = rows, augment
        # Cache to prevent reopening 2-4 MB originals each epoch.
        self.images = [image_tensor(r["path"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        x = self.images[index]
        if self.augment and torch.rand(()) < 0.5:
            x = torch.flip(x, dims=[2])
        return x, self.rows[index]["label"]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(ni, no):
            return [nn.Conv2d(ni, no, 3, padding=1), nn.BatchNorm2d(no), nn.ReLU(), nn.MaxPool2d(2)]
        self.features = nn.Sequential(*block(3, 16), *block(16, 32), *block(32, 64))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 64), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(64, 4))

    def forward(self, x):
        return self.head(self.features(x))


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(4, 16)
        self.net = nn.Sequential(
            nn.Linear(LATENT + 16, 256 * 4 * 4), nn.ReLU(), nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh())

    def forward(self, z, y):
        return self.net(torch.cat((z, self.embed(y)), dim=1))


class Discriminator(nn.Module):
    """GAN discriminator: real/fake head plus four-expression class head."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten())
        self.real_head = nn.Linear(256 * 4 * 4, 1)
        self.class_head = nn.Linear(256 * 4 * 4, 4)

    def forward(self, x):
        h = self.features(x)
        return self.real_head(h).flatten(), self.class_head(h)


def make_loaders(train, val, test, batch: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(Faces(train, True), batch_size=batch, shuffle=True, generator=g),
            DataLoader(Faces(val), batch_size=batch), DataLoader(Faces(test), batch_size=batch))


@torch.no_grad()
def evaluate(model, loader, device: str, is_gan: bool = False):
    model.eval()
    actual, predicted = [], []
    for x, y in loader:
        out = model(x.to(device))
        logits = out[1] if is_gan else out
        predicted.extend(logits.argmax(1).cpu().numpy().tolist())
        actual.extend(y.numpy().tolist())
    cm = np.zeros((4, 4), dtype=int)
    for a, p in zip(actual, predicted):
        cm[a, p] += 1
    recall, precision, f1 = [], [], []
    for i in range(4):
        tp = cm[i, i]
        p = tp / max(1, cm[:, i].sum())
        r = tp / max(1, cm[i, :].sum())
        precision.append(float(p)); recall.append(float(r))
        f1.append(float(2 * p * r / (p + r)) if p + r else 0.0)
    return {"accuracy": float(np.trace(cm) / max(1, cm.sum())),
            "macro_f1": float(np.mean(f1)),
            "balanced_accuracy": float(np.mean(recall)),
            "precision_by_class": dict(zip(CLASSES, precision)),
            "recall_by_class": dict(zip(CLASSES, recall)),
            "f1_by_class": dict(zip(CLASSES, f1)),
            "confusion_matrix": cm.tolist(),
            "labels": list(CLASSES), "actual": actual, "predicted": predicted}


def train_cnn(train_loader, val_loader, epochs, device, lr):
    model = CNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best, best_loss, best_state = -1.0, float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        model.train(); losses=[]
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward(); opt.step(); losses.append(loss.item())
        val = evaluate(model, val_loader, device)
        avg = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": avg, "val_accuracy": val["accuracy"]})
        if val["accuracy"] > best or (val["accuracy"] == best and avg < best_loss):
            best, best_loss = val["accuracy"], avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, history


def train_gan(train_loader, val_loader, epochs, device, lr):
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    g_opt = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    adversarial, classification = nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()
    best, best_loss, best_state = -1.0, float("inf"), None
    history = []
    for epoch in range(1, epochs + 1):
        generator.train(); discriminator.train(); d_losses=[]; g_losses=[]
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            n = len(y)
            synthetic_labels = torch.randint(0, 4, (n,), device=device)
            z = torch.randn(n, LATENT, device=device)
            fake = generator(z, synthetic_labels)
            # Train discriminator: authentic expression classes + real/fake detection.
            d_opt.zero_grad(set_to_none=True)
            real_rf, real_cls = discriminator(x)
            fake_rf, _ = discriminator(fake.detach())
            d_loss = (adversarial(real_rf, torch.ones_like(real_rf)) +
                      adversarial(fake_rf, torch.zeros_like(fake_rf)) +
                      classification(real_cls, y))
            d_loss.backward(); d_opt.step()
            # Train conditional generator to create class-consistent expressions.
            g_opt.zero_grad(set_to_none=True)
            fake_rf, fake_cls = discriminator(fake)
            g_loss = adversarial(fake_rf, torch.ones_like(fake_rf)) + classification(fake_cls, synthetic_labels)
            g_loss.backward(); g_opt.step()
            d_losses.append(d_loss.item()); g_losses.append(g_loss.item())
        val = evaluate(discriminator, val_loader, device, True)
        avg = float(np.mean(d_losses))
        history.append({"epoch": epoch, "train_loss": avg,
                        "generator_loss": float(np.mean(g_losses)), "val_accuracy": val["accuracy"]})
        if val["accuracy"] > best or (val["accuracy"] == best and avg < best_loss):
            best, best_loss = val["accuracy"], avg
            best_state = {k: v.detach().cpu().clone() for k, v in discriminator.state_dict().items()}
    discriminator.load_state_dict(best_state)
    return discriminator, generator, history


def save_visuals(path: Path, model_name: str, metrics: dict, history: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(4), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("Actual label")
    ax.set_title(f"{model_name} — unseen subject")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout(); fig.savefig(path / f"{model_name}_confusion.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot([v["epoch"] for v in history], [v["train_loss"] for v in history], label="Train loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(model_name)
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path / f"{model_name}_loss.png", dpi=140); plt.close(fig)


def one_fold(rows, fold, args, device):
    seed_everything(args.seed + fold)
    train, val, test = split_subjects(rows, fold)
    fold_path = Path(args.output) / f"fold_{fold}"
    fold_path.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = make_loaders(train, val, test, args.batch_size, args.seed + fold)
    payload = {"fold": fold, "counts": {"train": len(train), "val": len(val), "test": len(test)},
               "subjects": {"train": sorted({x["subject"] for x in train}),
                            "val": sorted({x["subject"] for x in val}),
                            "test": sorted({x["subject"] for x in test})}, "models": {}}
    for name in ("cnn", "gan"):
        if name == "cnn":
            model, history = train_cnn(train_loader, val_loader, args.cnn_epochs, device, args.lr)
        else:
            model, generator, history = train_gan(train_loader, val_loader, args.gan_epochs, device, args.lr)
            torch.save({"model": generator.state_dict(), "latent": LATENT}, fold_path / "generator.pt")
        metrics = evaluate(model, test_loader, device, name == "gan")
        predictions = [{"filename": row["filename"], "true": CLASSES[y], "predicted": CLASSES[p]}
                       for row, y, p in zip(test, metrics.pop("actual"), metrics.pop("predicted"))]
        payload["models"][name] = {**metrics, "test_predictions": predictions}
        torch.save({"model": model.state_dict(), "type": name, "classes": CLASSES, "size": SIZE},
                   fold_path / f"{name}.pt")
        (fold_path / f"{name}_history.json").write_text(json.dumps(history, indent=2))
        save_visuals(fold_path, name, metrics, history)
        print(f"Fold {fold}: {name.upper()} test accuracy {metrics['accuracy']:.3f}, macro F1 {metrics['macro_f1']:.3f}", flush=True)
    (fold_path / "metrics.json").write_text(json.dumps(payload, indent=2))
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--fold", default="all", help="0-5, or all (leave-one-subject-out testing)")
    parser.add_argument("--cnn-epochs", type=int, default=35)
    parser.add_argument("--gan-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    for name in ("cnn_epochs", "gan_epochs", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_','-')} must be >= 1")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    rows = discover(args.data)
    count = Counter(r["class"] for r in rows)
    print(f"Dataset: {len(rows)} images, {len(set(x['subject'] for x in rows))} subjects, class counts={dict(count)}; device={device}", flush=True)
    subjects = sorted({r["subject"] for r in rows})
    if args.fold == "all":
        folds = range(len(subjects))
    else:
        try:
            num = int(args.fold)
            if num < 0 or num >= len(subjects):
                raise ValueError
        except ValueError:
            parser.error(f"--fold must be 'all' or 0..{len(subjects)-1}")
        folds = (num,)
    args.output.mkdir(parents=True, exist_ok=True)
    out = [one_fold(rows, fold, args, device) for fold in folds]
    summary = []
    for p in out:
        for name, m in p["models"].items():
            summary.append({"fold": p["fold"], "test_subject": p["subjects"]["test"][0],
                            "model": name, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                            "balanced_accuracy": m["balanced_accuracy"]})
    with (args.output / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, summary[0].keys()); w.writeheader(); w.writerows(summary)
    aggregate = {name: {metric: {"mean": float(np.mean([r[metric] for r in summary if r["model"] == name])),
                                  "std": float(np.std([r[metric] for r in summary if r["model"] == name]))}
                        for metric in ("accuracy", "macro_f1", "balanced_accuracy")}
                 for name in ("cnn", "gan")}
    (args.output / "aggregate.json").write_text(json.dumps({"metrics": aggregate,
        "method": "Each test fold is one unseen subject; one separate subject for validation",
        "dataset": {"images": len(rows), "subjects": len(subjects), "counts": dict(count)},
        "config": vars(args) | {"data": str(args.data), "output": str(args.output)}}, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    return aggregate

if __name__ == "__main__":
    main()
