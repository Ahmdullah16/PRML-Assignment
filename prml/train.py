"""Controlled LOSO experiment, reusing the recovered architectures and preprocessing."""
import argparse
import copy
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from . import CLASSES, SIZE, LATENT
from .data import discover, audit, split_subjects, Faces
from .models import CNN, Generator, Discriminator
from .metrics import classification_metrics, aggregate


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def loaders(parts, batch, seed):
    return [DataLoader(Faces(rows, augment=i == 0), batch_size=batch, shuffle=i == 0,
                       generator=torch.Generator().manual_seed(seed), num_workers=0)
            for i, rows in enumerate(parts)]


@torch.inference_mode()
def evaluate(model, loader, device, is_gan=False):
    model.eval()
    actual, pred, probabilities, total_loss = [], [], [], 0.
    for x, y in loader:
        out = model(x.to(device))
        logits = out[1] if is_gan else out
        total_loss += nn.functional.cross_entropy(logits, y.to(device), reduction="sum").item()
        probabilities.extend(logits.softmax(1).cpu().tolist())
        actual.extend(y.tolist())
        pred.extend(logits.argmax(1).cpu().tolist())
    return classification_metrics(actual, pred) | dict(loss=total_loss / len(actual),
                                                       actual=actual, predicted=pred, probabilities=probabilities)


def discriminator_loss(real_out, fake_out, real_labels, fake_labels):
    bce, ce = nn.functional.binary_cross_entropy_with_logits, nn.functional.cross_entropy
    return (bce(real_out[0], torch.ones_like(real_out[0])) +
            bce(fake_out[0], torch.zeros_like(fake_out[0])) +
            ce(real_out[1], real_labels) + ce(fake_out[1], fake_labels))


def generator_loss(fake_out, labels):
    # Non-saturating source objective, explicitly differs from literal minimax AC-GAN.
    return (nn.functional.binary_cross_entropy_with_logits(fake_out[0], torch.ones_like(fake_out[0])) +
            nn.functional.cross_entropy(fake_out[1], labels))


def train_model(kind, train_loader, val_loader, epochs, device, lr):
    model = (CNN() if kind == "cnn" else Discriminator()).to(device)
    generator = Generator().to(device) if kind == "gan" else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999) if kind == "cnn" else (.5, .999))
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(.5, .999)) if generator else None
    best_loss, best_epoch, best, best_g = float("inf"), 0, None, None
    history, started = [], time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        if generator:
            generator.train()
        losses, g_losses, correct, count = [], [], 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if kind == "cnn":
                out = model(x)
                loss = nn.functional.cross_entropy(out, y)
                correct += (out.argmax(1) == y).sum().item()
            else:
                labels = torch.randint(0, 4, (len(y),), device=device)
                fake = generator(torch.randn(len(y), LATENT, device=device), labels)
                real_out = model(x)
                loss = discriminator_loss(real_out, model(fake.detach()), y, labels)
                correct += (real_out[1].argmax(1) == y).sum().item()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            if generator:
                # Freeze D weights while retaining dD/dx for the generator update.
                optimizer.zero_grad(set_to_none=True)
                for p in model.parameters():
                    p.requires_grad_(False)
                g_optimizer.zero_grad(set_to_none=True)
                g_loss = generator_loss(model(fake), labels)
                if not torch.isfinite(g_loss):
                    raise FloatingPointError("Nonfinite generator loss")
                g_loss.backward()
                g_optimizer.step()
                for p in model.parameters():
                    p.requires_grad_(True)
                g_losses.append((g_loss.item(), len(y)))
            losses.append((loss.item(), len(y)))
            count += len(y)
        val = evaluate(model, val_loader, device, kind == "gan")
        row = dict(epoch=epoch, train_loss=sum(v * n for v, n in losses) / count,
                   train_accuracy=correct / count, val_accuracy=val["accuracy"], val_loss=val["loss"])
        if generator:
            row["generator_loss"] = sum(v * n for v, n in g_losses) / count
        history.append(row)
        # Only validation CE selects the checkpoint; no test access here.
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_g = {k: v.detach().cpu().clone() for k, v in generator.state_dict().items()} if generator else None
    model.load_state_dict(best)
    if generator:
        generator.load_state_dict(best_g)
        generator.eval()
    return model, generator, history, best_epoch, time.perf_counter() - started


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/runs/corrected"))
    parser.add_argument("--cnn-epochs", type=int, default=25)
    parser.add_argument("--gan-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=.0002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--duplicate-review", type=Path, help="JSON with reviewed cross-subject SHA256 pairs")
    parser.add_argument("--smoke", action="store_true", help="One fold only, never a research result")
    args = parser.parse_args(argv)
    if min(args.cnn_epochs, args.gan_epochs, args.batch_size, args.threads) < 1 or args.lr <= 0:
        parser.error("Epochs, batch size, threads and learning rate must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if (args.output / "run.json").exists():
        raise FileExistsError("Output already contains a run; choose a new directory to preserve provenance")
    torch.set_num_threads(args.threads)
    rows = discover(args.data)
    dataset = audit(rows)
    # Cross-subject perceptual matches require human review; no silent removal.
    cross = [r for r in dataset["near_duplicates"] if not r["same_subject"]]
    if cross:
        reviewed = json.loads(args.duplicate_review.read_text())["distinct_pairs"] if args.duplicate_review else []
        hashes = {r["filename"]: r["sha256"] for r in rows}
        if any(sorted([hashes[r["a"]], hashes[r["b"]]]) not in reviewed for r in cross):
            raise ValueError(f"Review {len(cross)} cross-subject near duplicates before training; supply --duplicate-review")
    write_json(args.output / "dataset.json", dataset)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (subprocess.SubprocessError, OSError):
        commit = "unavailable"
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    write_json(args.output / "run.json", dict(status="running", config=config, commit=commit, source_sha256=source_hashes,
               environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__,
                                platform=platform.platform(), device=device, threads=args.threads,
                                cpu=platform.processor(), gpu=torch.cuda.get_device_name() if device == "cuda" else None)))
    folds = []
    for fold in range(1 if args.smoke else len(dataset["subjects"])):
        parts = split_subjects(rows, fold)
        manifest = {name: sorted({r["subject"] for r in part}) for name, part in zip(("train", "val", "test"), parts)}
        fold_path = args.output / f"fold_{fold}"
        payload = dict(fold=fold, subjects=manifest, counts=dict(zip(("train", "val", "test"), map(len, parts))), models={})
        for kind in ("cnn", "gan"):
            # Recreate identical shuffle/augmentation RNG state for both methods.
            seed_everything(args.seed + fold)
            train_loader, val_loader, test_loader = loaders(parts, args.batch_size, args.seed + fold)
            model, generator, history, best_epoch, seconds = train_model(
                kind, train_loader, val_loader, getattr(args, "cnn_epochs" if kind == "cnn" else "gan_epochs"), device, args.lr)
            metrics = evaluate(model, test_loader, device, kind == "gan")
            predictions = [dict(filename=r["filename"], subject=r["subject"], true=CLASSES[y], predicted=CLASSES[p], probabilities=probs)
                           for r, y, p, probs in zip(parts[2], metrics.pop("actual"), metrics.pop("predicted"), metrics.pop("probabilities"))]
            # Latency on already preprocessed image: synchronized and warmed, excludes file I/O.
            model.eval()
            sample = next(iter(test_loader))[0][:1].to(device)
            with torch.inference_mode():
                for _ in range(5):
                    model(sample)
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(30):
                    model(sample)
                if device == "cuda":
                    torch.cuda.synchronize()
                latency = 1000 * (time.perf_counter() - t0) / 30
            payload["models"][kind] = metrics | dict(test_predictions=predictions, best_epoch=best_epoch,
                    training_seconds=seconds, inference_ms=latency, parameters=sum(p.numel() for p in model.parameters()),
                    generator_parameters=sum(p.numel() for p in generator.parameters()) if generator else 0)
            fold_path.mkdir(parents=True, exist_ok=True)
            saved = dict(model={k: v.cpu() for k, v in model.state_dict().items()}, type=kind, classes=CLASSES, size=SIZE,
                         seed=args.seed + fold, best_epoch=best_epoch, subjects=manifest, experiment="corrected-acgan-v1")
            torch.save(saved, fold_path / f"{kind}.pt")
            if generator:
                torch.save(dict(model={k: v.cpu() for k, v in generator.state_dict().items()}, classes=CLASSES,
                                size=SIZE, latent=LATENT, best_epoch=best_epoch), fold_path / "generator.pt")
            write_json(fold_path / f"{kind}_history.json", history)
            print(f'Fold {fold} {kind}: accuracy={metrics["accuracy"]:.4f}, selected epoch={best_epoch}, {seconds:.1f}s', flush=True)
        folds.append(payload)
        write_json(fold_path / "metrics.json", payload)
    write_json(args.output / "aggregate.json", dict(provenance="newly executed corrected experiment", smoke=args.smoke,
                dataset=dataset | {"manifest": "dataset.json"}, metrics=aggregate(folds)))
    run = json.loads((args.output / "run.json").read_text())
    write_json(args.output / "run.json", run | {"status": "complete"})
    return folds


if __name__ == "__main__":
    main()
