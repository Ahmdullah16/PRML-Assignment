"""Seed-42 shuffled subject holdout with separately evaluated train/validation/test sets."""
import hashlib
import json
import platform
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from torch.utils.data import DataLoader
from . import CLASSES, SIZE
from .data import discover, audit, Faces, EXTENSIONS
from .train import seed_everything, loaders, train_model, evaluate, write_json
from .inference import load_checkpoint

SPLIT_NAMES = ("train", "validation", "test")


def extract_dataset(archive, destination):
    """Extract only class images, rejecting traversal, duplicates and unexpected image paths."""
    archive, destination = Path(archive), Path(destination).resolve()
    members, seen = [], set()
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            name = PurePosixPath(entry.filename.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in entry.filename:
                raise ValueError(f"Unsafe ZIP member: {entry.filename}")
            if entry.is_dir() or name.suffix.lower() not in EXTENSIONS:
                continue
            if len(name.parts) != 2 or name.parts[0].lower() not in CLASSES:
                raise ValueError(f"Unexpected image path in ZIP: {entry.filename}")
            relative = Path(name.parts[0].lower()) / name.name
            if str(relative).lower() in seen:
                raise ValueError("Duplicate ZIP image path")
            seen.add(str(relative).lower())
            members.append((entry, relative))
        if not members or sum(e.file_size for e, _ in members) > 512 * 1024**2:
            raise ValueError("Dataset ZIP is empty or exceeds the supported image size budget")
        for entry, relative in members:
            target = destination / relative
            payload = z.read(entry)
            if target.exists() and target.read_bytes() != payload:
                raise FileExistsError("Extraction target contains different data; select a new destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(payload)
    return destination


def shuffled_subject_split(rows, seed=42):
    """Hamilton rounding of 50:30:20 across identities; six identities become 3:2:1."""
    subjects = np.array(sorted({r["subject"] for r in rows}))
    if len(subjects) < 3:
        raise ValueError("At least three subjects are required")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects).tolist()
    quotas = np.array([.5, .3, .2]) * len(subjects)
    counts = np.floor(quotas).astype(int)
    for i in np.argsort(-(quotas - counts), kind="stable")[:len(subjects) - counts.sum()]:
        counts[i] += 1
    if (counts == 0).any():
        raise ValueError("Requested ratio leaves an empty subject partition")
    boundaries = np.cumsum(counts)
    groups = dict(zip(SPLIT_NAMES, (shuffled[:boundaries[0]], shuffled[boundaries[0]:boundaries[1]], shuffled[boundaries[1]:])))
    parts = {}
    for name, ids in groups.items():
        part = [r for r in rows if r["subject"] in ids]
        rng.shuffle(part)
        if {r["label"] for r in part} != set(range(4)):
            raise ValueError(f"All four classes must occur in {name}")
        parts[name] = part
    keys = [{r["filename"] for r in part} for part in parts.values()]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    manifest = dict(seed=seed, method="NumPy default_rng seed42; shuffled subject holdout; largest-remainder rounding",
                    requested_ratios=dict(zip(SPLIT_NAMES, [.5, .3, .2])), shuffled_subject_order=shuffled,
                    partitions={name: dict(subjects=groups[name], images=len(part), actual_ratio=len(part) / len(rows),
                         class_counts=dict(Counter(r["class"] for r in part)),
                         records=[{k: r[k] for k in ("filename", "subject", "class", "label", "sha256")} for r in part])
                         for name, part in parts.items()})
    return parts, manifest


def shuffled_class_split(rows, seed=42):
    """Shuffle each 12-image class, take 6/3/3, then independently shuffle each combined set."""
    parts = {name: [] for name in SPLIT_NAMES}
    class_orders = {}
    # Separate streams make class shuffling and final partition mixing independent and reproducible.
    streams = np.random.SeedSequence(seed).spawn(len(CLASSES) + len(SPLIT_NAMES))
    for i, cls in enumerate(CLASSES):
        group = sorted((r for r in rows if r["class"] == cls), key=lambda r: r["filename"])
        if len(group) != 12:
            raise ValueError(f"Expected exactly 12 images for {cls}; found {len(group)}")
        np.random.default_rng(streams[i]).shuffle(group)
        class_orders[cls] = [r["filename"] for r in group]
        for name, subset in zip(SPLIT_NAMES, (group[:6], group[6:9], group[9:])):
            parts[name].extend(subset)
    for i, name in enumerate(SPLIT_NAMES):
        np.random.default_rng(streams[len(CLASSES) + i]).shuffle(parts[name])
    images = [{r["filename"] for r in parts[n]} for n in SPLIT_NAMES]
    assert not (images[0] & images[1] or images[0] & images[2] or images[1] & images[2])
    subjects = {n: {r["subject"] for r in parts[n]} for n in SPLIT_NAMES}
    overlaps = {f"{a}_vs_{b}": sorted(subjects[a] & subjects[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    manifest = dict(seed=seed, method="Per-class 6/3/3 image split; independent SeedSequence(42) streams per class and final partition",
        requested_ratios=dict(zip(SPLIT_NAMES, [.5, .25, .25])), per_class_shuffled_order=class_orders,
        subject_independent=False, subject_overlap=overlaps,
        partitions={name: dict(subjects=sorted(subjects[name]), images=len(part), actual_ratio=len(part) / len(rows),
            class_counts=dict(Counter(r["class"] for r in part)),
            records=[{k: r[k] for k in ("filename", "subject", "class", "label", "sha256")} for r in part])
            for name, part in parts.items()})
    return parts, manifest


def prepare_dataset(archive, destination, seed=42, strategy="subject"):
    data = extract_dataset(archive, destination)
    rows = discover(data)
    if strategy not in ("subject", "class_6_3_3"):
        raise ValueError("Unknown split strategy")
    parts, manifest = (shuffled_subject_split(rows, seed) if strategy == "subject" else shuffled_class_split(rows, seed))
    return rows, parts, manifest


def run_experiment(archive, data_dir, output, review, seed=42, cnn_epochs=25, gan_epochs=40, device="cpu", strategy="subject"):
    """Reuse only a hash/config-identical completed run; otherwise execute fixed-budget training."""
    torch.set_num_threads(4)
    output = Path(output)
    rows, parts, manifest = prepare_dataset(archive, data_dir, seed, strategy)
    config = dict(seed=seed, cnn_epochs=cnn_epochs, gan_epochs=gan_epochs, batch_size=8, lr=.0002,
                  device=device, selection="minimum validation classification cross-entropy", requested_ratio="50:30:20")
    if strategy == "class_6_3_3":
        config.update(requested_ratio="6/3/3 images per class (50:25:25)", split_strategy=strategy)
    archive_sha = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    summary_path = output / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["config"] != config or summary["archive_sha256"] != archive_sha or summary["split_manifest"] != manifest:
            raise ValueError("Existing run does not match requested ZIP/configuration/split; use a new output directory")
        verify_checkpoints(parts, output, summary)
        return summary
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Incomplete/nonempty output directory; preserve it and choose a new run directory")
    audited = audit(rows)
    approved = json.loads(Path(review).read_text())["distinct_pairs"]
    hashes = {r["filename"]: r["sha256"] for r in rows}
    for pair in audited["near_duplicates"]:
        if not pair["same_subject"] and sorted([hashes[pair["a"]], hashes[pair["b"]]]) not in approved:
            raise ValueError("Cross-subject perceptual match requires a hash-bound review")
    write_json(output / "split_manifest.json", manifest)
    write_json(output / "dataset.json", audited)
    experiment = "seed42-class-6-3-3-v1" if strategy == "class_6_3_3" else "seed42-subject-holdout-v1"
    summary = dict(experiment=experiment, archive_sha256=archive_sha, config=config,
                   split_manifest=manifest, models={}, environment=dict(python=platform.python_version(),
                     torch=str(torch.__version__), numpy=np.__version__, platform=platform.platform(),
                     cpu=platform.processor(), device=device, threads=4),
                   source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                  for name in ("holdout.py", "train.py", "models.py", "data.py", "metrics.py")})
    for kind in ("cnn", "gan"):
        seed_everything(seed)
        training, validation, _ = loaders([parts[n] for n in SPLIT_NAMES], 8, seed)
        net, generator, history, epoch, seconds = train_model(kind, training, validation,
            cnn_epochs if kind == "cnn" else gan_epochs, device, .0002)
        saved = dict(model={k: v.cpu() for k, v in net.state_dict().items()}, type=kind, classes=CLASSES, size=SIZE,
                     seed=seed, best_epoch=epoch, experiment=summary["experiment"],
                     subjects={name: manifest["partitions"][name]["subjects"] for name in SPLIT_NAMES},
                     archive_sha256=archive_sha)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(saved, output / f"{kind}.pt")
        # Generator is private and is never needed by public classification.
        if generator:
            torch.save(dict(model={k: v.cpu() for k, v in generator.state_dict().items()}, classes=CLASSES,
                            latent=64, size=SIZE, best_epoch=epoch), output / "generator.pt")
        model_result = dict(best_epoch=epoch, training_seconds=seconds,
                            classifier_parameters=sum(p.numel() for p in net.parameters()),
                            generator_parameters=sum(p.numel() for p in generator.parameters()) if generator else 0,
                            partitions={})
        for name, part in parts.items():
            # All three final evaluations are in eval mode, without augmentation.
            measured = evaluate(net, DataLoader(Faces(part), batch_size=8, shuffle=False), device, kind == "gan")
            predictions = [dict(filename=r["filename"], subject=r["subject"], true=CLASSES[y],
                                predicted=CLASSES[p], probabilities=prob)
                           for r, y, p, prob in zip(part, measured.pop("actual"), measured.pop("predicted"), measured.pop("probabilities"))]
            model_result["partitions"][name] = measured | dict(predictions=predictions)
        summary["models"][kind] = model_result
        write_json(output / f"{kind}_history.json", history)
        print(f'{kind}: selected epoch {epoch}; ' + ", ".join(
            f'{name} accuracy={model_result["partitions"][name]["accuracy"]:.3f}' for name in SPLIT_NAMES), flush=True)
    write_json(summary_path, summary)
    verify_checkpoints(parts, output, summary)
    return summary


def verify_checkpoints(parts, output, summary):
    checks = []
    for kind in ("cnn", "gan"):
        net, saved = load_checkpoint(Path(output) / f"{kind}.pt")
        assert saved["seed"] == summary["config"]["seed"]
        assert saved["archive_sha256"] == summary["archive_sha256"]
        for name, rows in parts.items():
            measured = evaluate(net, DataLoader(Faces(rows), batch_size=8), "cpu", kind == "gan")
            expected = summary["models"][kind]["partitions"][name]
            assert [CLASSES[y] for y in measured["actual"]] == [r["true"] for r in expected["predictions"]]
            assert [CLASSES[p] for p in measured["predicted"]] == [r["predicted"] for r in expected["predictions"]]
            assert np.allclose(measured["probabilities"], [r["probabilities"] for r in expected["predictions"]], atol=1e-6)
            for metric in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "loss"):
                assert np.isclose(measured[metric], expected[metric], atol=1e-6)
            checks.append(dict(model=kind, partition=name, images=len(rows), replay_verified=True))
    write_json(Path(output) / "verification.json", dict(checks=checks))


def export_experiment(run_dir, public_dir, deployment_dir):
    run_dir, public_dir, deployment_dir = map(Path, (run_dir, public_dir, deployment_dir))
    public_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "split_manifest.json", "verification.json", "cnn_history.json", "gan_history.json"):
        shutil.copyfile(run_dir / name, public_dir / name)
    # Minimal, CPU classifier snapshots only; never publish the generator or original images.
    files = {}
    for kind in ("cnn", "gan"):
        shutil.copyfile(run_dir / f"{kind}.pt", deployment_dir / f"{kind}.pt")
        path = deployment_dir / f"{kind}.pt"
        files[kind] = dict(file=path.name, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary = json.loads((run_dir / "summary.json").read_text())
    write_json(deployment_dir / "manifest.json", dict(experiment=summary["experiment"], files=files,
        note="Validation-selected classifier weights for public inference; no generator or source images"))
