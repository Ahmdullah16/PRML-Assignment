"""Recompute legacy metrics and (optionally) replay every recovered checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from prml import CLASSES
from prml.data import discover, split_subjects, Faces
from prml.inference import load_checkpoint
from prml.metrics import classification_metrics, aggregate
from prml.train import evaluate, write_json


def verify(root, data=None, checkpoints=None):
    torch.set_num_threads(4)
    rows = discover(data) if data else None
    folds, checks = [], []
    files = sorted(Path(root).glob("fold_*/metrics.json"))
    if len(files) != 6:
        raise ValueError("Historical run must contain all six folds")
    for file in files:
        f = json.loads(file.read_text())
        if rows:
            parts = split_subjects(rows, f["fold"])
            expected = {k: sorted({r["subject"] for r in part}) for k, part in zip(("train", "val", "test"), parts)}
            if f["subjects"] != expected:
                raise ValueError("Historical subject manifest mismatch")
        for kind, model in f["models"].items():
            predictions = model["test_predictions"]
            if tuple(model["labels"]) != CLASSES:
                raise ValueError("Historical labels mismatch")
            recomputed = classification_metrics([CLASSES.index(r["true"]) for r in predictions],
                                                 [CLASSES.index(r["predicted"]) for r in predictions])
            for key in ("accuracy", "macro_f1", "balanced_accuracy", "confusion_matrix",
                        "precision_by_class", "recall_by_class", "f1_by_class"):
                if key == "confusion_matrix":
                    assert recomputed[key] == model[key]
                elif isinstance(recomputed[key], dict):
                    assert all(abs(v - model[key][k]) < 1e-10 for k, v in recomputed[key].items())
                else:
                    assert abs(recomputed[key] - model[key]) < 1e-10
            for r in predictions:
                r["subject"] = f["subjects"]["test"][0]
            model.update(recomputed)
            check = dict(fold=f["fold"], model=kind, arithmetic_verified=True, checkpoint_replayed=False)
            if checkpoints and rows:
                path = Path(checkpoints) / file.parent.name / f"{kind}.pt"
                net, saved = load_checkpoint(path)
                measured = evaluate(net, DataLoader(Faces(parts[2]), batch_size=8), "cpu", kind == "gan")
                by_name = {r["filename"]: r for r in predictions}
                assert set(by_name) == {r["filename"] for r in parts[2]}
                for row, a, pred in zip(parts[2], measured["actual"], measured["predicted"]):
                    assert by_name[row["filename"]]["true"] == CLASSES[a]
                    assert by_name[row["filename"]]["predicted"] == CLASSES[pred]
                check.update(checkpoint_replayed=True, checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            checks.append(check)
        folds.append(f)
    metrics = aggregate(folds)
    historical = json.loads((Path(root) / "aggregate.json").read_text())
    for name in ("cnn", "gan"):
        for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
            assert abs(metrics[name]["fold_summary"][metric]["mean"] - historical["metrics"][name][metric]["mean"]) < 1e-10
    return dict(provenance="Recovered results; arithmetic verified; checkpoint replay status below. Original training not rerun.",
                checks=checks, metrics=metrics, original_config=historical["config"],
                limitations=["Original environment and source hashes not recorded", "Generator and discriminator epochs were not paired",
                             "D classification loss omitted generated labels", "Seed reset only once per fold, before CNN then GAN"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/historical"))
    p.add_argument("--data", type=Path)
    p.add_argument("--checkpoints", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/historical_verified.json"))
    a = p.parse_args()
    result = verify(a.results, a.data, a.checkpoints)
    write_json(a.output, result)
    print(json.dumps(result["checks"], indent=2))
