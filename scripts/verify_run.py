"""Replay a completed corrected run and verify every OOF probability and metric."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from prml import CLASSES
from prml.data import discover, split_subjects, Faces
from prml.inference import load_checkpoint
from prml.train import evaluate, write_json
from prml.metrics import aggregate


def main(data, run, output):
    torch.set_num_threads(4)
    rows = discover(data)
    dataset = json.loads((run / "dataset.json").read_text())
    assert {r["filename"]: r["sha256"] for r in rows} == {r["filename"]: r["sha256"] for r in dataset["manifest"]}
    folds, checks = [], []
    for i in range(len(dataset["subjects"])):
        parts = split_subjects(rows, i)
        path = run / f"fold_{i}"
        f = json.loads((path / "metrics.json").read_text())
        expected = {k: sorted({r["subject"] for r in p}) for k, p in zip(("train", "val", "test"), parts)}
        assert f["subjects"] == expected
        for kind in ("cnn", "gan"):
            net, saved = load_checkpoint(path / f"{kind}.pt")
            assert saved["subjects"] == expected and saved["best_epoch"] == f["models"][kind]["best_epoch"]
            measured = evaluate(net, DataLoader(Faces(parts[2]), batch_size=8), "cpu", kind == "gan")
            preds = f["models"][kind]["test_predictions"]
            assert [r["filename"] for r in preds] == [r["filename"] for r in parts[2]]
            assert [r["true"] for r in preds] == [CLASSES[y] for y in measured["actual"]]
            assert [r["predicted"] for r in preds] == [CLASSES[y] for y in measured["predicted"]]
            assert np.allclose([r["probabilities"] for r in preds], measured["probabilities"], atol=1e-6)
            for key in ("accuracy", "macro_f1", "macro_precision", "macro_recall", "balanced_accuracy"):
                assert abs(measured[key] - f["models"][kind][key]) < 1e-10
            checks.append(dict(fold=i, model=kind, checkpoint_replayed=True, probabilities_verified=True))
        folds.append(f)
    computed = aggregate(folds)
    saved = json.loads((run / "aggregate.json").read_text())
    assert computed == saved["metrics"]
    write_json(output, dict(checks=checks, dataset_hashes_verified=True, aggregate_verified=True))
    print(f"Verified {len(checks)} model checkpoints, every OOF probability, dataset hashes and aggregate metrics")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--run", type=Path, default=Path("results/runs/corrected"))
    p.add_argument("--output", type=Path, default=Path("results/corrected_verification.json"))
    a = p.parse_args()
    main(a.data, a.run, a.output)
