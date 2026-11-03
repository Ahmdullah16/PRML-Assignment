"""Copy only metadata and statistical figures into the public results directory."""
import argparse
import json
from pathlib import Path
import shutil
from prml.train import write_json
from prml.visualization import render_results, render_dataset


def export(source, target):
    source, target = Path(source), Path(target)
    run = json.loads((source / "run.json").read_text())
    result = json.loads((source / "aggregate.json").read_text())
    if run["status"] != "complete" or result["smoke"]:
        raise ValueError("Only complete research runs may be exported")
    run["config"]["data"] = "private local dataset (not distributed)"
    run["config"]["output"] = "private local experiment directory"
    write_json(target / "run.json", run)
    write_json(target / "aggregate.json", result)
    for folder in source.glob("fold_*"):
        for file in folder.glob("*.json"):
            (target / folder.name).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(file, target / folder.name / file.name)
    render_results(target / "aggregate.json", Path("results/figures/corrected"))
    render_dataset(json.loads(Path("results/dataset.json").read_text()), Path("results/figures/corrected"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=Path("results/runs/corrected"))
    p.add_argument("--target", type=Path, default=Path("results/corrected"))
    a = p.parse_args()
    export(a.source, a.target)
