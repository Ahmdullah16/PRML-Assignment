import argparse
from pathlib import Path
from prml.holdout import run_experiment, export_experiment

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data/seed42_zip"))
    p.add_argument("--output", type=Path, default=Path("results/runs/holdout_seed42"))
    p.add_argument("--export", action="store_true")
    a = p.parse_args()
    run_experiment(a.zip, a.data, a.output, Path("results/duplicate_review.json"))
    if a.export:
        export_experiment(a.output, Path("results/holdout_seed42"), Path("deployment"))
