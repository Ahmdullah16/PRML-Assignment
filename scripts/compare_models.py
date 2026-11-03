import argparse
import json
from pathlib import Path
from prml.visualization import render_results, render_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("results/historical_verified.json"))
    p.add_argument("--output", type=Path, default=Path("results/figures/historical"))
    a = p.parse_args()
    render_results(a.results, a.output)
    render_dataset(json.loads(Path("results/dataset.json").read_text()), a.output)
