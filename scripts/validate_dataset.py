import argparse
import json
from pathlib import Path
from prml.data import discover, audit

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/dataset.json"))
    a = p.parse_args()
    report = audit(discover(a.data))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "manifest"}, indent=2))
