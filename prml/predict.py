import argparse
import json
from pathlib import Path
from .inference import load_checkpoint, predict

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    a = p.parse_args()
    model, saved = load_checkpoint(a.checkpoint)
    print(json.dumps(predict(model, saved["type"], a.image), indent=2))
