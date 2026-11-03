"""Run a trained fold checkpoint on one new image (illustrative, not diagnostic)."""
import argparse
from pathlib import Path
import torch
from .train import CNN, Discriminator, image_tensor


def predict(checkpoint: Path, photo: Path):
    try:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(checkpoint, map_location="cpu")
    model = CNN() if saved["type"] == "cnn" else Discriminator()
    model.load_state_dict(saved["model"]); model.eval()
    with torch.no_grad():
        out = model(image_tensor(str(photo)).unsqueeze(0))
        logits = out if saved["type"] == "cnn" else out[1]
        scores = torch.softmax(logits[0], dim=0).tolist()
    return dict(zip(saved["classes"], [round(v, 4) for v in scores]))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    a = p.parse_args()
    print(predict(a.checkpoint, a.image))
