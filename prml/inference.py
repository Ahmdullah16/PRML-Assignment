"""Strict CPU checkpoint loading; no unsafe pickle fallback or random predictions."""
from pathlib import Path
import torch
from . import CLASSES, SIZE
from .models import CNN, Discriminator
from .data import image_tensor


def load_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Trained checkpoint unavailable")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if tuple(saved.get("classes", ())) != CLASSES or saved.get("size") != SIZE:
        raise ValueError("Checkpoint label mapping or image size is incompatible")
    if saved.get("type") not in ("cnn", "gan"):
        raise ValueError("Unrecognized checkpoint architecture")
    model = CNN() if saved["type"] == "cnn" else Discriminator()
    model.load_state_dict(saved["model"], strict=True)
    model.eval()
    return model, saved


@torch.inference_mode()
def predict(model, kind, source):
    model.eval()
    out = model(image_tensor(source).unsqueeze(0))
    logits = out if kind == "cnn" else out[1]
    probs = torch.softmax(logits[0], 0)
    if not torch.isfinite(probs).all():
        raise ValueError("Nonfinite model output")
    return dict(zip(CLASSES, probs.tolist()))
