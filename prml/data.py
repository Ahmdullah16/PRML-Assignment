"""Validate source images without silently dropping corrupt or ambiguous records."""
import hashlib
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from . import CLASSES, SIZE

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def image_tensor(source):
    with Image.open(source) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = im.resize((SIZE, SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).copy() / 127.5 - 1
    return torch.from_numpy(arr).permute(2, 0, 1)


def discover(root):
    root = Path(root).resolve()
    directories = {}
    for directory in [root, *sorted(p for p in root.rglob("*") if p.is_dir())]:
        name = re.sub(r"[ _-]+prml$", "", directory.name.lower())
        if name in CLASSES:
            if name in directories:
                raise ValueError(f"Multiple candidate folders for {name}; select a narrower root")
            directories[name] = directory
    if set(directories) != set(CLASSES):
        raise ValueError(f"Missing class folders: {set(CLASSES) - set(directories)}")
    rows, hashes = [], {}
    for label, name in enumerate(CLASSES):
        paths = sorted(p for p in directories[name].rglob("*") if p.suffix.lower() in EXTENSIONS)
        if not paths:
            raise ValueError(f"Empty class: {name}")
        for path in paths:
            match = re.fullmatch(r"(\d+)_[^_]+_[^_]+_([shnf])_[^_.]+\.[^.]+", path.name)
            if not match or match[2] != name[0]:
                raise ValueError(f"Ambiguous subject or inconsistent filename expression: {path.name}")
            try:
                with Image.open(path) as im:
                    im.verify()
                with Image.open(path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    im.load()
                    width, height = im.size
                    pixels = np.asarray(im)
                    pixel_hash = hashlib.sha256(pixels.tobytes() + str(im.size).encode()).hexdigest()
                    small = np.asarray(im.convert("L").resize((9, 8)), dtype=np.int16)
                    dhash = sum(int(v) << i for i, v in enumerate((small[:, 1:] > small[:, :-1]).flat))
            except Exception as exc:
                raise ValueError(f"Corrupt image: {path.name}") from exc
            if pixel_hash in hashes:
                raise ValueError(f"Exact duplicate images: {path.name}, {hashes[pixel_hash]}")
            hashes[pixel_hash] = path.name
            rows.append(dict(path=str(path), filename=path.name, label=label, **{"class": name},
                             subject=match[1], width=width, height=height, dhash=dhash,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return rows


def audit(rows):
    near = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if distance <= 5:
                near.append(dict(a=a["filename"], b=b["filename"], hamming=distance,
                                 same_subject=a["subject"] == b["subject"]))
    return dict(images=len(rows), classes=list(CLASSES), counts=dict(Counter(r["class"] for r in rows)),
                subjects=sorted({r["subject"] for r in rows}),
                dimensions=dict(Counter(f'{r["width"]}x{r["height"]}' for r in rows)),
                near_duplicates=near, near_duplicate_rule="64-bit dHash Hamming <=5; review candidates, not proof",
                manifest=[{k: v for k, v in r.items() if k not in ("path", "dhash")} for r in rows])


def split_subjects(rows, fold):
    subjects = sorted({r["subject"] for r in rows})
    if len(subjects) < 3 or not 0 <= fold < len(subjects):
        raise ValueError("Require at least three subjects and a valid fold")
    test, val = subjects[fold], subjects[(fold + 1) % len(subjects)]
    parts = ([r for r in rows if r["subject"] not in (test, val)],
             [r for r in rows if r["subject"] == val], [r for r in rows if r["subject"] == test])
    if any({r["label"] for r in part} != set(range(4)) for part in parts):
        raise ValueError("All four classes must occur in every partition")
    return parts


class Faces(Dataset):
    # Cached preprocessing and train-only horizontal flips reused from original_train.py.
    def __init__(self, rows, augment=False):
        self.rows, self.augment = rows, augment
        self.images = [image_tensor(r["path"]) for r in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.images[i]
        if self.augment and torch.rand(()) < .5:
            x = x.flip(2)
        return x, self.rows[i]["label"]
