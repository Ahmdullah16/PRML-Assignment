"""Audit local Kaggle images and create seed-42, duplicate-group-safe manifests.

No facial pixels are exported into public results. Test membership takes priority.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

FOLDERS = {'happy': 'happiness', 'sad': 'sadness', 'fear': 'fear', 'neutral': 'neutral'}
LABELS = ('sadness', 'happiness', 'neutral', 'fear')

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding='utf-8')

def prepare(root, output):
    rows, corrupt, small, hashes = [], [], [], defaultdict(list)
    counts, dimensions, formats = Counter(), Counter(), Counter()
    for partition in ('train_dir', 'test_dir'):
        for folder in sorted((root / partition).iterdir()):
            if not folder.is_dir():
                continue
            files = sorted(p for p in folder.rglob('*') if p.is_file())
            counts[f'{partition}/{folder.name}'] = len(files)
            if folder.name not in FOLDERS:
                continue
            for p in files:
                relative = p.relative_to(root).as_posix()
                try:
                    with Image.open(p) as im:
                        im.verify()
                    with Image.open(p) as im:
                        fmt = im.format
                        im = ImageOps.exif_transpose(im).convert('RGB')
                        im.load()
                        dimensions[str(im.size)] += 1
                        formats[str(fmt)] += 1
                        pixels = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
                        gray = im.convert('L')
                        a = np.asarray(gray.resize((9, 8)), dtype=np.int16)
                        dhash = int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), 'big')
                        small.append(np.asarray(gray.resize((32, 32)), dtype=np.float32).ravel() / 255.)
                    row = dict(file=relative, source_partition=partition, label=LABELS.index(FOLDERS[folder.name]),
                               expression=FOLDERS[folder.name], sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                               pixel_sha256=pixels, dhash=f'{dhash:016x}')
                    hashes[pixels].append(len(rows))
                    rows.append(row)
                except Exception as exc:
                    corrupt.append(dict(file=relative, error=f'{type(exc).__name__}: image decoding/verification failed'))
    n = len(rows)
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(b)] = find(a)
    exact = []
    for ids in hashes.values():
        if len(ids) > 1:
            exact.append([rows[i]['file'] for i in ids])
            for i in ids[1:]:
                union(ids[0], i)
    # Blockwise exhaustive dHash search, followed by a pixel-distance check.
    # Conservative heuristic groups candidates; it does not establish identity.
    dh = np.array([int(r['dhash'], 16) for r in rows], dtype=np.uint64)
    lookup = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    thumb = np.stack(small)
    near = []
    for i in range(n):
        distances = lookup[np.bitwise_xor(dh[i], dh[i+1:]).view(np.uint8).reshape(-1, 8)].sum(1)
        for j in (np.flatnonzero(distances <= 3) + i + 1):
            j = int(j)
            if rows[i]['pixel_sha256'] == rows[j]['pixel_sha256']:
                continue
            rmse = float(np.sqrt(np.mean((thumb[i] - thumb[j]) ** 2)))
            if rmse <= .06:
                union(i, j)
                near.append(dict(a=rows[i]['file'], b=rows[j]['file'], hamming=int(distances[j-i-1]), rmse=rmse))
        if i % 5000 == 0:
            print(f'Near-duplicate audit {i}/{n}', flush=True)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    eligible_train, eligible_test, excluded = [], [], []
    for ids in groups.values():
        gid = min(rows[i]['sha256'] for i in ids)
        has_test = any(rows[i]['source_partition'] == 'test_dir' for i in ids)
        seen = set()
        for i in ids:
            row = rows[i] | {'duplicate_group': gid}
            if row['source_partition'] == 'test_dir':
                eligible_test.append(row | {'partition': 'test'})
            elif has_test:
                excluded.append(row | {'reason': 'duplicate_or_conservative_near_duplicate_group_overlaps_test'})
            elif len({rows[j]['label'] for j in ids}) > 1:
                excluded.append(row | {'reason': 'conflicting_labels_in_duplicate_group'})
            elif row['pixel_sha256'] in seen:
                excluded.append(row | {'reason': 'redundant_exact_training_duplicate'})
            else:
                eligible_train.append(row)
                seen.add(row['pixel_sha256'])
    # Stratify whole duplicate groups within each class, approximately 15% validation.
    rng = np.random.default_rng(42)
    train, validation = [], []
    for label in range(4):
        grouped = defaultdict(list)
        for r in eligible_train:
            if r['label'] == label:
                grouped[r['duplicate_group']].append(r)
        keys = sorted(grouped)
        rng.shuffle(keys)
        target = round(sum(map(len, grouped.values())) * .15)
        total = 0
        for key in keys:
            part = 'validation' if total < target else 'train'
            dest = validation if part == 'validation' else train
            dest.extend(r | {'partition': part} for r in grouped[key])
            if part == 'validation':
                total += len(grouped[key])
    parts = {'train': sorted(train, key=lambda r:r['file']), 'validation': sorted(validation, key=lambda r:r['file']),
             'test': sorted(eligible_test, key=lambda r:r['file'])}
    for key in ('pixel_sha256', 'duplicate_group', 'file'):
        sets = [set(r[key] for r in p) for p in parts.values()]
        assert not any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3))
    stats = dict(dataset='dilkushsingh/facial-emotion-dataset', seed=42, validation_fraction=.15,
        source_url='https://www.kaggle.com/datasets/dilkushsingh/facial-emotion-dataset',
        upstream='https://github.com/dilkushsingh/Facial_Emotion_Dataset',
        version='Local snapshot; Kaggle version ID unavailable; exact files identified by SHA256 manifest',
        source_counts=dict(counts), selected_decodable=n, corrupt=corrupt, dimensions=dict(dimensions), formats=dict(formats),
        exact_duplicate_groups=exact, near_duplicate_pairs=near, excluded_training=excluded,
        near_duplicate_rule='64-bit dHash Hamming <=3 AND grayscale 32x32 RMSE <=0.06; conservative grouping, not subject identity',
        counts={p: dict(Counter(r['expression'] for r in v)) for p,v in parts.items()},
        totals={p:len(v) for p,v in parts.items()}, labels=list(LABELS), subject_ids_available=False,
        limitations='Image-level experiment; subject independence unavailable. Heuristic cannot rule out all near duplicates. Test duplicates retained; no test image moved into training.',
        preprocessing='EXIF transpose, RGB, bilinear 64x64, float32 /127.5 -1; training-only random horizontal flip',
        metadata_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'LICENSE',root/'README.md') if p.exists()})
    write(output/'dataset.json', stats)
    write(output/'split_manifest.json', dict(labels=list(LABELS), seed=42, partitions=parts))
    print(json.dumps(stats['totals']), flush=True)
    return parts, stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('results/kaggle_seed42'))
    a = parser.parse_args()
    prepare(a.data, a.output)
