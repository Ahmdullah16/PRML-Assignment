import zipfile
from collections import Counter
import numpy as np
import pytest
from prml.holdout import shuffled_subject_split, shuffled_class_split, extract_dataset


def test_seed42_subject_split():
    rows = [dict(subject=f"{i:03}", filename=f"{i:03}_{label}_{rep}.jpg", label=label,
                 **{"class": ("sadness", "happiness", "neutral", "fear")[label]}, sha256=f"hash{i}{label}{rep}")
            for i in range(6) for label in range(4) for rep in range(2)]
    parts, manifest = shuffled_subject_split(rows, 42)
    again, manifest_again = shuffled_subject_split(rows, 42)
    assert parts == again and manifest == manifest_again
    assert [len(part) for part in parts.values()] == [24, 16, 8]
    ids = [set(r["subject"] for r in part) for part in parts.values()]
    assert not(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(set.union(*ids)) == 6
    assert [Counter(r["label"] for r in p) for p in parts.values()] == [dict.fromkeys(range(4), n) for n in (6, 4, 2)]
    assert manifest["shuffled_subject_order"] == np.random.default_rng(42).permutation(sorted(ids[0] | ids[1] | ids[2])).tolist()
    assert shuffled_subject_split(rows, 43)[1] != manifest


def test_zip_traversal(tmp_path):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("../escape.jpg", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        extract_dataset(path, tmp_path / "extract")
    assert not (tmp_path / "escape.jpg").exists()


def test_per_class_6_3_3_then_partition_shuffle():
    rows = [dict(subject=f"{i:03}", filename=f"{i:03}_{label}_{rep}.jpg", label=label,
                 **{"class": ("sadness", "happiness", "neutral", "fear")[label]}, sha256=f"hash{i}{label}{rep}")
            for i in range(6) for label in range(4) for rep in range(2)]
    parts, manifest = shuffled_class_split(rows, 42)
    assert shuffled_class_split(list(reversed(rows)), 42) == (parts, manifest)
    assert shuffled_class_split(rows, 43)[1] != manifest
    assert [len(parts[n]) for n in ("train", "validation", "test")] == [24, 12, 12]
    for name, count in zip(("train", "validation", "test"), (6, 3, 3)):
        assert Counter(r["class"] for r in parts[name]) == dict.fromkeys(("sadness", "happiness", "neutral", "fear"), count)
        assert [r["label"] for r in parts[name]] != sorted(r["label"] for r in parts[name])
    keys = [{r["filename"] for r in p} for p in parts.values()]
    assert len(set.union(*keys)) == 48
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    for cls, order in manifest["per_class_shuffled_order"].items():
        for name, expected in zip(("train", "validation", "test"), (order[:6], order[6:9], order[9:])):
            assert {r["filename"] for r in parts[name] if r["class"] == cls} == set(expected)
