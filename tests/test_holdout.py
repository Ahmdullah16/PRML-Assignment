import zipfile
from collections import Counter
import numpy as np
import pytest
from prml.holdout import shuffled_subject_split, extract_dataset


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
