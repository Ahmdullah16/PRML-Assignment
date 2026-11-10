import hashlib
import io
import json
from pathlib import Path
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest
from prml import CLASSES
from prml.inference import load_checkpoint, predict

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["cnn", "gan"])
def test_deployed_classifier(kind):
    manifest = json.loads((ROOT / "deployment/manifest.json").read_text())
    path = ROOT / "deployment" / manifest["files"][kind]["file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["files"][kind]["sha256"]
    model, metadata = load_checkpoint(path)
    assert metadata["seed"] == 42 and metadata["type"] == kind
    assert metadata['experiment'] == manifest['experiment']
    if metadata['experiment'] == 'kaggle-four-class-seed42-v1':
        summary = json.loads((ROOT / 'results/kaggle_seed42/summary.json').read_text())
        assert metadata['split_manifest_sha256'] == summary['split_manifest_sha256']
        assert metadata['best_epoch'] == summary['models'][kind]['best_epoch']
        assert metadata['experiment'] == summary['experiment']
    else:
        summary = json.loads((ROOT / "results/class_6_3_3_seed42/summary.json").read_text())
        assert metadata["experiment"] == "seed42-class-6-3-3-v1"
        assert metadata["subjects"] == {n: p["subjects"] for n, p in summary["split_manifest"]["partitions"].items()}
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buffer, format="PNG")
    buffer.seek(0)
    scores = predict(model, kind, buffer)
    assert tuple(scores) == CLASSES and abs(sum(scores.values()) - 1) < 1e-6
    buffer.seek(0)
    assert scores == predict(model, kind, buffer)
    assert not (ROOT / "deployment/generator.pt").exists()


def test_public_upload_enabled(monkeypatch):
    monkeypatch.delenv("PRML_CHECKPOINT_DIR", raising=False)
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.radio[0].set_value("Prediction Studio").run()
    assert not app.exception
    assert not app.get("file_uploader")[0].proto.disabled
    assert not app.warning


def test_holdout_reports_and_executed_notebook():
    for name in ("TRAIN", "VALIDATION", "TEST", "EXPERIMENT"):
        assert (ROOT / f"reports/class_6_3_3_seed42/{name}_REPORT.md").exists()
    nb = json.loads((ROOT / "notebooks/PRML_Class_6_3_3_Seed42.ipynb").read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(cells) >= 10
    assert all(c["execution_count"] is not None for c in cells)
    assert not [out for c in cells for out in c.get("outputs", []) if out["output_type"] == "error"]
