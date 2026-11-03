from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


@pytest.mark.parametrize("page", ["Overview", "Dataset Explorer", "CNN Explorer", "GAN Explorer", "Model Comparison", "Prediction Studio", "Research Report"])
def test_all_app_pages(page, monkeypatch):
    monkeypatch.setenv("PRML_CHECKPOINT_DIR", str(APP.parent / "missing-checkpoints"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.radio[0].set_value(page).run()
    assert not app.exception
    if page == "Prediction Studio":
        assert any("unavailable" in message.value.lower() for message in app.warning)


def test_results_dashboard_without_torch(monkeypatch):
    import sys
    monkeypatch.setenv("PRML_CHECKPOINT_DIR", str(APP.parent / "missing-checkpoints"))
    monkeypatch.setitem(sys.modules, "torch", None)
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    app.radio[0].set_value("Prediction Studio").run()
    assert not app.exception
    assert any("unavailable" in message.value.lower() for message in app.warning)
