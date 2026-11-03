import copy
import io
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader
from prml import CLASSES
from prml.models import CNN, Generator, Discriminator
from prml.data import discover, split_subjects, Faces, image_tensor, audit
from prml.metrics import classification_metrics, aggregate
from prml.inference import load_checkpoint, predict
from prml.train import train_model, discriminator_loss, generator_loss, seed_everything

torch.set_num_threads(2)


@pytest.fixture
def dataset(tmp_path):
    rng = np.random.default_rng(123)
    for name in CLASSES:
        folder = tmp_path / name
        folder.mkdir()
        for sid in ("001", "002", "003"):
            Image.fromarray(rng.integers(0, 256, (20, 24, 3), dtype=np.uint8)).save(folder / f'{sid}_x_x_{name[0]}_a.png')
    return tmp_path


def test_data_and_loso(dataset):
    rows = discover(dataset)
    assert len(rows) == 12
    assert audit(rows)["counts"] == dict.fromkeys(CLASSES, 3)
    tested = []
    for fold in range(3):
        train, val, test = split_subjects(rows, fold)
        ids = [{r["subject"] for r in p} for p in (train, val, test)]
        assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        assert all({r["label"] for r in p} == set(range(4)) for p in (train, val, test))
        tested.extend(r["filename"] for r in test)
    assert len(tested) == len(set(tested)) == len(rows)
    x = image_tensor(rows[0]["path"])
    assert x.shape == (3, 64, 64) and x.min() >= -1 and x.max() <= 1
    ds = Faces(rows)
    assert torch.equal(ds[0][0], ds[0][0])


def test_corrupt_ambiguous_duplicate(dataset):
    f = next((dataset / "sadness").glob("*.png"))
    original = f.read_bytes()
    f.write_bytes(b"broken")
    with pytest.raises(ValueError, match="Corrupt"):
        discover(dataset)
    f.write_bytes(original)
    other = dataset / "sadness" / "004_x_x_s_a.png"
    other.write_bytes(original)
    with pytest.raises(ValueError, match="duplicate"):
        discover(dataset)
    other.unlink()
    f.rename(f.with_name("unknown.png"))
    with pytest.raises(ValueError, match="Ambiguous"):
        discover(dataset)


def test_shapes_and_probabilities():
    x = torch.randn(2, 3, 64, 64)
    logits = CNN()(x)
    assert logits.shape == (2, 4)
    assert torch.allclose(logits.softmax(1).sum(1), torch.ones(2))
    generated = Generator()(torch.randn(2, 64), torch.tensor([0, 3]))
    assert generated.shape == x.shape and generated.min() >= -1 and generated.max() <= 1
    source, classes = Discriminator()(generated)
    assert source.shape == (2,) and classes.shape == (2, 4)


def test_adversarial_gradients():
    generator, discriminator = Generator(), Discriminator()
    y = torch.tensor([0, 3])
    fake = generator(torch.randn(2, 64), y)
    real = discriminator(torch.randn(2, 3, 64, 64))
    fake_out = discriminator(fake.detach())
    fake_out[1].retain_grad()
    discriminator_loss(real, fake_out, y, y).backward()
    assert fake_out[1].grad.abs().sum() > 0  # fake classification term reaches D
    assert all(p.grad is None for p in generator.parameters())
    discriminator.zero_grad(set_to_none=True)
    for p in discriminator.parameters():
        p.requires_grad_(False)
    generator_loss(discriminator(fake), y).backward()
    assert generator.embed.weight.grad.abs().sum() > 0
    assert generator.net[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in discriminator.parameters())


def test_metric_hand_calculation():
    m = classification_metrics([0, 1, 2, 3], [0, 0, 0, 0])
    assert m["accuracy"] == m["balanced_accuracy"] == .25
    assert m["macro_precision"] == .0625
    assert m["macro_f1"] == .1
    m = classification_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert m["macro_f1"] == 1
    with pytest.raises(ValueError):
        classification_metrics([0], [4])


@pytest.mark.parametrize("kind", ["cnn", "gan"])
def test_checkpoint_and_deterministic_inference(kind, tmp_path):
    seed_everything(42)
    net = CNN() if kind == "cnn" else Discriminator()
    path = tmp_path / "model.pt"
    saved = dict(model=net.state_dict(), type=kind, size=64, classes=CLASSES)
    torch.save(saved, path)
    loaded, meta = load_checkpoint(path)
    photo = io.BytesIO()
    Image.new("RGB", (24, 24), "white").save(photo, format="PNG")
    photo.seek(0)
    one = predict(loaded, kind, photo)
    photo.seek(0)
    assert one == predict(loaded, kind, photo)
    assert abs(sum(one.values()) - 1) < 1e-6
    torch.save(saved | {"classes": tuple(reversed(CLASSES))}, path)
    with pytest.raises(ValueError, match="label"):
        load_checkpoint(path)


@pytest.mark.parametrize("kind", ["cnn", "gan"])
def test_one_epoch_training(kind, dataset):
    rows = discover(dataset)
    train, val, _ = split_subjects(rows, 0)
    tl, vl = DataLoader(Faces(train), batch_size=4), DataLoader(Faces(val), batch_size=4)
    model, gen, history, selected, seconds = train_model(kind, tl, vl, 1, "cpu", .0002)
    assert len(history) == selected == 1 and seconds > 0
    assert np.isfinite(history[0]["val_loss"])
    if gen:
        assert not gen.training


def test_historical_arithmetic():
    from scripts.verify_results import verify
    result = verify(Path(__file__).resolve().parents[1] / "results/historical")
    assert len(result["checks"]) == 12
    assert result["metrics"]["cnn"]["pooled"]["accuracy"] == pytest.approx(1 / 3)
