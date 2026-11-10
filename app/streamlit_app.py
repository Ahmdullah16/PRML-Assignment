"""Inference-only research dashboard; never trains or downloads private data."""
import io
import json
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st
from prml import CLASSES

PAGES = ["Overview", "Dataset Explorer", "CNN Explorer", "GAN Explorer", "Model Comparison", "Prediction Studio", "Research Report"]


@st.cache_data
def read_json(path, modified):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_json(path):
    return read_json(str(path), path.stat().st_mtime_ns) if path.exists() else None


@st.cache_resource
def checkpoint(path, modified):
    import torch
    torch.set_num_threads(2)
    from prml.inference import load_checkpoint
    return load_checkpoint(path)


def image_if_exists(path, caption):
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")


def main():
    st.set_page_config(page_title="PRML | Expression Lab", page_icon="◈", layout="wide")
    st.markdown("""<style>
        .block-container{max-width:1250px;padding-top:2.6rem}
        [data-testid="stMetric"]{background:white;padding:1.2rem;border-radius:12px;border:1px solid #dbe5ed}
        h1{letter-spacing:-.035em} h2{letter-spacing:-.02em}
        [data-testid="stSidebar"]{border-right:1px solid #dbe5ed}
    </style>""", unsafe_allow_html=True)
    st.sidebar.markdown("## ◈ Expression Lab\nIIT Guwahati · PRML")
    page = st.sidebar.radio("Research workspace", PAGES)
    available = {"Recovered baseline": ROOT / "results/historical_verified.json"}
    if (ROOT / "results/corrected/aggregate.json").exists():
        available["Corrected experiment"] = ROOT / "results/corrected/aggregate.json"
    if (ROOT / "results/kaggle_seed42/summary.json").exists():
        available["Kaggle four-class retraining"] = ROOT / "results/kaggle_seed42/summary.json"
    selection = st.sidebar.selectbox("Evidence source", list(available), index=len(available) - 1)
    if selection == "Kaggle four-class retraining" or (page == "Prediction Studio" and "Kaggle four-class retraining" in available):
        from app.kaggle_dashboard import render
        render(page, ROOT, checkpoint)
        return
    result = get_json(available[selection])
    dataset = get_json(ROOT / "results/dataset.json")
    figure_root = ROOT / ("results/figures/corrected" if selection == "Corrected experiment" else "results/figures/historical")
    st.sidebar.caption("Ahmdullah · M.Tech\nSignal Processing and Machine Learning\nElectronics and Electrical Engineering")
    st.caption("FOUR-CLASS FACIAL EXPRESSION RECOGNITION")
    st.title(page)
    st.caption("Prediction models: seed-42 per-class 6/3/3 image split · 64 × 64 RGB" if page == "Prediction Studio" else
               f"LOSO evidence: {selection} · subject-independent evaluation · 64 × 64 RGB")
    if page == "Overview":
        deployed = get_json(ROOT / "deployment/manifest.json")
        if deployed and deployed.get("experiment") == "kaggle-four-class-seed42-v1":
            st.info("Public prediction now uses the separately retrained Kaggle four-class models. The historical evidence selected below remains a separate experiment; choose Kaggle four-class retraining for current model results.")
        elif deployed:
            st.info("Public prediction uses the corrected per-class 6/3/3 split: 24 training / 12 validation / 12 test images, seed 42. Test image accuracy: CNN 41.67%, AC-GAN 25%. Subjects overlap across these sets. The six-fold LOSO results below are separate subject-independent evidence.")
        st.markdown("## Two learning objectives. One controlled comparison.")
        st.write("Compare a supervised CNN with the auxiliary expression head of an AC-GAN discriminator. "
                 "Six people rotate through independent test folds, with a separate validation subject in every fold.")
        cols = st.columns(4)
        for col, label, value in zip(cols, ["Original images", "Expressions", "Subjects", "Uniform-guess accuracy"], ["48", "4", "6", "25%"]):
            col.metric(label, value)
        if result:
            a, b = st.columns(2)
            for col, kind in zip((a, b), ("cnn", "gan")):
                col.metric(f"{kind.upper()} · pooled accuracy", f'{result["metrics"][kind]["pooled"]["accuracy"]:.2%}')
            image_if_exists(figure_root / "comparison.png", selection + "; pooled out-of-fold predictions")
        st.info("This system estimates a facial-expression category from image appearance. It does not measure a person's actual emotional or mental state.")
        st.write("The dataset is too small to establish broad generalization. Read the report for the differences between recovered and corrected training.")
    elif page == "Dataset Explorer":
        if dataset:
            st.dataframe(pd.DataFrame({"Expression": CLASSES, "Images": [dataset["counts"][c] for c in CLASSES]}), hide_index=True)
            image_if_exists(figure_root / "dataset.png", "Audited dataset · original image statistics")
            st.write("EXIF orientation → RGB → bilinear resize to 64 × 64 → normalize to [−1, 1]. Horizontal flips occur during training only, for both methods.")
            st.write("Each fold: 32 training images / 4 subjects; 8 validation images / 1 subject; 8 test images / 1 subject.")
            if (ROOT / "results/class_6_3_3_seed42/summary.json").exists():
                st.write("The corrected seed-42 image split shuffles each class separately and takes 6 training / 3 validation / 3 test images, then shuffles each combined set. Totals: 24 / 12 / 12. Image membership is disjoint; subject identities overlap. See the new split reports separately from LOSO.")
            st.caption(f'{len(dataset["near_duplicates"])} perceptual-match candidates; exact duplicates rejected. Cross-subject candidates reviewed as distinct photographs.')
        st.info("Original photographs remain private. Public deployment includes statistics and diagrams. Authorized local viewing is enabled only by server configuration.")
        private = os.getenv("PRML_PRIVATE_DATA")
        if private and os.getenv("PRML_ALLOW_FACE_DISPLAY") == "1":
            from prml.data import discover
            rows = discover(private)
            expression = st.selectbox("Expression examples", CLASSES)
            cols = st.columns(4)
            for col, row in zip(cols, [r for r in rows if r["class"] == expression][:4]):
                col.image(row["path"], caption=row["filename"])
    elif page in ("CNN Explorer", "GAN Explorer"):
        kind = "cnn" if page == "CNN Explorer" else "gan"
        image_if_exists(figure_root / ("cnn_architecture.png" if kind == "cnn" else "acgan_architecture.png"), "Architecture reused for checkpoint compatibility")
        if kind == "cnn":
            st.write("Three convolution blocks learn local patterns. Batch normalization conditions activations; ReLU introduces nonlinearity; max pooling reduces spatial size. A 64-unit dense head with dropout outputs four logits.")
            st.latex(r"P(y=k\mid x)=\frac{e^{z_k}}{\sum_{j=1}^{4}e^{z_j}},\quad \mathcal L=-\log P(y\mid x)")
        else:
            st.write("Noise and a class embedding enter the generator. The discriminator returns a source logit and four expression logits. Only its expression head performs recognition.")
            st.latex(r"L_D=BCE(D_s(x),1)+BCE(D_s(G(z,c)),0)+CE(D_c(x),c)+CE(D_c(G(z,c)),c)")
            st.latex(r"L_G=BCE(D_s(G(z,c)),1)+CE(D_c(G(z,c)),c)")
            st.caption("Corrected non-saturating source loss; original discriminator omitted the final fake-class CE term. Generator output quality and diversity are not established.")
            st.info("Synthetic sample grids are retained privately to avoid publishing potentially memorized training faces.")
        fold = st.selectbox("Training fold", list(range(6)))
        image_if_exists(figure_root / f"fold_{fold}_{kind}_history.png", f"{selection} · fold {fold}; losses have model-specific units")
    elif page == "Model Comparison":
        holdout = get_json(ROOT / "results/class_6_3_3_seed42/summary.json")
        if holdout:
            with st.expander("Corrected seed-42 per-class 6 / 3 / 3 experiment", expanded=True):
                st.write("Each class: 6 training / 3 validation / 3 test images. Totals: 24/12/12 (50%/25%/25%). Every class and each combined set shuffle separately. Subjects can occur in multiple sets; these are image-level metrics.")
                st.dataframe(pd.DataFrame([dict(Model=kind.upper(), Partition=part, Images=m["n"],
                    Accuracy=m["accuracy"], **{"Macro F1": m["macro_f1"], "Balanced accuracy": m["balanced_accuracy"]})
                    for kind, model in holdout["models"].items() for part, m in model["partitions"].items()]), hide_index=True)
                image_if_exists(ROOT / "results/class_6_3_3_seed42/figures/split_accuracy.png", "Per-class 6/3/3 image split, seed 42; separate from LOSO")
        if not result:
            st.info("Verified results unavailable.")
        else:
            metrics = result["metrics"]
            st.dataframe(pd.DataFrame({kind.upper(): metrics[kind]["pooled"] for kind in ("cnn", "gan")}).loc[
                ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1"]])
            st.caption("Metrics above are pooled across held-out images. Mean fold macro F1 is a different statistic.")
            st.json({k: v["fold_summary"] for k, v in metrics.items()}, expanded=False)
            for file in ("fold_accuracy.png", "class_metrics.png"):
                image_if_exists(figure_root / file, selection)
            cols = st.columns(2)
            for col, kind in zip(cols, ("cnn", "gan")):
                with col:
                    image_if_exists(figure_root / f"{kind}_confusion.png", "Rows: true; columns: predicted; counts are images")
            st.write("Accuracy near 25% warrants caution. A constant class predictor also scores 25% here; this does not demonstrate random predictions or useful recognition.")
            st.download_button("Download verified experiment JSON", json.dumps(result, indent=2), file_name="prml-results.json", mime="application/json")
    elif page == "Prediction Studio":
        st.write("Upload one cropped face image. Scores are uncalibrated softmax outputs. Uploads are processed in memory and are not written to disk by this app.")
        st.info("Predictions describe apparent expression categories, not actual emotions or mental state. These classroom models have limited measured accuracy.")
        choice = st.selectbox("Model", ["Both models", "CNN", "AC-GAN classifier"])
        override = os.getenv("PRML_CHECKPOINT_DIR")
        checkpoint_root = Path(override) if override else ROOT / "deployment"
        kinds = ("cnn", "gan") if choice == "Both models" else (("cnn",) if choice == "CNN" else ("gan",))
        if override and not (checkpoint_root / "cnn.pt").exists():
            fold = st.selectbox("Local LOSO demonstration fold (not selected by test score)", list(range(6)))
            checkpoint_root = checkpoint_root / f"fold_{fold}"
        else:
            st.caption("Public models: seed-42 per-class 6/3/3 image split; validation-selected checkpoints. CNN test accuracy 41.67%; AC-GAN 25%. Twelve test images, three per class. Subjects overlap with training; these are not unseen-person accuracy estimates.")
        paths = {kind: checkpoint_root / f"{kind}.pt" for kind in kinds}
        ready = all(path.is_file() for path in paths.values())
        if not ready:
            st.warning("Prediction unavailable: trained checkpoints have not been provisioned on this server.")
        uploaded = st.file_uploader("Image (PNG or JPEG)", type=["png", "jpg", "jpeg"], disabled=not ready)
        if uploaded is not None and ready:
            try:
                from prml.inference import predict
                raw = uploaded.getvalue()
                with Image.open(io.BytesIO(raw)) as im:
                    im.verify()
                st.image(raw, width=220, caption="Uploaded image")
                cols = st.columns(len(kinds))
                for col, kind in zip(cols, kinds):
                    with col:
                        net, meta = checkpoint(str(paths[kind]), paths[kind].stat().st_mtime_ns)
                        scores = predict(net, kind, io.BytesIO(raw))
                        st.subheader(kind.upper())
                        st.metric("Predicted expression", max(scores, key=scores.get).title())
                        st.bar_chart(pd.DataFrame({"Softmax score": scores}))
                        st.caption(f'Checkpoint: {meta.get("experiment", "recovered baseline")}; selected epoch {meta.get("best_epoch", "not recorded")}')
            except (ValueError, OSError, RuntimeError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                st.error(f"Unable to process image or checkpoint: {exc}")
    elif page == "Research Report":
        report_kind = st.selectbox("Report", ["Per-class 6/3/3 experiment", "Training", "Validation", "Test", "Original LOSO study", "Previous seed-42 subject holdout"])
        split_reports = {"Per-class 6/3/3 experiment": "EXPERIMENT", "Training": "TRAIN", "Validation": "VALIDATION", "Test": "TEST"}
        report = ROOT / "reports/ASSIGNMENT_REPORT.md"
        if report_kind in split_reports:
            report = ROOT / "reports/class_6_3_3_seed42" / f"{split_reports[report_kind]}_REPORT.md"
        elif report_kind == "Previous seed-42 subject holdout":
            report = ROOT / "reports/seed42/EXPERIMENT_REPORT.md"
        if report.exists():
            content = report.read_text(encoding="utf-8")
            st.download_button("Download research report", content, file_name=report.name, mime="text/markdown")
            st.markdown(content)


if __name__ == "__main__":
    main()
