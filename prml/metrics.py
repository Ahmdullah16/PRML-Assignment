"""Metrics from fixed four-class confusion matrices; undefined precision is zero."""
import numpy as np
from . import CLASSES


def classification_metrics(actual, predicted):
    a, p = np.asarray(actual), np.asarray(predicted)
    if a.ndim != 1 or p.shape != a.shape or not len(a):
        raise ValueError("Predictions and labels must be nonempty paired vectors")
    if not np.isin(a, range(4)).all() or not np.isin(p, range(4)).all():
        raise ValueError("Labels must be integers 0..3")
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (a.astype(int), p.astype(int)), 1)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(4), where=cm.sum(0) != 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(4), where=cm.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(4), where=(precision + recall) != 0)
    return dict(accuracy=float(tp.sum() / cm.sum()), balanced_accuracy=float(recall.mean()),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()), macro_f1=float(f1.mean()),
                precision_by_class=dict(zip(CLASSES, precision.tolist())),
                recall_by_class=dict(zip(CLASSES, recall.tolist())), f1_by_class=dict(zip(CLASSES, f1.tolist())),
                confusion_matrix=cm.tolist(), labels=list(CLASSES), n=len(a))


def aggregate(folds, bootstrap_seed=2026):
    result = {}
    rng = np.random.default_rng(bootstrap_seed)
    for name in ("cnn", "gan"):
        predictions = [r for f in folds for r in f["models"][name]["test_predictions"]]
        keys = [(r["subject"], r["filename"]) for r in predictions]
        if len(keys) != len(set(keys)):
            raise ValueError("Repeated out-of-fold image")
        actual = [CLASSES.index(r["true"]) for r in predictions]
        pred = [CLASSES.index(r["predicted"]) for r in predictions]
        metrics = classification_metrics(actual, pred)
        fold_mean = {k: dict(mean=float(np.mean([f["models"][name][k] for f in folds])),
                             std_population=float(np.std([f["models"][name][k] for f in folds])))
                     for k in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1")}
        # Subject clusters, not images. Descriptive resampling conditional on these fixed OOF fits.
        accuracies = np.array([f["models"][name]["accuracy"] for f in folds])
        boot = rng.choice(accuracies, size=(10000, len(accuracies)), replace=True).mean(1)
        result[name] = dict(pooled=metrics, fold_summary=fold_mean,
                            subject_accuracy_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                            bootstrap_note="Exploratory percentile interval over held-out subject accuracies; "
                            "six clusters, overlapping training sets; not a population guarantee",
                            predictions=predictions)
    return result
