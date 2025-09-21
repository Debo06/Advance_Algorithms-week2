import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .utils import savefig_no_style
except Exception:
    from utils import savefig_no_style

def evaluate(pipe, X_test, y_test, outdir: Path):
    proba = pipe.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
    }
    pd.DataFrame(cm, index=["actual_0","actual_1"], columns=["pred_0","pred_1"]).to_csv(outdir / "confusion_matrix.csv")
    pd.DataFrame([metrics]).to_csv(outdir / "metrics.csv", index=False)
    # ROC curve
    fpr, tpr, thr = roc_curve(y_test, proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC Curve")
    savefig_no_style(fig, outdir / "roc_curve.png")
    return proba

def threshold_sweep(y_true, proba, outdir: Path):
    rows = []
    for t in np.linspace(0.1, 0.9, 17):
        p = (proba >= t).astype(int)
        rows.append({
            "threshold": round(float(t),3),
            "accuracy": float((p == y_true).mean()),
            "precision": float( ( (p==1) & (y_true==1) ).sum() / max((p==1).sum(), 1) ),
            "recall": float( ( (p==1) & (y_true==1) ).sum() / max((y_true==1).sum(), 1) ),
            "pred_positives": int((p==1).sum())
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "threshold_sweep.csv", index=False)
    # Simple plot of precision/recall vs threshold
    fig, ax = plt.subplots()
    ax.plot(df["threshold"], df["precision"], label="precision")
    ax.plot(df["threshold"], df["recall"], label="recall")
    ax.set_xlabel("threshold"); ax.set_ylabel("score"); ax.set_title("Threshold Sweep")
    ax.legend()
    savefig_no_style(fig, outdir / "threshold_sweep.png")
