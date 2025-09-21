import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from .utils import savefig_no_style
except Exception:
    from utils import savefig_no_style

def eda_summary(df: pd.DataFrame, figs_dir: Path):
    # Class balance
    if "approved" in df.columns:
        vc = df["approved"].value_counts(dropna=False)
        fig, ax = plt.subplots()
        ax.bar(vc.index.astype(str), vc.values)
        ax.set_title("Class Balance")
        ax.set_xlabel("approved")
        ax.set_ylabel("count")
        savefig_no_style(fig, figs_dir / "class_balance.png")
        plt.close(fig)

    # Histograms (numeric)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != "approved"]
    for c in num_cols:
        fig, ax = plt.subplots()
        ax.hist(df[c].dropna(), bins=30)
        ax.set_title(f"Histogram — {c}")
        savefig_no_style(fig, figs_dir / f"hist_{c}.png")
        plt.close(fig)

    # Boxplots (selected)
    for c in num_cols[:6]:
        fig, ax = plt.subplots()
        ax.boxplot(df[c].dropna(), vert=True)
        ax.set_title(f"Boxplot — {c}")
        savefig_no_style(fig, figs_dir / f"box_{c}.png")
        plt.close(fig)

    # Correlation heatmap (numeric only)
    if num_cols:
        corr = df[num_cols].corr(numeric_only=True)
        fig, ax = plt.subplots()
        im = ax.imshow(corr.values)
        ax.set_title("Correlation (numeric features)")
        ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90)
        ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols)
        # Avoid specifying colors explicitly per instructions.
        savefig_no_style(fig, figs_dir / "corr_numeric.png")
        plt.close(fig)
