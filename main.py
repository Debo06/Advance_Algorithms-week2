import argparse
from pathlib import Path
import sys
import pandas as pd

# Support running as a script
_THIS = Path(__file__).resolve()
sys.path.append(str(_THIS.parent))

from utils import ensure_dirs  # noqa
import data_gen  # noqa
import eda  # noqa
import preprocess  # noqa
import model as model_mod  # noqa
import eval as eval_mod  # noqa

def load_data(mode: str, input_path: str, rows: int):
    if mode == "synthetic":
        return data_gen.generate_synthetic_credit(n_rows=rows, seed=42)
    elif mode == "csv":
        p = Path(input_path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {p}")
        return pd.read_csv(p)
    else:
        raise ValueError("mode must be 'synthetic' or 'csv'")

def main(args):
    figs, artifacts, data_dir = ensure_dirs()
    df = load_data(args.mode, args.input_path, args.rows)

    # Ensure target name
    target = args.target or "approved"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in data.")

    # EDA
    eda.eda_summary(df, figs)

    # Split
    X_train, X_test, y_train, y_test = preprocess.split_data(df, target, test_size=args.test_size, random_state=args.random_state)

    # Preprocessor + Model
    pre, num_cols, cat_cols = preprocess.build_preprocessor(X_train)
    class_weight = "balanced" if args.class_weight_balanced else None
    pipe = model_mod.build_model(pre, class_weight=class_weight, C=args.C, solver=args.solver, max_iter=args.max_iter)

    # Fit
    pipe.fit(X_train, y_train)

    # Feature names & coefficients/odds
    # Fit a small transform to get names
    pre.fit(X_train)
    feat_names = model_mod.get_feature_names(pre, X_train)
    coef_df = model_mod.extract_coeffs(pipe, feat_names)
    coef_df.to_csv(artifacts / "coefficients_odds.csv", index=False)

    # Evaluate
    proba = eval_mod.evaluate(pipe, X_test, y_test, artifacts)
    eval_mod.threshold_sweep(y_test.values, proba, artifacts)

    # Save predictions
    preds = (proba >= 0.5).astype(int)
    out = pd.DataFrame({"y_true": y_test.values, "proba": proba, "pred": preds})
    out.to_csv(artifacts / "preds_test.csv", index=False)

    print("Done. See figures/ and artifacts/.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 2 — Credit Approval with Logistic Regression")
    parser.add_argument("--mode", choices=["synthetic","csv"], default="synthetic")
    parser.add_argument("--input_path", type=str, default="data/credit.csv")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--class_weight_balanced", action="store_true")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--solver", type=str, default="lbfgs")
    parser.add_argument("--max_iter", type=int, default=200)
    args = parser.parse_args()
    main(args)
