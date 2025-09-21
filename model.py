import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_model(preprocessor, class_weight=None, C=1.0, solver="lbfgs", max_iter=200):
    clf = LogisticRegression(C=C, solver=solver, max_iter=max_iter, class_weight=class_weight)
    pipe = Pipeline([("pre", preprocessor), ("lr", clf)])
    return pipe

def extract_coeffs(pipe, feature_names):
    lr = pipe.named_steps["lr"]
    coef = lr.coef_.ravel()
    odds = np.exp(coef)
    df = pd.DataFrame({"feature": feature_names, "coef": coef, "odds_ratio": odds})
    return df.sort_values("odds_ratio", ascending=False)

def get_feature_names(preprocessor, X_sample: pd.DataFrame):
    # Derive transformed feature names (num + onehot cat)
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(cat_encoder.get_feature_names_out(cat_features))
    return list(num_features) + cat_names
