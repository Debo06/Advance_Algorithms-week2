import numpy as np
import pandas as pd

def generate_synthetic_credit(n_rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Numeric drivers
    income = rng.normal(65000, 18000, size=n_rows).clip(15000, None)
    credit_score = rng.normal(690, 70, size=n_rows).clip(300, 850)
    debt_to_income = np.abs(rng.normal(0.32, 0.12, size=n_rows)).clip(0.01, 0.95)
    age = rng.normal(38, 10, size=n_rows).clip(18, 85)
    existing_loans = rng.poisson(0.8, size=n_rows)
    # Categoricals
    employment = rng.choice(["full_time","part_time","self_employed","unemployed"], p=[0.58,0.18,0.18,0.06], size=n_rows)
    home_ownership = rng.choice(["own","mortgage","rent","other"], p=[0.25,0.35,0.36,0.04], size=n_rows)
    purpose = rng.choice(["auto","home_improvement","debt_consolidation","education","medical","business","vacation"], size=n_rows)
    region = rng.choice(["NA","EU","APAC","LATAM"], p=[0.5,0.2,0.2,0.1], size=n_rows)

    # Latent linear score to create a realistic relationship
    lin = (
        0.00003 * income +
        0.01 * (credit_score - 650) -
        1.8 * debt_to_income -
        0.12 * existing_loans +
        0.02 * (age - 35)
    )
    # Categorical contributions
    cat_bonus = (
        (employment == "full_time") * 0.35 +
        (home_ownership == "own") * 0.25 +
        (purpose == "debt_consolidation") * (-0.15) +
        (purpose == "home_improvement") * 0.1 +
        (region == "NA") * 0.05
    )
    score = lin + cat_bonus + rng.normal(0, 0.4, size=n_rows)
    prob = 1 / (1 + np.exp(-score))
    approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "income": income.round(2),
        "credit_score": credit_score.round(0),
        "debt_to_income": debt_to_income.round(3),
        "age": age.round(0),
        "existing_loans": existing_loans,
        "employment": employment,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "region": region,
        "approved": approved,
    })

    # Introduce missingness
    mask = rng.random(n_rows) < 0.03
    df.loc[mask, "income"] = np.nan
    mask = rng.random(n_rows) < 0.03
    df.loc[mask, "employment"] = None
    return df
