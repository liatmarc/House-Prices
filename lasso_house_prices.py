# lasso_house_prices.py
# LASSO on Kaggle House Prices — complete script (sklearn 1.7+ friendly)

# lasso_house_prices.py
# LASSO on Kaggle House Prices — sklearn 1.7+ friendly, results saved to /results

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Results dir (next to this script)
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("RUNNING SCRIPT:", os.path.abspath(__file__))
print("Saving plots to:", RESULTS_DIR)


def rmse(y_true, y_pred):
    """Version-proof RMSE"""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def resolve_data_dir(cli_dir: str | None) -> Path:
    """Pick where train.csv lives: CLI -> Kaggle path -> Downloads -> CWD."""
    DEFAULT_KAGGLE = Path("/kaggle/input/house-prices-advanced-regression-techniques")
    if cli_dir:
        p = Path(cli_dir)
        if (p / "train.csv").exists():
            return p
    if (DEFAULT_KAGGLE / "train.csv").exists():
        return DEFAULT_KAGGLE
    downloads = Path.home() / "Downloads"
    if (downloads / "train.csv").exists():
        return downloads
    cwd = Path.cwd()
    if (cwd / "train.csv").exists():
        return cwd
    raise FileNotFoundError(
        "Could not find train.csv. Pass --data-dir, or place train.csv in "
        "Kaggle input, your Downloads, or the current folder."
    )

def get_feature_names(preproc: ColumnTransformer, numeric_cols, categorical_cols):
    """Get feature names after ColumnTransformer."""
    try:
        return preproc.get_feature_names_out()
    except Exception:
        ohe = preproc.named_transformers_["cat"]["ohe"]
        cat_names = ohe.get_feature_names_out(categorical_cols)
        return np.array(list(numeric_cols) + list(cat_names))

def main():
    # --- CLI args ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Folder containing train.csv")
    args = parser.parse_args()

    # --- DATA DIR ---
    data_dir = resolve_data_dir(args.data_dir)
    print("SCRIPT PATH   =", os.path.abspath(__file__))
    print("Using DATA_DIR =", data_dir.resolve())

    # --- LOAD ---
    df = pd.read_csv(data_dir / "train.csv")
    target_col = "SalePrice"
    y = np.log1p(df[target_col].copy())  # log(1+price) helps stability
    X = df.drop(columns=[target_col, "Id"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

    # --- PREPROCESS ---
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # sklearn 1.7+
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    # --- BASELINES ---
    alpha_grid = np.logspace(-4, 2, 50)
    models = {
        "OLS": Pipeline([("preprocess", preprocessor), ("model", LinearRegression())]),
        "RidgeCV": Pipeline([("preprocess", preprocessor), ("model", RidgeCV(alphas=alpha_grid, cv=5))]),
        "LassoCV": Pipeline([("preprocess", preprocessor), ("model", LassoCV(alphas=alpha_grid, cv=5, max_iter=20000, random_state=RANDOM_STATE))]),
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        err = rmse(y_test, preds)
        results[name] = err
        print(f"{name} RMSE (log-price): {err:.4f}")

    # best alpha for LASSO
    best_alpha = models["LassoCV"].named_steps["model"].alpha_
    print(f"Best alpha (LASSO CV): {best_alpha:.6f}")

    # --- WHAT LASSO KEPT ---
    preproc_fit = models["LassoCV"].named_steps["preprocess"]
    feature_names = get_feature_names(preproc_fit, numeric_cols, categorical_cols)
    lasso_coefs = models["LassoCV"].named_steps["model"].coef_
    kept = pd.DataFrame({"feature": feature_names, "coef": lasso_coefs})
    kept = kept[kept.coef != 0].sort_values("coef", key=lambda s: s.abs(), ascending=False)
    print("\nTop LASSO features:")
    print(kept.head(15).to_string(index=False))

    # --- SPARSITY & PERFORMANCE PATHS ---
    X_train_pre = preproc_fit.transform(X_train)
    X_test_pre = preproc_fit.transform(X_test)

    alphas_for_path = np.logspace(-4, 1, 15)
    nonzeros, rmses = [], []
    for a in alphas_for_path:
        m = Lasso(alpha=a, max_iter=20000, random_state=RANDOM_STATE).fit(X_train_pre, y_train)
        nonzeros.append(int(np.count_nonzero(m.coef_)))
        rmses.append(rmse(y_test, m.predict(X_test_pre)))

    plt.figure()
    plt.plot(alphas_for_path, nonzeros, marker="o")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("# non-zero coefficients")
    plt.title("LASSO sparsity path")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sparsity_path.png")

    plt.figure()
    plt.plot(alphas_for_path, rmses, marker="o")
    plt.xscale("log")
    plt.xlabel("alpha")
    plt.ylabel("RMSE (log-price)")
    plt.title("Performance vs regularization")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "performance_path.png")

    # --- RANDOM FOREST COMPARISON ---
    rf = Pipeline([
        ("preprocess", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf.fit(X_train, y_train)
    rf_rmse = rmse(y_test, rf.predict(X_test))
    print(f"\nRandomForest RMSE (log-price): {rf_rmse:.4f}")

    rf_pre = rf.named_steps["preprocess"]
    rf_names = rf_pre.get_feature_names_out()
    rf_imps = rf.named_steps["rf"].feature_importances_
    rf_top = pd.DataFrame({"feature": rf_names, "importance": rf_imps}).sort_values("importance", ascending=False).head(15)
    print("\nRandom Forest top features:")
    print(rf_top.to_string(index=False))

    # --- DIAGNOSTICS: Residuals & buckets ---
    lasso_preds = models["LassoCV"].predict(X_test)
    resid = y_test - lasso_preds

    plt.figure()
    plt.scatter(lasso_preds, resid, alpha=0.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted log-price")
    plt.ylabel("Residual (actual - pred)")
    plt.title("Residuals — should look like noise")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "residuals.png")

    q = pd.qcut(y_test, q=3, labels=["Low", "Mid", "High"])
    rmse_by_bucket = pd.DataFrame({
        "bucket": ["Low", "Mid", "High"],
        "rmse": [
            rmse(y_test[q == "Low"],  lasso_preds[q == "Low"]),
            rmse(y_test[q == "Mid"],  lasso_preds[q == "Mid"]),
            rmse(y_test[q == "High"], lasso_preds[q == "High"]),
        ]
    })
    print("\nRMSE by price bucket (log-scale):")
    print(rmse_by_bucket.to_string(index=False))
    
def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=120)
    plt.close(fig)

    # Sparsity
    fig1, ax1 = plt.subplots()
    ax1.plot(alphas_for_path, nonzeros, marker="o")
    ax1.set_xscale("log"); ax1.set_xlabel("alpha"); ax1.set_ylabel("# non-zero")
    ax1.set_title("LASSO sparsity path")
    savefig(fig1, "sparsity_path.png")

    # Performance
    fig2, ax2 = plt.subplots()
    ax2.plot(alphas_for_path, rmses, marker="o")
    ax2.set_xscale("log"); ax2.set_xlabel("alpha"); ax2.set_ylabel("RMSE (log-price)")
    ax2.set_title("Performance vs regularization")
    savefig(fig2, "performance_path.png")

    # Residuals
    fig3, ax3 = plt.subplots()
    ax3.scatter(lasso_preds, resid, alpha=0.5)
    ax3.axhline(0, linestyle="--")
    ax3.set_xlabel("Predicted log-price"); ax3.set_ylabel("Residual")
    ax3.set_title("Residuals — should look like noise")
    savefig(fig3, "residuals.png")


if __name__ == "__main__":
    main()
