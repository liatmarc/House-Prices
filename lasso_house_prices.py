# lasso_house_prices.py
# LASSO on Kaggle House Prices — complete script (sklearn 1.7+ friendly)
# lasso_house_prices.py
# LASSO on Kaggle House Prices — sklearn 1.7+ friendly, auto-writes results/summary.md

import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Lasso, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --- Results dir (next to this script) ---
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("RUNNING SCRIPT:", os.path.abspath(__file__))
print("Saving plots to:", RESULTS_DIR)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def resolve_data_dir(cli_dir: str | None) -> Path:
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
    raise FileNotFoundError("Could not find train.csv. Pass --data-dir or place train.csv in Kaggle input / Downloads / CWD.")

def get_feature_names(preproc: ColumnTransformer, numeric_cols, categorical_cols):
    try:
        return preproc.get_feature_names_out()
    except Exception:
        ohe = preproc.named_transformers_["cat"]["ohe"]
        cat_names = ohe.get_feature_names_out(categorical_cols)
        return np.array(list(numeric_cols) + list(cat_names))

def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=120)
    plt.close(fig)

def main():
    # --- CLI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default=None, help="Folder containing train.csv")
    args = ap.parse_args()

    # --- DATA ---
    data_dir = resolve_data_dir(args.data_dir)
    print("SCRIPT PATH   =", os.path.abspath(__file__))
    print("Using DATA_DIR =", data_dir.resolve())

    df = pd.read_csv(data_dir / "train.csv")
    target_col = "SalePrice"
    y = np.log1p(df[target_col].copy())
    X = df.drop(columns=[target_col, "Id"])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Numeric features: {len(numeric_cols)}, Categorical features: {len(categorical_cols)}")

    # --- PREPROCESS ---
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])  # sklearn>=1.2
    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols),
                                      ("cat", cat_pipe, categorical_cols)], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE)

    # --- MODELS: OLS / Ridge / LASSO ---
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

    best_alpha = models["LassoCV"].named_steps["model"].alpha_
    print(f"Best alpha (LASSO CV): {best_alpha:.6f}")

    # --- LASSO kept ---
    preproc_fit = models["LassoCV"].named_steps["preprocess"]
    feature_names = get_feature_names(preproc_fit, numeric_cols, categorical_cols)
    lasso_coefs = models["LassoCV"].named_steps["model"].coef_
    kept = pd.DataFrame({"feature": feature_names, "coef": lasso_coefs})
    kept = kept[kept.coef != 0].sort_values("coef", key=lambda s: s.abs(), ascending=False)
    print("\nTop LASSO features:")
    print(kept.head(15).to_string(index=False))

    # --- LASSO paths (sparsity & performance) ---
    X_train_pre = preproc_fit.transform(X_train)
    X_test_pre  = preproc_fit.transform(X_test)

    alphas_for_path = np.logspace(-4, 1, 15)
    nonzeros, rmses = [], []
    for a in alphas_for_path:
        m = Lasso(alpha=a, max_iter=20000, random_state=RANDOM_STATE).fit(X_train_pre, y_train)
        nonzeros.append(int(np.count_nonzero(m.coef_)))
        rmses.append(rmse(y_test, m.predict(X_test_pre)))

    fig1, ax1 = plt.subplots()
    ax1.plot(alphas_for_path, nonzeros, marker="o")
    ax1.set_xscale("log"); ax1.set_xlabel("alpha"); ax1.set_ylabel("# non-zero coefficients"); ax1.set_title("LASSO sparsity path")
    savefig(fig1, "sparsity_path.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(alphas_for_path, rmses, marker="o")
    ax2.set_xscale("log"); ax2.set_xlabel("alpha"); ax2.set_ylabel("RMSE (log-price)"); ax2.set_title("Performance vs regularization")
    savefig(fig2, "performance_path.png")

    # --- Random Forest baseline ---
    rf = Pipeline([("preprocess", preprocessor),
                   ("rf", RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])
    rf.fit(X_train, y_train)
    rf_rmse = rmse(y_test, rf.predict(X_test))
    results["RandomForest"] = rf_rmse
    print(f"\nRandomForest RMSE (log-price): {rf_rmse:.4f}")

    rf_pre = rf.named_steps["preprocess"]
    rf_names = rf_pre.get_feature_names_out()
    rf_imps = rf.named_steps["rf"].feature_importances_
    rf_top = pd.DataFrame({"feature": rf_names, "importance": rf_imps}).sort_values("importance", ascending=False).head(15)
    print("\nRandom Forest top features:")
    print(rf_top.to_string(index=False))

    # --- Diagnostics (residuals) ---
    lasso_preds = models["LassoCV"].predict(X_test)
    resid = y_test - lasso_preds
    fig3, ax3 = plt.subplots()
    ax3.scatter(lasso_preds, resid, alpha=0.5)
    ax3.axhline(0, linestyle="--")
    ax3.set_xlabel("Predicted log-price"); ax3.set_ylabel("Residual (actual - pred)"); ax3.set_title("Residuals — should look like noise")
    savefig(fig3, "residuals.png")

    # --- Elastic Net sweep (plots + best) ---
    l1_grid = [0.2, 0.5, 0.8, 0.95]
    enet_rmse_by_l1, enet_nz_by_l1 = [], []
    best_enet = {"l1_ratio": None, "alpha": None, "rmse": np.inf, "coef": None}

    for r in l1_grid:
        enet_cv = ElasticNetCV(alphas=alpha_grid, l1_ratio=[r], cv=5, max_iter=20000, random_state=RANDOM_STATE)
        enet_cv.fit(X_train_pre, y_train)   # fit on preprocessed arrays
        preds = enet_cv.predict(X_test_pre)
        err = rmse(y_test, preds)
        nz = int(np.count_nonzero(enet_cv.coef_))
        enet_rmse_by_l1.append(err)
        enet_nz_by_l1.append(nz)
        if err < best_enet["rmse"]:
            best_enet.update({"l1_ratio": r, "alpha": float(enet_cv.alpha_), "rmse": err, "coef": enet_cv.coef_.copy()})

    # ENet plots
    fig4, ax4 = plt.subplots()
    ax4.plot(l1_grid, enet_rmse_by_l1, marker="o")
    ax4.set_xlabel("l1_ratio"); ax4.set_ylabel("RMSE (log-price)"); ax4.set_title("Elastic Net: RMSE by l1_ratio")
    savefig(fig4, "enet_rmse_by_l1.png")

    fig5, ax5 = plt.subplots()
    ax5.plot(l1_grid, enet_nz_by_l1, marker="o")
    ax5.set_xlabel("l1_ratio"); ax5.set_ylabel("# non-zero coefficients"); ax5.set_title("Elastic Net: sparsity by l1_ratio")
    savefig(fig5, "enet_sparsity_by_l1.png")

    # ENet top features (best l1_ratio)
    enet_keep = pd.DataFrame({"feature": feature_names, "coef": best_enet["coef"]})
    enet_keep = enet_keep[enet_keep.coef != 0].sort_values("coef", key=lambda s: s.abs(), ascending=False).head(15)
    fig6, ax6 = plt.subplots(figsize=(8,6))
    ax6.barh(enet_keep["feature"][::-1], enet_keep["coef"][::-1])
    ax6.set_title(f"Elastic Net top 15 (l1_ratio={best_enet['l1_ratio']}, alpha={best_enet['alpha']:.4g})")
    savefig(fig6, "enet_top_features.png")

    # --- Write summary.md ---
    summary_lines = []
    summary_lines.append("# Summary\n")
    summary_lines.append("## Metrics (RMSE on log-price)\n")
    for k in ["OLS", "RidgeCV", "LassoCV", "RandomForest"]:
        if k in results:
            summary_lines.append(f"- **{k}**: {results[k]:.4f}")
    summary_lines.append(f"- **LASSO best alpha**: {best_alpha:.6f}")
    summary_lines.append(f"- **Elastic Net best**: RMSE={best_enet['rmse']:.4f}, l1_ratio={best_enet['l1_ratio']}, alpha={best_enet['alpha']:.6g}\n")

    summary_lines.append("## Plots\n")
    summary_lines.append("- LASSO sparsity path: `results/sparsity_path.png`")
    summary_lines.append("- LASSO performance path: `results/performance_path.png`")
    summary_lines.append("- Residuals: `results/residuals.png`")
    summary_lines.append("- Elastic Net RMSE by l1_ratio: `results/enet_rmse_by_l1.png`")
    summary_lines.append("- Elastic Net sparsity by l1_ratio: `results/enet_sparsity_by_l1.png`")
    summary_lines.append("- Elastic Net top features: `results/enet_top_features.png`\n")

    summary_lines.append("## Top LASSO features (first 15)\n")
    summary_lines.append(kept.head(15).to_markdown(index=False))
    summary_lines.append("\n## Top Elastic Net features (first 15)\n")
    summary_lines.append(enet_keep.to_markdown(index=False))

    (RESULTS_DIR / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    print("\nWrote results/summary.md")

if __name__ == "__main__":
    main()
