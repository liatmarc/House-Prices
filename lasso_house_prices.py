# LASSO on Kaggle House Prices — Minimal Script
# Run on Kaggle or locally. See README.md for instructions.

import numpy as np
import pandas as pd
from pathlib import Path
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

DEFAULT_KAGGLE = Path('/kaggle/input/house-prices-advanced-regression-techniques')
DATA_DIR = DEFAULT_KAGGLE if DEFAULT_KAGGLE.exists() else Path('path/to/your/downloaded/folder')

def get_feature_names(preproc, numeric_cols, categorical_cols):
    try:
        return preproc.get_feature_names_out()
    except Exception:
        cat_steps = preproc.named_transformers_['cat']['ohe']
        cat_ohe_names = cat_steps.get_feature_names_out(categorical_cols)
        return np.array(list(numeric_cols) + list(cat_ohe_names))

def main():
    df = pd.read_csv(DATA_DIR / 'train.csv')
    target_col = 'SalePrice'
    y = np.log1p(df[target_col].copy())
    X = df.drop(columns=[target_col, 'Id'])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, numeric_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE
    )

    alpha_grid = np.logspace(-4, 2, 50)
    models = {
        'OLS': Pipeline([('preprocess', preprocessor), ('model', LinearRegression())]),
        'RidgeCV': Pipeline([('preprocess', preprocessor), ('model', RidgeCV(alphas=alpha_grid, cv=5))]),
        'LassoCV': Pipeline([('preprocess', preprocessor), ('model', LassoCV(alphas=alpha_grid, cv=5, max_iter=20000, random_state=RANDOM_STATE))]),
        'ElasticNetCV': Pipeline([('preprocess', preprocessor), ('model', ElasticNetCV(alphas=alpha_grid, l1_ratio=[0.2,0.5,0.8,0.95], cv=5, max_iter=20000, random_state=RANDOM_STATE))])
    }

    results = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        results[name] = rmse
        print(f"{name} RMSE (log-price): {rmse:.4f}")

    best_alpha = models['LassoCV'].named_steps['model'].alpha_
    print(f"Best alpha (LASSO CV): {best_alpha:.6f}")

    if 'ElasticNetCV' in models:
        enet_cv = models['ElasticNetCV'].named_steps['model']
        print(f"Best Elastic Net alpha: {enet_cv.alpha_:.6f}, l1_ratio: {getattr(enet_cv, 'l1_ratio_', 'NA')}")

    # Inspect selected features
    preproc_fit = models['LassoCV'].named_steps['preprocess']
    feature_names = get_feature_names(preproc_fit, numeric_cols, categorical_cols)
    coefs = models['LassoCV'].named_steps['model'].coef_
    kept = pd.DataFrame({'feature': feature_names, 'coef': coefs})
    kept = kept[kept.coef != 0].sort_values('coef', key=lambda s: s.abs(), ascending=False)
    print("\nTop LASSO features:")
    print(kept.head(15).to_string(index=False))

    # Simple sparsity & performance paths
    X_train_pre = preproc_fit.transform(X_train)
    X_test_pre = preproc_fit.transform(X_test)

    alphas_for_path = np.logspace(-4, 1, 15)
    nonzeros, rmses = [], []
    for a in alphas_for_path:
        m = Lasso(alpha=a, max_iter=20000, random_state=RANDOM_STATE).fit(X_train_pre, y_train)
        nonzeros.append(np.count_nonzero(m.coef_))
        rmses.append(mean_squared_error(y_test, m.predict(X_test_pre), squared=False))

    plt.figure()
    plt.plot(alphas_for_path, nonzeros, marker='o')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('# non-zero coefficients')
    plt.title('LASSO sparsity path')
    plt.tight_layout()
    plt.savefig('sparsity_path.png')

    plt.figure()
    plt.plot(alphas_for_path, rmses, marker='o')
    plt.xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('RMSE (log-price)')
    plt.title('Performance vs regularization')
    plt.tight_layout()
    plt.savefig('performance_path.png')

    # Random Forest comparison
    rf = Pipeline([('preprocess', preprocessor),
                   ('rf', RandomForestRegressor(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1))])
    rf.fit(X_train, y_train)
    rf_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
    print(f"\nRandomForest RMSE (log-price): {rf_rmse:.4f}")

    rf_pre = rf.named_steps['preprocess']
    rf_names = get_feature_names(rf_pre, numeric_cols, categorical_cols)
    rf_imps = rf.named_steps['rf'].feature_importances_
    rf_top = pd.DataFrame({'feature': rf_names, 'importance': rf_imps}).sort_values('importance', ascending=False).head(15)
    print("\nRandom Forest top features:")
    print(rf_top.to_string(index=False))

if __name__ == "__main__":
    main()
