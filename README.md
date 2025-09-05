# LASSO on Kaggle House Prices — Simple Workflow

This bundle contains:
- `Kaggle_LASSO_House_Prices.ipynb` — beginner‑friendly Kaggle notebook
- `lasso_house_prices.py` — GitHub‑ready script
- `README.md` — this file

## Run on Kaggle
1. Create a new Notebook.
2. Click **Add data** → search **“House Prices - Advanced Regression Techniques”** (competition) and add it.
3. Upload `Kaggle_LASSO_House_Prices.ipynb` (or copy the code from `lasso_house_prices.py` into a new Kaggle cell).
4. Run cells top to bottom.

## Run locally
1. Download the competition data and unzip so that `train.csv` is available.
2. Set `DATA_DIR` inside the notebook/script to the folder containing `train.csv` when not on Kaggle.
3. Create a virtual environment and install dependencies:
   ```bash
   pip install -U numpy pandas scikit-learn matplotlib
   ```
4. Run the script:
   ```bash
   python lasso_house_prices.py
   ```
   This will print metrics and save two figures:
   - `sparsity_path.png`
   - `performance_path.png`

## What this demonstrates
- **Regularization**: LASSO shrinks weak features to zero (built‑in feature selection).
- **Model comparison**: OLS vs. Ridge vs. LASSO (with cross‑validation).
- **“Push further”**:
  - **Sparsity & performance paths** across different penalties.
  - **Random Forest comparison** to see agreement with a nonlinear model.
  - (Notebook only) **Stability selection** via bootstrap for robust feature picks.

## Notes
- We model **log-transformed** prices (`np.log1p`) for stability.
- Use a proper **train/test split** (already included).
- You can extend this with Elastic Net, target encoding, or segmented models.
