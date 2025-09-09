# Summary

## Metrics (RMSE on log-price)

- **OLS**: 0.1273
- **RidgeCV**: 0.1353
- **LassoCV**: 0.1232
- **RandomForest**: 0.1397
- **LASSO best alpha**: 0.000543
- **Elastic Net best**: RMSE=0.1256, l1_ratio=0.8, alpha=0.000719686

## Plots

- LASSO sparsity path: `results/sparsity_path.png`
- LASSO performance path: `results/performance_path.png`
- Residuals: `results/residuals.png`
- Elastic Net RMSE by l1_ratio: `results/enet_rmse_by_l1.png`
- Elastic Net sparsity by l1_ratio: `results/enet_sparsity_by_l1.png`
- Elastic Net top features: `results/enet_top_features.png`

## Top LASSO features (first 15)

| feature                    |       coef |
|:---------------------------|-----------:|
| cat__RoofMatl_ClyTile      | -1.36511   |
| cat__Condition2_PosN       | -0.362949  |
| cat__Neighborhood_StoneBr  |  0.112481  |
| num__GrLivArea             |  0.111807  |
| cat__Neighborhood_Crawfor  |  0.109277  |
| cat__Exterior1st_BrkFace   |  0.0933346 |
| cat__MSZoning_C (all)      | -0.0779789 |
| num__OverallQual           |  0.074812  |
| cat__Functional_Typ        |  0.0621939 |
| cat__Neighborhood_NridgHt  |  0.0612623 |
| cat__Functional_Maj2       | -0.0561465 |
| cat__SaleCondition_Abnorml | -0.0549444 |
| cat__MSZoning_RM           | -0.0526438 |
| cat__BsmtQual_Ex           |  0.0518102 |
| num__YearBuilt             |  0.0488391 |

## Top Elastic Net features (first 15)

| feature                    |       coef |
|:---------------------------|-----------:|
| cat__RoofMatl_ClyTile      | -0.988744  |
| cat__Condition2_PosN       | -0.277767  |
| cat__Neighborhood_StoneBr  |  0.113621  |
| cat__Neighborhood_Crawfor  |  0.109149  |
| num__GrLivArea             |  0.10878   |
| cat__Exterior1st_BrkFace   |  0.0933188 |
| num__OverallQual           |  0.0768413 |
| cat__Neighborhood_NridgHt  |  0.0660238 |
| cat__MSZoning_C (all)      | -0.0633525 |
| cat__Functional_Typ        |  0.0610675 |
| cat__LandContour_Bnk       | -0.0545499 |
| cat__BsmtQual_Ex           |  0.054331  |
| cat__MSZoning_RM           | -0.0531465 |
| cat__SaleCondition_Abnorml | -0.0520499 |
| cat__Condition1_Norm       |  0.0486954 |