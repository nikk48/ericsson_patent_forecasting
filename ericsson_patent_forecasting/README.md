# Ericsson Patent Forecasting (Tasks 1, 2, and 3)

This is a ready-to-run coursework project folder covering:
- Task 1: Exploratory Data Analysis (EDA)
- Task 2: Baseline forecasting of total annual patent counts
- Task 3: Segmentation-based cluster forecasting with aggregation

## Important first step
Put your CSV file inside `data/raw/` and name it:

`ericsson_patent_rich_dataset.csv`

## Folder structure
- `data/raw/` -> original dataset
- `data/processed/` -> cleaned and transformed data
- `config/` -> central project settings such as split years and tuning grids
- `src/` -> reusable Python modules
- `drivers/` -> scripts you run directly
- `outputs/figures/` -> saved charts
- `outputs/tables/` -> saved tables, metrics, and forecasts
- `tests/` -> lightweight reliability checks for key transformations

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run tasks individually
### Task 1
```bash
python3 drivers/run_eda.py
```

### Task 2
Run Task 1 first, then:
```bash
python3 drivers/run_forecasting.py
```

### Task 3
Run Task 2 first, then:
```bash
python3 drivers/run_task3.py
```

## Run everything in one command
```bash
python3 drivers/run_all_tasks.py
```

## Key Task 2 outputs
- `task2_model_metrics.csv` -> validation/test metrics
- `task2_rolling_validation_metrics.csv` -> rolling-origin validation averages used for model selection
- `task2_feature_ablation.csv` -> feature-group comparison table
- `task2_leakage_feature_check.csv` -> confirms only lag-safe forecasting inputs are used
- `recommended_model_summary.csv` -> final model-selection summary
- `linear_regression_test_predictions.csv`
- `naive_last_value_test_predictions.csv`
- `random_forest_test_predictions.csv`
- `xgboost_test_predictions.csv`
- `task2_future_forecasts.csv`
- `linear_regression_future_forecasts.csv`
- `naive_last_value_future_forecasts.csv`
- `random_forest_future_forecasts.csv`
- `xgboost_future_forecasts.csv`

## Key Task 3 outputs
- `task3_text_k_selection.csv`
- `task3_text_segmentation_sensitivity.csv`
- `task3_non_text_k_selection.csv`
- `task3_text_cluster_profiles.csv`
- `task3_non_text_cluster_profiles.csv`
- `task3_cluster_balance.csv`
- `task3_segmented_metrics_detailed.csv`
- `task3_baseline_vs_segmented_metrics.csv`
- `task3_segmented_aggregate_predictions.csv`
- `task3_segmented_future_forecasts.csv`
- `task3_future_baseline_vs_segmented.csv`
- `limitations_summary.csv`

## Notes
- Task 2 uses leakage-free lagged features for forecasting.
- Task 2 includes a naive last-value benchmark and rolling-origin validation.
- Task 3 uses two segmentation schemes:
  - Text embeddings (TF-IDF + SVD + KMeans)
  - Non-text feature clustering (KMeans)
- Task 3 now fits segmentation objects on the training window only, then freezes them for later years.
- If XGBoost cannot load (for example missing `libomp` on macOS),
  the Task 2 pipeline automatically skips XGBoost and runs with
  Linear Regression + Random Forest.
