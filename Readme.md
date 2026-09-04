# Freight Rate Prediction — Spotter ML Assessment

Predicts freight `posted_rate` from load characteristics (route, distance, equipment,
weight, date, and where available, live market signals).


## Setup

```bash
pip install -r requirements.txt
```

Place the provided data files in `data/`:
```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

## Run

```bash
python train.py     # trains full model (model.joblib) + reduced model (model_reduced.joblib), prints holdout metrics
python predict.py    # writes validation_predictions.csv, december_predictions.csv, december_chart.png
```

## Approach summary

- **Validation split:** time-based (train on Jan 1 – Sep 14 2025, holdout on Sep 15 – Oct 31 2025)
  rather than random, since both prediction targets (validation.csv, December chart) are
  out-of-time forecasts, not i.i.d. samples. See the report for full reasoning.
- **Two models:** `validation.csv` includes `market_index`/`quote_signal`/coordinates;
  `december_chart_inputs.csv` doesn't. A "full" model is used for validation predictions;
  a "reduced" model trained without those columns is used for the December chart, so December
  predictions don't rely on features that don't exist in that scenario.
- **Model:** LightGBM gradient-boosted trees.
- **Full model holdout:** MAE ≈ $160, MAPE ≈ 7.1%, R² ≈ 0.83 (vs. $193 MAE for a distance-only baseline).
- **Reduced model holdout:** MAE ≈ $169, MAPE ≈ 7.3%.

See `Spotter_ML_Assessment_Report.docx` for full detail, data quality findings, and the
December chart.

## Files

| File | Purpose |
|---|---|
| `features.py` | Shared feature engineering (train/inference use identical transforms) |
| `train.py` | Trains both models, prints holdout metrics, saves `model.joblib` / `model_reduced.joblib` |
| `predict.py` | Generates `validation_predictions.csv`, `december_predictions.csv`, `december_chart.png` |
| `build_report.js` | Generates the DOCX report |samaple line