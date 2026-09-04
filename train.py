"""
Train a freight rate prediction model.

Split strategy
--------------
train_test.csv covers Jan-Oct 2025. The two things we ultimately need to predict
(validation.csv loads, and a December 2025 rate chart) are BOTH out-of-time relative
to training data -- i.e. this is fundamentally a forecasting problem, not an i.i.d.
regression problem. A random row-level split would overstate performance, because it
lets the model "see" nearby dates in training that a true holdout wouldn't have.

So: instead of a random split, we hold out the most recent month (Sep 15 - Oct 31)
as validation, and train on everything before that. This mimics the real prediction
task (predict rates for a future period we haven't seen) and gives an honest estimate
of how the model will do on December.
"""
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from features import build_feature_frame, fit_medians

DATA_DIR = "data"
SPLIT_DATE = "2025-09-15"  # everything from here to end-of-data is the holdout


def main():
    df = pd.read_csv(f"{DATA_DIR}/train_test.csv")
    df["date"] = pd.to_datetime(df["date"])

    train_df = df[df["date"] < SPLIT_DATE].reset_index(drop=True)
    val_df = df[df["date"] >= SPLIT_DATE].reset_index(drop=True)
    print(f"Train: {len(train_df)} rows ({train_df['date'].min().date()} - {train_df['date'].max().date()})")
    print(f"Val:   {len(val_df)} rows ({val_df['date'].min().date()} - {val_df['date'].max().date()})")

    medians = fit_medians(train_df)

    X_train = build_feature_frame(train_df, medians)
    y_train = train_df["posted_rate"]
    X_val = build_feature_frame(val_df, medians)
    y_val = val_df["posted_rate"]

    model = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=["equipment"],
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[],
    )

    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    mape = mean_absolute_percentage_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    print(f"\nHoldout (time-based, {SPLIT_DATE} onward):")
    print(f"  MAE:  ${mae:,.2f}")
    print(f"  MAPE: {mape*100:.2f}%")
    print(f"  R2:   {r2:.4f}")

    # simple distance-only baseline for comparison (sanity check that the model
    # is adding value beyond the single dominant feature)
    from sklearn.linear_model import LinearRegression
    baseline = LinearRegression().fit(train_df[["distance"]], y_train)
    base_preds = baseline.predict(val_df[["distance"]])
    base_mae = mean_absolute_error(y_val, base_preds)
    base_mape = mean_absolute_percentage_error(y_val, base_preds)
    print(f"\nBaseline (distance-only linear regression):")
    print(f"  MAE:  ${base_mae:,.2f}")
    print(f"  MAPE: {base_mape*100:.2f}%")

    # feature importance
    importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nFeature importances:")
    print(importances)

    # sanity check: how much accuracy do we lose on the holdout without market_index/quote_signal?
    reduced_cols_check = ["distance", "weight", "month_sin", "month_cos", "day_of_week",
                           "weight_missing", "weight_per_mile", "equipment"]
    reduced_train_model = LGBMRegressor(**model.get_params())
    reduced_train_model.fit(X_train[reduced_cols_check], y_train, categorical_feature=["equipment"])
    reduced_preds = reduced_train_model.predict(X_val[reduced_cols_check])
    reduced_mae = mean_absolute_error(y_val, reduced_preds)
    reduced_mape = mean_absolute_percentage_error(y_val, reduced_preds)
    print(f"\nReduced model (no market_index/quote_signal/lat-lon, mirrors December inputs):")
    print(f"  MAE:  ${reduced_mae:,.2f}")
    print(f"  MAPE: {reduced_mape*100:.2f}%")

    # ---- refit on ALL labeled data (train+val) for the final model used at inference ----
    full_medians = fit_medians(df)
    X_full = build_feature_frame(df, full_medians)
    y_full = df["posted_rate"]
    final_model = LGBMRegressor(**model.get_params())
    final_model.fit(X_full, y_full, categorical_feature=["equipment"])

    import joblib
    joblib.dump({"model": final_model, "medians": full_medians}, "model.joblib")

    # ---- reduced model: same as above but WITHOUT market_index/quote_signal/lat-lon.
    # The December chart inputs only provide pickup/delivery/distance/equipment/weight/date
    # (no live market_index or quote_signal, no coordinates) -- those are presumably real-time
    # signals unavailable for a forward-looking forecast. Train a second model restricted to
    # the columns actually available in that scenario, so December predictions aren't silently
    # relying on features that don't exist there.
    from features import build_feature_frame as _bff
    reduced_cols = ["distance", "weight", "month_sin", "month_cos", "day_of_week",
                     "weight_missing", "weight_per_mile", "equipment"]

    def build_reduced(df, medians):
        X = _bff(df, medians)
        return X[reduced_cols]

    X_full_reduced = build_reduced(df, full_medians)
    reduced_model = LGBMRegressor(**model.get_params())
    reduced_model.fit(X_full_reduced, y_full, categorical_feature=["equipment"])
    joblib.dump({"model": reduced_model, "medians": full_medians, "cols": reduced_cols},
                "model_reduced.joblib")
    print("Saved model_reduced.joblib (for December-style forecasts without market_index/quote_signal)")

    metrics = {
        "holdout_mae": mae, "holdout_mape": mape, "holdout_r2": r2,
        "baseline_mae": base_mae, "baseline_mape": base_mape,
        "reduced_model_mae": reduced_mae, "reduced_model_mape": reduced_mape,
        "split_date": SPLIT_DATE,
        "train_rows": len(train_df), "holdout_rows": len(val_df),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nSaved model.joblib and metrics.json")


if __name__ == "__main__":
    main()