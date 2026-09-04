"""
Generate:
  1. validation_predictions.csv  (load_id, predicted_rate) for the 12,000 validation loads
  2. december_predictions.csv    (December chart inputs + predicted_rate)
"""
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from features import build_feature_frame

DATA_DIR = "data"

bundle = joblib.load("model.joblib")
model, medians = bundle["model"], bundle["medians"]

# ---- 1. Validation set (12,000 loads) ----
val_df = pd.read_csv(f"{DATA_DIR}/validation.csv")
X_val = build_feature_frame(val_df, medians)
val_df["predicted_rate"] = model.predict(X_val).round(2)

template = pd.read_csv(f"{DATA_DIR}/validation_predictions_template.csv")
out = template[["load_id"]].merge(val_df[["load_id", "predicted_rate"]], on="load_id", how="left")
assert out["predicted_rate"].isnull().sum() == 0, "Missing predictions for some load_ids!"
assert len(out) == len(template) == 12000
out.to_csv("validation_predictions.csv", index=False)
print(f"Wrote validation_predictions.csv ({len(out)} rows)")
print(out.head())

# ---- 2. December chart ----
# Uses the REDUCED model: december_chart_inputs.csv has no market_index/quote_signal/lat-lon,
# so we can't use the full model here (see train.py for why).
reduced_bundle = joblib.load("model_reduced.joblib")
reduced_model, reduced_cols = reduced_bundle["model"], reduced_bundle["cols"]

dec_df = pd.read_csv(f"{DATA_DIR}/december_chart_inputs.csv")
X_dec_full = build_feature_frame(dec_df.drop(columns=["predicted_rate"]), medians)
X_dec = X_dec_full[reduced_cols]
dec_df["predicted_rate"] = reduced_model.predict(X_dec).round(2)
dec_df.to_csv("december_predictions.csv", index=False)
print(f"\nWrote december_predictions.csv ({len(dec_df)} rows)")
print(dec_df[["pickup", "delivery", "date", "predicted_rate"]])

# chart
dec_df["date"] = pd.to_datetime(dec_df["date"])
plt.figure(figsize=(10, 5))
for (pickup, delivery), grp in dec_df.groupby(["pickup", "delivery"]):
    grp = grp.sort_values("date")
    plt.plot(grp["date"], grp["predicted_rate"], marker="o", label=f"{pickup} -> {delivery}")
plt.xlabel("Date")
plt.ylabel("Predicted Rate ($)")
plt.title("December 2025 Freight Rate Predictions")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("december_chart.png", dpi=150)
print("\nSaved december_chart.png")