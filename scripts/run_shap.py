import yaml, joblib, numpy as np, matplotlib.pyplot as plt, shap
from tensorflow import keras
from src.data_loader import load_prices
from src.preprocessing import series_to_supervised
from src.shap_analysis import compute_shap_deep

cfg = yaml.safe_load(open("config/default.yaml"))
df = load_prices(cfg["data"]["csv_path"], cfg["data"]["date_col"], cfg["data"]["target_col"])

model = keras.models.load_model("artifacts/models/lstm_aapl.keras")
scaler = joblib.load("artifacts/models/scaler_aapl.gz")

lb = cfg["preprocess"]["lookback_train"]
scaled = scaler.transform(df["Close"].values.reshape(-1,1))
X, _ = series_to_supervised(scaled, lb)

bg = X[:128]
sample = X[-1024:] if len(X) > 1024 else X

vals = compute_shap_deep(model, bg, sample)

# Save neutral-caption plots (OK for B/W)
shap.summary_plot(vals, sample, show=False)
plt.title("SHAP Summary (Apple)"); plt.tight_layout()
plt.savefig("artifacts/figures/shap_summary_aapl.png", dpi=200); plt.close()

shap.summary_plot(vals, sample, plot_type="bar", show=False)
plt.title("SHAP Mean |Value| per Lag (Apple)"); plt.tight_layout()
plt.savefig("artifacts/figures/shap_bar_aapl.png", dpi=200); plt.close()

shap.dependence_plot(lb-1, vals, sample, show=False)
plt.title("SHAP Dependence: Lag_1 (Apple)"); plt.tight_layout()
plt.savefig("artifacts/figures/shap_dependence_lag1_aapl.png", dpi=200); plt.close()
