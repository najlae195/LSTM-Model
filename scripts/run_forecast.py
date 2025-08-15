import yaml, joblib, pandas as pd, matplotlib.pyplot as plt
from tensorflow import keras
from src.data_loader import load_prices
from src.forecast import iterative_forecast

cfg = yaml.safe_load(open("config/default.yaml"))
df = load_prices(cfg["data"]["csv_path"], cfg["data"]["date_col"], cfg["data"]["target_col"])

model = keras.models.load_model("artifacts/models/lstm_aapl.keras")
scaler = joblib.load("artifacts/models/scaler_aapl.gz")

lb = cfg["preprocess"]["lookback_forecast"]
scaled = scaler.transform(df["Close"].values.reshape(-1,1))
last_window = scaled[-lb:].reshape(lb,1)

future = iterative_forecast(model, scaler, last_window, cfg["forecast"]["horizon_days"])
last_date = df["Date"].iloc[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=len(future))

pd.DataFrame({"Date": future_dates, "Forecast": future}).to_csv("artifacts/reports/forecast_60d.csv", index=False)

plt.figure()
plt.plot(df["Date"].iloc[-200:], df["Close"].iloc[-200:], label="Actual")
plt.plot(future_dates, future, linestyle="--", label="Forecast (60d)")
plt.xlabel("Date"); plt.ylabel("Price"); plt.title("LSTM 60-Day Iterative Forecast (AAPL)")
plt.legend(); plt.tight_layout()
plt.savefig("artifacts/figures/forecast_60d.png", dpi=200)
