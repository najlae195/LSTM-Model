import yaml, pandas as pd, numpy as np, joblib
from tensorflow import keras
from src.data_loader import load_prices
from src.preprocessing import series_to_supervised
from src.evaluate import metrics_table
from src.baselines import arima_forecast, svr_forecast

cfg = yaml.safe_load(open("config/default.yaml"))
df = load_prices(cfg["data"]["csv_path"], cfg["data"]["date_col"], cfg["data"]["target_col"])

values = df["Close"].values.astype(float)
n = int(len(values) * cfg["split"]["train_ratio"])
train, test = values[:n], values[n:]

# LSTM (load trained)
model = keras.models.load_model("artifacts/models/lstm_aapl.keras")
scaler = joblib.load("artifacts/models/scaler_aapl.gz")

lb = cfg["preprocess"]["lookback_train"]
scaled = scaler.transform(values.reshape(-1,1))
X_all, y_all = series_to_supervised(scaled, lb)
X_test = X_all[n-lb:]
y_test = scaler.inverse_transform(y_all[n-lb:]).ravel()
y_lstm = scaler.inverse_transform(model.predict(X_test, verbose=0)).ravel()

# ARIMA & SVR
y_arima = arima_forecast(train, test)
y_svr   = svr_forecast(train, test)

out = pd.concat([
    metrics_table(y_test[:len(y_lstm)], y_lstm).assign(Model="LSTM"),
    metrics_table(test, y_arima).assign(Model="ARIMA"),
    metrics_table(test, y_svr).assign(Model="SVR"),
], ignore_index=True)[["Model","RMSE","MAE","R2","MAPE(%)","SMAPE(%)"]]
out.to_csv("artifacts/reports/compare_models.csv", index=False)
print(out.to_string(index=False))
