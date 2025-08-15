import yaml, joblib
from tensorflow import keras
from src.data_loader import load_prices
from src.preprocessing import get_scaler, series_to_supervised
from src.evaluate import metrics_table

cfg = yaml.safe_load(open("config/default.yaml"))

# Load AAPL-pretrained model
model = keras.models.load_model("artifacts/models/lstm_aapl.keras")

# MSFT data + scaler refit (target domain)
df_msft = load_prices(cfg["data"]["msft_csv_path"], cfg["data"]["date_col"], cfg["data"]["target_col"])
values = df_msft["Close"].values.reshape(-1,1)
scaler = get_scaler(cfg["preprocess"]["scaler"])
scaled = scaler.fit_transform(values)

lb = cfg["preprocess"]["lookback_train"]
X, y = series_to_supervised(scaled, lb)
n = int(len(X) * cfg["split"]["train_ratio"])
Xtr, Ytr, Xte, Yte = X[:n], y[:n], X[n:], y[n:]

# Fine-tune
model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["model"]["learning_rate"]),
              loss="mse", metrics=["mae"])
model.fit(Xtr, Ytr, validation_split=cfg["split"]["val_ratio"],
          epochs=30, batch_size=cfg["train"]["batch_size"], verbose=1)

Yp = model.predict(Xte, verbose=0)
Yp_inv = scaler.inverse_transform(Yp).ravel()
Yte_inv = scaler.inverse_transform(Yte).ravel()

mt = metrics_table(Yte_inv, Yp_inv)
mt.to_csv("artifacts/reports/metrics_lstm_msft.csv", index=False)
print(mt.to_string(index=False))

model.save("artifacts/models/lstm_msft_finetuned.keras")
joblib.dump(scaler, "artifacts/models/scaler_msft.gz")
