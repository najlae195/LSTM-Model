import argparse, yaml, joblib
from tensorflow import keras
from src.utils import set_seed, ensure_dirs
from src.data_loader import load_prices
from src.train import prepare_data, callbacks
from src.model import build_lstm
from src.evaluate import metrics_table

def main(cfg_path):
    ensure_dirs(); set_seed(42)
    cfg = yaml.safe_load(open(cfg_path))
    df = load_prices(cfg["data"]["csv_path"], cfg["data"]["date_col"], cfg["data"]["target_col"])
    scaler, (Xtr, Ytr, Xte, Yte) = prepare_data(cfg, df)
    model = build_lstm(
        (cfg["preprocess"]["lookback_train"],1),
        tuple(cfg["model"]["lstm_layers"]),
        cfg["model"]["dropout"],
        cfg["model"]["learning_rate"]
    )
    model.fit(
        Xtr, Ytr,
        validation_split=cfg["split"]["val_ratio"],
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        callbacks=callbacks(cfg),
        verbose=1
    )
    Yp = model.predict(Xte, verbose=0)
    Yp_inv = scaler.inverse_transform(Yp)
    Yte_inv = scaler.inverse_transform(Yte)
    mt = metrics_table(Yte_inv, Yp_inv)
    mt.to_csv("artifacts/reports/metrics_lstm_aapl.csv", index=False)
    print(mt.to_string(index=False))
    model.save("artifacts/models/lstm_aapl.keras")
    joblib.dump(scaler, "artifacts/models/scaler_aapl.gz")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    main(args.config)
