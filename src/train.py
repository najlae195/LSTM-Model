import numpy as np
from tensorflow import keras
from .preprocessing import get_scaler, series_to_supervised

def prepare_data(cfg, df):
    values = df["Close"].values.reshape(-1,1)
    scaler = get_scaler(cfg["preprocess"]["scaler"])
    scaled = scaler.fit_transform(values)
    X, y = series_to_supervised(scaled, cfg["preprocess"]["lookback_train"])
    n = int(len(X) * cfg["split"]["train_ratio"])
    return scaler, (X[:n], y[:n], X[n:], y[n:])

def callbacks(cfg):
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg["train"]["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=cfg["train"]["reduce_lr_patience"],
            factor=cfg["train"]["reduce_lr_factor"],
            verbose=1
        )
    ]
