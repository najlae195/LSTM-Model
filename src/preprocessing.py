import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_scaler(name: str):
    return MinMaxScaler() if name.lower()=="minmax" else StandardScaler()

def series_to_supervised(series: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i-lookback:i])
        y.append(series[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y
