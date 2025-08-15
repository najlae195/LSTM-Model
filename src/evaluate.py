import numpy as np
import pandas as pd

def rmse(y, yh): return float(np.sqrt(np.mean((y - yh)**2)))
def mae(y, yh):  return float(np.mean(np.abs(y - yh)))
def mape(y, yh): return float(np.mean(np.abs((y - yh)/np.maximum(np.abs(y),1e-9))) * 100)
def r2(y, yh):
    ssr = np.sum((y - yh)**2)
    sst = np.sum((y - np.mean(y))**2)
    return float(1 - ssr/np.maximum(sst, 1e-9))
def smape(y, yh):
    denom = (np.abs(y) + np.abs(yh)) / 2.0
    return float(np.mean(np.abs(y - yh) / np.maximum(denom, 1e-9)) * 100)

def metrics_table(y_true, y_pred):
    return pd.DataFrame([{
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE(%)": mape(y_true, y_pred),
        "SMAPE(%)": smape(y_true, y_pred)
    }])
