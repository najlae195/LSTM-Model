import numpy as np

def iterative_forecast(model, scaler, last_window_scaled, horizon):
    window = last_window_scaled.copy().reshape(-1,1)
    preds_scaled = []
    for _ in range(horizon):
        x = window.reshape(1, window.shape[0], 1)
        yhat = model.predict(x, verbose=0)[0,0]
        preds_scaled.append(yhat)
        window = np.vstack([window[1:], [yhat]])
    preds_scaled = np.array(preds_scaled).reshape(-1,1)
    return scaler.inverse_transform(preds_scaled).flatten()
