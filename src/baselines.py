import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

def arima_forecast(train, test, order=(5,1,0)):
    model = sm.tsa.ARIMA(train, order=order)
    res = model.fit()
    preds = res.forecast(steps=len(test))
    return np.array(preds)

def svr_forecast(train, test):
    # simple lag-1 feature
    def fea(x): return np.asarray([x[i-1] for i in range(1, len(x))]).reshape(-1,1)
    y_train = train[1:]; X_train = fea(train)
    scaler = StandardScaler().fit(X_train)
    svr = SVR(C=100, epsilon=0.1, kernel="rbf")
    svr.fit(scaler.transform(X_train), y_train.ravel())
    preds, hist = [], list(train)
    for _ in range(len(test)):
        x = scaler.transform([[hist[-1]]])
        p = svr.predict(x)[0]
        preds.append(p)
        hist.append(p)
    return np.array(preds)
