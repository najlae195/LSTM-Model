# LSTM-Model

Reproducible LSTM stock-forecasting code (Apple → Microsoft transfer learning), with ARIMA & SVR
baselines, 60-day iterative forecast, SHAP analysis, and CI workflow.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Add data: data/raw/AAPL.csv and data/raw/MSFT.csv (columns: Date, Close)
python scripts/run_train.py --config config/default.yaml
python scripts/run_eval.py
python scripts/run_forecast.py
python scripts/run_shap.py
