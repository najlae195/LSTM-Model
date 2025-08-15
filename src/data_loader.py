import pandas as pd

def load_prices(csv_path: str, date_col: str="Date", target_col: str="Close"):
    df = pd.read_csv(csv_path)
    if date_col not in df or target_col not in df:
        raise ValueError(f"CSV must contain columns '{date_col}' and '{target_col}'.")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df = df.dropna(subset=[target_col]).copy()
    return df[[date_col, target_col]].rename(columns={date_col:"Date", target_col:"Close"})
