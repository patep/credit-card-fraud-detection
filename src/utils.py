import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURE_COLUMNS = ["Hour_of_Day"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def prepare_features(df: pd.DataFrame, scaler: RobustScaler = None, fit: bool = False):
    """
    Converts raw 'Time' (seconds) into cyclic 'Hour_of_Day' (0-24)
    and scales 'Amount' and 'Hour_of_Day' using RobustScaler.
    """
    data = df.copy()
    data["Hour_of_Day"] = (data["Time"] / 3600.0) % 24.0
    X = data[FEATURE_COLUMNS].copy()

    scale_cols = ["Amount", "Hour_of_Day"]
    if fit:
        scaler = RobustScaler()
        X[scale_cols] = scaler.fit_transform(X[scale_cols])
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        X[scale_cols] = scaler.transform(X[scale_cols])

    return X, scaler