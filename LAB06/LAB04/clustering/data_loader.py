import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data-raisin", "Raisin_Dataset.csv"
)


def load_raw_data(path=DATA_PATH):
    return pd.read_csv(path)


def prepare_data(path=DATA_PATH):
  
    df = load_raw_data(path)
    feature_cols = [c for c in df.columns if c != "Class"]

    X = df[feature_cols].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    return X_scaled, df, feature_cols, scaler


if __name__ == "__main__":
    X_scaled, df, cols, scaler = prepare_data()
    print("X_scaled shape:", X_scaled.shape)
    print("Feature columns:", cols)