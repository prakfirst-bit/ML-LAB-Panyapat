import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data-raisin", "Raisin_Dataset.csv"
)


def load_raw_data(path=DATA_PATH):
    """โหลดไฟล์ csv ดิบ"""
    df = pd.read_csv(path)
    return df


def prepare_data(path=DATA_PATH, test_size=0.2, random_state=42):

    df = load_raw_data(path)

    # column ทั้งหมด: Area, MajorAxisLength, MinorAxisLength, Eccentricity,
    # ConvexArea, Extent, Perimeter, Class
    feature_cols = [c for c in df.columns if c != "Class"]

    X = df[feature_cols].astype(float)

    y_raw = df["Class"].values  # "Kecimen" / "Besni"
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train.astype(np.int32),
        y_test.astype(np.int32),
        scaler,
        le,
        feature_cols,
    )


if __name__ == "__main__":
    Xtr, Xte, ytr, yte, scaler, le, cols = prepare_data()
    print("X_train:", Xtr.shape, "X_test:", Xte.shape)
    print("Classes:", list(le.classes_))