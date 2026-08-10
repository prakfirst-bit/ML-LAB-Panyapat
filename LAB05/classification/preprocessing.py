import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def flatten_images(X):
    return X.reshape(X.shape[0], -1).astype(np.float32)

def standardize_features(X_train, X_test, save_path=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    if save_path:
        joblib.dump(scaler, save_path)
    return X_train_scaled, X_test_scaled, scaler