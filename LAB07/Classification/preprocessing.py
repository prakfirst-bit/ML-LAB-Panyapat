import numpy as np


def preprocess_images(x):
    x = x.astype("float32") / 255.0  # standardize
    x = np.expand_dims(x, axis=-1)   # (N, 28, 28) -> (N, 28, 28, 1)
    return x