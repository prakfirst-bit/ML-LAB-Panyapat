import json
import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import CLASSES


def split_and_save(x_train, y_train, x_test, y_test, output_dir="outputs", val_size=0.1, seed=42):
    X_train, X_val, y_train_split, y_val = train_test_split(
        x_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/X_val.npy", X_val)
    np.save(f"{output_dir}/X_test.npy", x_test)
    np.save(f"{output_dir}/y_train.npy", y_train_split)
    np.save(f"{output_dir}/y_val.npy", y_val)
    np.save(f"{output_dir}/y_test.npy", y_test)

    with open(f"{output_dir}/classes.json", "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, ensure_ascii=False, indent=2)

    print(f"แบ่งข้อมูล -> train={len(X_train)} val={len(X_val)} test={len(x_test)}")
    return (X_train, y_train_split), (X_val, y_val), (x_test, y_test)