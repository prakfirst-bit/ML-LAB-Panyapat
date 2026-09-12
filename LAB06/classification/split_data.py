import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset(X, y):
    # แบ่งข้อมูลเป็น Train (70%), Validation (15%), Test (15%)[cite: 1]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    # บันทึกไฟล์ไว้ในโฟลเดอร์ outputs/
    np.save("outputs/X_train.npy", X_train)
    np.save("outputs/X_val.npy", X_val)
    np.save("outputs/X_test.npy", X_test)
    np.save("outputs/y_train.npy", y_train)
    np.save("outputs/y_val.npy", y_val)
    np.save("outputs/y_test.npy", y_test)
    
    print(f"Data Split complete -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test