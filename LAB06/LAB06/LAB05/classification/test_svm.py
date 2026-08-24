import os
import joblib
import numpy as np

# 1. แก้การ import ให้ดึงจาก evaluate.py ในโฟลเดอร์เดียวกันโดยตรง
from evaluate import evaluate_model


def main():
    print("-" * 40)
    print("Testing Saved SVM Model")
    print("-" * 40)

    # 2. โหลด Model, Scaler และข้อมูล Test set
    model = joblib.load("outputs/svm_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")

    # กำหนดชื่อ Class ให้ตรงกับข้อมูล
    classes = ["COVID", "Normal"]

    # 3. Scale ข้อมูล และสั่งให้ Model ทำนายผล (y_pred) ออกมาก่อน
    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)

    # 4. โหลดรูปภาพดิบ (X_test_raw) เพื่อเอาไปวาดรูป 4 ช่อง
    raw_path = "outputs/X_test_raw.npy"
    X_test_raw = np.load(raw_path) if os.path.exists(raw_path) else None

    # 5. ส่งค่า y_test, y_pred และ X_test_raw ให้ evaluate_model ประเมินผลและเซฟรูป
    evaluate_model(y_test, y_pred, classes, "outputs", X_test_raw=X_test_raw)

    # 6. ปริ้นท์ตัวอย่างการทำนาย 10 รายการแรกออกมาดูใน Terminal
    print("\nSample predictions:")
    for i in range(min(10, len(y_pred))):
        true_label = classes[y_test[i]]
        pred_label = classes[y_pred[i]]
        print(f"  #{i}: true={true_label:8s} pred={pred_label:8s}")


if __name__ == "__main__":
    main()