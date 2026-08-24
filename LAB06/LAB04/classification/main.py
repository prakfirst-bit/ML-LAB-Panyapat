import os
import numpy as np
import pandas as pd

from data_loader import prepare_data
from knn_np import KNNClassifierNP          # <-- เปลี่ยนจาก knn_tf import KNNClassifierTF
from evaluate import (
    compute_accuracy,
    plot_k_curve,
    plot_confusion_matrix,
    print_classification_report,
    OUTPUT_DIR,
)

K_VALUES = [1, 3, 5, 7, 9, 11]


def main():
    print("=" * 60)
    print("LAB 4: KNN Classification บน Raisin Dataset (NumPy)")
    print("=" * 60)

    X_train, X_test, y_train, y_test, scaler, le, feature_cols = prepare_data()
    print(f"จำนวนข้อมูล train: {X_train.shape[0]}, test: {X_test.shape[0]}")
    print(f"จำนวน class: {len(le.classes_)} -> {list(le.classes_)}\n")

    accuracies = []
    for k in K_VALUES:
        model = KNNClassifierNP(k=k).fit(X_train, y_train)   # <-- เปลี่ยนชื่อคลาส
        y_pred = model.predict(X_test)
        acc = compute_accuracy(y_test, y_pred)
        accuracies.append(acc)
        print(f"k = {k:2d} -> Accuracy = {acc:.4f}")

    best_idx = int(np.argmax(accuracies))
    best_k = K_VALUES[best_idx]
    print(f"\nค่า k ที่ดีที่สุด: k = {best_k} (Accuracy = {accuracies[best_idx]:.4f})")

    best_model = KNNClassifierNP(k=best_k).fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)

    print("\nClassification Report (k=%d):" % best_k)
    print_classification_report(y_test, y_pred_best, le.classes_)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_k_curve(K_VALUES, accuracies)
    plot_confusion_matrix(y_test, y_pred_best, le.classes_)

    pred_df = pd.DataFrame({
        "actual": le.inverse_transform(y_test),
        "predicted": le.inverse_transform(y_pred_best),
        "correct": y_test == y_pred_best,
    })
    pred_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"บันทึกผลการทำนายที่: {pred_path}")

    print("\nเสร็จสิ้น! ผลลัพธ์ทั้งหมดอยู่ในโฟลเดอร์ classification/outputs/")


if __name__ == "__main__":
    main()