import matplotlib.pyplot as plt


plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False  
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def plot_k_curve(k_values, accuracies, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "01_k_curve.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(k_values, accuracies, marker="o", color="#2E86AB")
    best_idx = int(np.argmax(accuracies))
    plt.scatter(
        [k_values[best_idx]], [accuracies[best_idx]],
        color="red", zorder=5,
        label=f"Best k = {k_values[best_idx]} (acc={accuracies[best_idx]:.3f})"
    )
    plt.xlabel("k (จำนวนเพื่อนบ้าน)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy เทียบกับค่า k ของ KNN (TensorFlow implementation)")
    plt.xticks(k_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึกกราฟ k-curve ที่: {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "02_confusion_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึก confusion matrix ที่: {save_path}")


def print_classification_report(y_true, y_pred, class_names):
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    ))