import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


def plot_predictions(X_test_raw, y_test, y_pred, classes, output_dir):
    indices = random.sample(range(len(y_test)), min(4, len(y_test)))

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Prediction Sample: X-Ray", fontsize=16)

    for i, idx in enumerate(indices):
        ax = axes[i // 2, i % 2]
        ax.imshow(X_test_raw[idx], cmap="gray")
        ax.axis("off")

        true_label = classes[y_test[idx]]
        pred_label = classes[y_pred[idx]]

        color = "green" if true_label == pred_label else "red"
        title = f"True: {true_label}\nPred: {pred_label}"
        ax.set_title(title, color=color, fontsize=12, backgroundcolor="white")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, "prediction_sample.png"))
    plt.close()
    print(f"Prediction samples saved to '{output_dir}/prediction_sample.png'")


def evaluate_model(y_test, y_pred, classes, output_dir, X_test_raw=None):
    print("-" * 40)
    print("Evaluation")
    print("-" * 40)

    # 1. Classification Report
    print("Classification Report:")
    report = classification_report(y_test, y_pred, target_names=classes)
    print(report)

    # 2. Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 3. วาด Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar(im)

    tick_marks = [0, 1]
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # 4. วาดรูปสี่ช่อง (ถ้ามีข้อมูลรูปภาพดิบ)
    if X_test_raw is not None:
        plot_predictions(X_test_raw, y_test, y_pred, classes, output_dir)

    # บันทึกเป็น text report
    with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))