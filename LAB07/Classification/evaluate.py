import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def evaluate_model(model, X_test, y_test, class_names, output_dir="outputs"):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {acc:.4f}\n")

    report = classification_report(y_test, y_pred, target_names=class_names)
    print("Classification Report:\n", report)
    with open(f"{output_dir}/classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"บันทึก confusion matrix ไว้ที่ {output_dir}/confusion_matrix.png")

    return acc, cm


def plot_training_history(history_path="outputs/history.json", output_dir="outputs"):
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["accuracy"], label="train")
    axes[0].plot(history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["loss"], label="train")
    axes[1].plot(history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png", dpi=150)
    plt.close()
    print(f"บันทึกกราฟ training history ไว้ที่ {output_dir}/training_history.png")