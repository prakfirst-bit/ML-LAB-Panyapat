import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loader import load_dataset
from preprocessing import flatten_images, standardize_features
from split_data import split_dataset
from svm_model import train_svm
from evaluate import evaluate_model

# Path ไปยังโฟลเดอร์ dataset และ outputs
DATASET_DIR = "../Covid19-dataset"
OUT_DIR = "outputs"
IMG_SIZE = (64, 64)

os.makedirs(OUT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, kernel_name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Normal", "COVID"], 
                yticklabels=["Normal", "COVID"])
    plt.title(f"Confusion Matrix ({kernel_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_sample_predictions(X_raw, y_true, y_pred, img_size, num_images, save_path):
    class_map = {0: "Normal", 1: "COVID"}
    indices = np.random.choice(len(X_raw), min(num_images, len(X_raw)), replace=False)
    
    plt.figure(figsize=(10, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_images, i + 1)
        img = X_raw[idx].reshape(img_size)
        plt.imshow(img, cmap='gray')
        
        actual = class_map.get(y_true[idx], str(y_true[idx]))
        pred = class_map.get(y_pred[idx], str(y_pred[idx]))
        
        color = 'green' if actual == pred else 'red'
        plt.title(f"Act: {actual}\nPred: {pred}", color=color, fontsize=10)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_model(model, save_path):
    joblib.dump(model, save_path)

def main():
    print("Loading dataset...")
    X, y = load_dataset(DATASET_DIR, img_size=IMG_SIZE)
    print(f"Loaded {X.shape[0]} images, size {IMG_SIZE}")

    np.save(os.path.join(OUT_DIR, "features.npy"), X)
    np.save(os.path.join(OUT_DIR, "labels.npy"), y)
    with open(os.path.join(OUT_DIR, "classes.json"), "w") as f:
        json.dump({"0": "Normal", "1": "COVID"}, f)

    X_flat = flatten_images(X)
    X_train, X_test, y_train, y_test = split_dataset(X_flat, y)

    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "y_test.npy"), y_test)

    X_train_s, X_test_s, scaler = standardize_features(
        X_train, X_test, save_path=os.path.join(OUT_DIR, "scaler.pkl")
    )

    # ---------------------------------------------------------
    # เพิ่ม Gaussian Noise ใส่ข้อมูลทดสอบ
    # ---------------------------------------------------------
    noise = np.random.normal(loc=0.0, scale=1.5, size=X_test_s.shape)
    X_test_noisy = X_test_s + noise
    # ---------------------------------------------------------

    kernels = ["linear", "poly", "rbf"]
    results = {}
    best_model, best_acc, best_kernel = None, -1, None
    best_y_pred = None

    for k in kernels:
        print(f"\nTraining SVM (kernel={k}) ...")
        model = train_svm(X_train_s, y_train, kernel=k)
        
        # ประเมินผลโดยส่งข้อมูลที่ใส่ Noise เข้าไป
        acc, y_pred = evaluate_model(model, X_test_noisy, y_test, kernel_name=k)
        
        plot_confusion_matrix(y_test, y_pred, k,
                              save_path=os.path.join(OUT_DIR, f"confusion_matrix_{k}.png"))
        
        # เซฟภาพ Visual Predictions แยกตามแต่ละ kernel
        plot_sample_predictions(X_test, y_test, y_pred, img_size=IMG_SIZE, num_images=4,
                                save_path=os.path.join(OUT_DIR, f"visual_predictions_{k}.png"))
        
        results[k] = acc
        if acc > best_acc:
            best_model, best_acc, best_kernel = model, acc, k
            best_y_pred = y_pred

    print("\n=== Summary ===")
    for k, acc in results.items():
        print(f"{k:10s}: {acc:.4f}")
    print(f"\nBest kernel: {best_kernel} (accuracy = {best_acc:.4f})")

    save_model(best_model, os.path.join(OUT_DIR, "svm_model.pkl"))
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()