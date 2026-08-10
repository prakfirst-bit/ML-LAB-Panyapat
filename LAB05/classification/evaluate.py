from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, kernel_name=""):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[{kernel_name}] Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "COVID"]))
    return acc, y_pred

def plot_confusion_matrix(y_test, y_pred, kernel_name, save_path=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Normal", "COVID"], yticklabels=["Normal", "COVID"])
    plt.title(f"Confusion Matrix - {kernel_name} kernel")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_sample_predictions(X_test_raw, y_test, y_pred, img_size=(64, 64), num_images=4, save_path=None):
    """
    แสดงภาพผลลัพธ์ 4 รูป พร้อมข้อความ Pred/True สีเขียวถ้าถูก สีแดงถ้าผิด
    """
    class_map = {0: "Normal", 1: "COVID"}
    num_correct = np.sum(y_test == y_pred)
    total = len(y_test)
    
    # สุ่มเลือกรูปมา 4 รูป
    indices = np.random.choice(len(y_test), num_images, replace=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))
    fig.suptitle(f"Prediction: {num_correct}/{total} correct", fontsize=14)
    
    for i, idx in enumerate(indices):
        ax = axes[i // 2, i % 2]
        
        # Reshape ข้อมูล 1D กลับมาเป็นรูปภาพ 2D เพื่อแสดงผล
        img = X_test_raw[idx].reshape(img_size)
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        true_label = class_map[y_test[idx]]
        pred_label = class_map[y_pred[idx]]
        
        # ทายถูกใช้สีเขียว ทายผิดใช้สีแดง
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color, fontsize=11)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved prediction image to {save_path}")
    plt.close()