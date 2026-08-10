import numpy as np
import joblib
from evaluate import evaluate_model, plot_confusion_matrix

def main():
    model = joblib.load("outputs/svm_model.pkl")
    scaler = joblib.load("outputs/scaler.pkl")
    X_test = np.load("outputs/X_test.npy")
    y_test = np.load("outputs/y_test.npy")

    X_test_s = scaler.transform(X_test)
    acc, y_pred = evaluate_model(model, X_test_s, y_test, kernel_name="best_model")
    plot_confusion_matrix(y_test, y_pred, "best_model",
                           save_path="outputs/confusion_matrix_test.png")

    print("\nSample predictions:")
    for i in range(min(10, len(y_pred))):
        true_label = "COVID" if y_test[i] == 1 else "Normal"
        pred_label = "COVID" if y_pred[i] == 1 else "Normal"
        print(f"  #{i}: true={true_label:6s} pred={pred_label:6s}")

if __name__ == "__main__":
    main()