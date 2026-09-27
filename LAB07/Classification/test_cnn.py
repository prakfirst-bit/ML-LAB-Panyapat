import json
import numpy as np
import matplotlib.pyplot as plt

from cnn_model import load_trained_model


def test_with_random_images(output_dir="outputs", n=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    X_test = np.load(f"{output_dir}/X_test.npy")
    y_test = np.load(f"{output_dir}/y_test.npy")
    with open(f"{output_dir}/classes.json", encoding="utf-8") as f:
        class_names = json.load(f)

    model = load_trained_model(f"{output_dir}/cnn_model.keras")

    idx = np.random.choice(len(X_test), n, replace=False)
    preds = model.predict(X_test[idx], verbose=0)
    pred_labels = np.argmax(preds, axis=1)

    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.5))
    for ax, i, p in zip(axes, idx, pred_labels):
        ax.imshow(X_test[i].squeeze(), cmap="gray")
        true_label = class_names[y_test[i]]
        pred_label = class_names[p]
        color = "green" if p == y_test[i] else "red"
        ax.set_title(f"จริง: {true_label}\nทาย: {pred_label}", fontsize=9, color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_sample.png", dpi=150)
    print(f"บันทึกตัวอย่างการทำนายไว้ที่ {output_dir}/prediction_sample.png")


if __name__ == "__main__":
    test_with_random_images()