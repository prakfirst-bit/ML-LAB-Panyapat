import os
import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_raw_data, CLASSES
from preprocessing import preprocess_images
from split_data import split_and_save
from cnn_model import build_cnn, train_model
from evaluate import evaluate_model, plot_training_history
from test_cnn import test_with_random_images

OUTPUT_DIR = "outputs"


EXPERIMENTS = [
    {"name": "baseline_2conv_5ep",  "config": {"num_conv_layers": 2, "base_filters": 32, "dense_units": 64}, "epochs": 5},
    {"name": "baseline_2conv_10ep", "config": {"num_conv_layers": 2, "base_filters": 32, "dense_units": 64}, "epochs": 10},
    {"name": "baseline_2conv_15ep", "config": {"num_conv_layers": 2, "base_filters": 32, "dense_units": 64}, "epochs": 15},
    {"name": "shallow_1conv_10ep",  "config": {"num_conv_layers": 1, "base_filters": 32, "dense_units": 64}, "epochs": 10},
    {"name": "deep_3conv_10ep",     "config": {"num_conv_layers": 3, "base_filters": 16, "dense_units": 128}, "epochs": 10},
]


def run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test):
    tf.keras.backend.clear_session()  
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(CLASSES), **exp["config"])
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=exp["epochs"],
        batch_size=64,
        verbose=2,
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    return {
        "name": exp["name"],
        "config": exp["config"],
        "epochs": exp["epochs"],
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history.history,
    }


def save_comparison_table(results, output_dir):
    header = f"{'Experiment':<22}{'Conv':<6}{'Filters':<9}{'Dense':<8}{'Epochs':<8}{'Test Acc':<10}{'Test Loss'}"
    lines = [header, "-" * len(header)]
    for r in results:
        c = r["config"]
        lines.append(
            f"{r['name']:<22}{c['num_conv_layers']:<6}{c['base_filters']:<9}{c['dense_units']:<8}"
            f"{r['epochs']:<8}{r['test_acc']:<10.4f}{r['test_loss']:.4f}"
        )
    table_text = "\n".join(lines)

    print("\n=== สรุปผลเปรียบเทียบทุกการทดลอง ===")
    print(table_text)

    with open(f"{output_dir}/comparison_results.txt", "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"\nบันทึกตารางเปรียบเทียบไว้ที่ {output_dir}/comparison_results.txt")


def plot_comparison(results, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for r in results:
        label = f"{r['name']} ({r['test_acc']*100:.2f}%)"
        h = r["history"]
        axes[0, 0].plot(h["accuracy"], label=label)
        axes[0, 1].plot(h["val_accuracy"], label=label)
        axes[1, 0].plot(h["loss"], label=label)
        axes[1, 1].plot(h["val_loss"], label=label)

    titles = [
        ["(a) Training Accuracy", "(b) Validation Accuracy"],
        ["(c) Training Loss", "(d) Validation Loss"],
    ]
    ylabels = [["Accuracy", "Accuracy"], ["Loss", "Loss"]]

    for i in range(2):
        for j in range(2):
            axes[i, j].set_title(titles[i][j])
            axes[i, j].set_xlabel("Epochs")
            axes[i, j].set_ylabel(ylabels[i][j])
            axes[i, j].legend(fontsize=7, loc="best")
            axes[i, j].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plot.png", dpi=150)
    plt.close()
    print(f"บันทึกกราฟเปรียบเทียบไว้ที่ {output_dir}/comparison_plot.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    
    (x_train, y_train), (x_test, y_test) = load_raw_data()
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)

    
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_and_save(
        x_train, y_train, x_test, y_test, OUTPUT_DIR
    )

    results = []
    for exp in EXPERIMENTS:
        print(f"\n=== เทรน {exp['name']} (config={exp['config']}, epochs={exp['epochs']}) ===")
        r = run_experiment(exp, X_train, y_train, X_val, y_val, X_test, y_test)
        results.append(r)
        print(f"-> Test accuracy: {r['test_acc']:.4f}, Test loss: {r['test_loss']:.4f}")

   
    save_comparison_table(results, OUTPUT_DIR)
    plot_comparison(results, OUTPUT_DIR)

    
    best = max(results, key=lambda r: r["test_acc"])
    print(f"\n>>> Config ที่ดีที่สุด: {best['name']} (test accuracy={best['test_acc']:.4f}) <<<")
    print("กำลังเทรนโมเดลนี้ใหม่แบบเต็ม เพื่อบันทึกโมเดล + สร้างรายงานละเอียด...")

    tf.keras.backend.clear_session()
    best_model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(CLASSES), **best["config"])
    train_model(best_model, X_train, y_train, X_val, y_val, epochs=best["epochs"], output_dir=OUTPUT_DIR)

    evaluate_model(best_model, X_test, y_test, CLASSES, OUTPUT_DIR)
    plot_training_history(f"{OUTPUT_DIR}/history.json", OUTPUT_DIR)
    test_with_random_images(OUTPUT_DIR, n=4)


if __name__ == "__main__":
    main()