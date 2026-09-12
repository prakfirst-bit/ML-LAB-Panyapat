import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from data_loader import load_data
from preprocessing import preprocess_images
from split_data import split_dataset
from nn_model import build_model, train_model
from evaluate import evaluate_and_plot
from test_nn import test_random_samples

def main():
    print("=== Step 1: Loading Chest X-Ray Data ===")
    features, labels, classes = load_data("../chest_xray")
    
    print("\n=== Step 2: Preprocessing & Resizing Images ===")
    X = preprocess_images(features, target_size=(64, 64))
    
    print("\n=== Step 3: Splitting Dataset ===")
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, labels)
    
    print("\n=== Step 4: Building & Training Neural Network Model with Class Weights ===")
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    
    class_weight_dict = {int(i): float(weight) for i, weight in enumerate(class_weights)}
    print(f"Calculated Class Weights: {class_weight_dict}")

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(classes))
    history = train_model(model, X_train, y_train, X_val, y_val, epochs=20, class_weight=class_weight_dict)
    
    print("\n=== Step 5: Evaluating Model Performance ===")
    evaluate_and_plot(model, X_test, y_test, history, classes)
    
    print("\n=== Step 6: Testing Model with Random Samples ===")
    test_random_samples(model, X_test, y_test)
    
    print("\nProcess Completed Successfully! Check output files in the 'outputs' directory.")

if __name__ == "__main__":
    main()