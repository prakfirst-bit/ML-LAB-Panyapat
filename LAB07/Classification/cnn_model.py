import json
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape=(28, 28, 1), num_conv_layers=2, base_filters=32,
              dense_units=64, num_classes=10):
    model = models.Sequential(name=f"fashion_cnn_conv{num_conv_layers}_dense{dense_units}")
    model.add(layers.Input(shape=input_shape))

    filters = base_filters
    for _ in range(num_conv_layers):
        model.add(layers.Conv2D(filters, (3, 3), activation="relu", padding="same"))
        model.add(layers.MaxPooling2D((2, 2)))
        filters *= 2  # ชั้นถัดไปเพิ่ม filter เป็น 2 เท่า

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation="relu"))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=64,
                 output_dir="outputs"):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )
    model.save(f"{output_dir}/cnn_model.keras")
    with open(f"{output_dir}/history.json", "w") as f:
        json.dump(history.history, f)
    return history


def load_trained_model(path="outputs/cnn_model.keras"):
    return tf.keras.models.load_model(path)