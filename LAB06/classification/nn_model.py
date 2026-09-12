import json
import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape, num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=20, class_weight=None):
    print(f"Training Neural Network for {epochs} Epochs...")
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=epochs, 
        batch_size=32,
        class_weight=class_weight  # รับค่าถ่วงน้ำหนักตรงนี้
    )
    
    model.save("outputs/nn_model.keras")
    with open("outputs/history.json", "w") as f:
        json.dump(history.history, f)
        
    return history