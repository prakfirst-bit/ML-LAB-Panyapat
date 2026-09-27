
import tensorflow as tf

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def load_raw_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    print(f"โหลดข้อมูลสำเร็จ: train={x_train.shape}, test={x_test.shape}")
    return (x_train, y_train), (x_test, y_test)