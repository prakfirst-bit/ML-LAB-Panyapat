import numpy as np
class KNNClassifierNP:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.num_classes = None

    def fit(self, X_train, y_train):
        self.X_train = np.asarray(X_train, dtype=np.float32)
        self.y_train = np.asarray(y_train, dtype=np.int32)
        self.num_classes = int(np.max(y_train)) + 1
        return self

    def _pairwise_euclidean_distance(self, X_test):

        X_test = np.asarray(X_test, dtype=np.float32)

        test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)      # (n_test, 1)
        train_sq = np.sum(self.X_train ** 2, axis=1)               # (n_train,)
        cross_term = X_test @ self.X_train.T                       # (n_test, n_train)

        dist_sq = test_sq - 2.0 * cross_term + train_sq
        dist_sq = np.maximum(dist_sq, 0.0)  # กัน floating point error ติดลบเล็กน้อย
        return np.sqrt(dist_sq)

    def predict(self, X_test):
        if self.X_train is None:
            raise RuntimeError("ต้องเรียก fit() ก่อน predict()")

        distances = self._pairwise_euclidean_distance(X_test)  # (n_test, n_train)

        # หา index ของ k จุดที่ระยะทางน้อยที่สุด (เรียงจากน้อยไปมาก แล้วตัด k ตัวแรก)
        top_k_idx = np.argsort(distances, axis=1)[:, :self.k]   # (n_test, k)

        # ดึง label ของเพื่อนบ้าน k ตัว
        neighbor_labels = self.y_train[top_k_idx]  # (n_test, k)

        # โหวตเสียงข้างมากทีละแถว
        predictions = np.array([
            np.bincount(row, minlength=self.num_classes).argmax()
            for row in neighbor_labels
        ])
        return predictions.astype(np.int32)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(50, 4)).astype(np.float32)
    y_train = rng.integers(0, 2, size=50).astype(np.int32)
    X_test = rng.normal(size=(5, 4)).astype(np.float32)

    model = KNNClassifierNP(k=3).fit(X_train, y_train)
    print("Predictions:", model.predict(X_test))