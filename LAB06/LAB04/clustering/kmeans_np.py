import numpy as np


class KMeansNP:
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iters_ = None

    def _init_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[idx].copy()

    @staticmethod
    def _pairwise_sq_distance(X, centroids):
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)         # (n, 1)
        c_sq = np.sum(centroids ** 2, axis=1)                  # (k,)
        cross = X @ centroids.T                                 # (n, k)
        dist_sq = X_sq - 2.0 * cross + c_sq
        return np.maximum(dist_sq, 0.0)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            dist_sq = self._pairwise_sq_distance(X, centroids)  # (n, k)
            labels = np.argmin(dist_sq, axis=1)                  # (n,)

            new_centroids = np.zeros_like(centroids)
            for cluster_id in range(self.k):
                points_in_cluster = X[labels == cluster_id]
                if len(points_in_cluster) > 0:
                    new_centroids[cluster_id] = points_in_cluster.mean(axis=0)
                else:
                    # cluster ว่าง: คงตำแหน่งเดิมไว้
                    new_centroids[cluster_id] = centroids[cluster_id]

            shift = np.sum((new_centroids - centroids) ** 2)
            centroids = new_centroids

            if shift < self.tol:
                break

        # คำนวณผลลัพธ์สุดท้าย
        dist_sq = self._pairwise_sq_distance(X, centroids)
        labels = np.argmin(dist_sq, axis=1)
        min_dist_sq = np.min(dist_sq, axis=1)

        self.centroids = centroids
        self.labels_ = labels
        self.inertia_ = float(np.sum(min_dist_sq))
        self.n_iters_ = iteration + 1

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        dist_sq = self._pairwise_sq_distance(X, self.centroids)
        return np.argmin(dist_sq, axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=0, scale=0.5, size=(20, 2)),
        rng.normal(loc=5, scale=0.5, size=(20, 2)),
    ]).astype(np.float32)

    model = KMeansNP(k=2).fit(X)
    print("Centroids:\n", model.centroids)
    print("Inertia:", model.inertia_)
    print("Iterations used:", model.n_iters_)