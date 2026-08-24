import numpy as np
import pandas as pd


def find_nearest_animal_to_centroid(X_scaled, centroids, sample_names):
    """
    สำหรับ centroid แต่ละอัน หาว่า 'จุดข้อมูลจริง' อยู่ใกล้ที่สุด
    ใช้หลักการเดียวกับ KNN แต่ k=1 และค้นจาก centroid ไปหาจุดข้อมูล
    """
    X = np.asarray(X_scaled, dtype=np.float32)
    C = np.asarray(centroids, dtype=np.float32)

    X_sq = np.sum(X ** 2, axis=1)
    C_sq = np.sum(C ** 2, axis=1, keepdims=True)
    cross = C @ X.T
    dist_sq = C_sq - 2.0 * cross + X_sq
    dist_sq = np.maximum(dist_sq, 0.0)

    nearest_idx = np.argmin(dist_sq, axis=1)
    return [sample_names[i] for i in nearest_idx]


def majority_label_per_cluster(cluster_labels, true_labels):
    """หา label จริงที่พบมากที่สุดในแต่ละ cluster (majority voting)"""
    df = pd.DataFrame({"cluster": cluster_labels, "true_type": true_labels})
    summary = {}
    for cluster_id, group in df.groupby("cluster"):
        counts = group["true_type"].value_counts()
        summary[cluster_id] = (counts.idxmax(), int(counts.max()), len(group))
    return summary


def clustering_purity(cluster_labels, true_labels):
    """purity score: สัดส่วนจุดที่ตรงกับ label ข้างมากใน cluster ของตัวเอง"""
    df = pd.DataFrame({"cluster": cluster_labels, "true_type": true_labels})
    correct = sum(g["true_type"].value_counts().max() for _, g in df.groupby("cluster"))
    return correct / len(df)