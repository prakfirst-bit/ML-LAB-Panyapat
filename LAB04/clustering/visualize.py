import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ตั้งค่าให้ใช้ฟอนต์ภาษาไทยของ Windows (เช่น Tahoma หรือ Leelawadee UI)
plt.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False  # แก้ปัญหาเครื่องหมายลบแสดงผลผิดพลาด

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def plot_elbow(k_range, inertias, save_path=None):
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "01_elbow.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.plot(list(k_range), inertias, marker="o", color="#E07A5F")
    plt.xlabel("จำนวนกลุ่ม (k)")
    plt.ylabel("Inertia (ผลรวมระยะทางยกกำลังสอง)")
    plt.title("Elbow Method สำหรับหาค่า k ที่เหมาะสม")
    plt.xticks(list(k_range))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึกกราฟ elbow ที่: {save_path}")


def plot_clusters_2d(X_scaled, cluster_labels, centroids, save_path=None):
    """ลดมิติจาก features เหลือ 2D ด้วย PCA แล้ว plot จุดข้อมูล + centroid"""
    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "02_clusters.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_scaled)
    centroids_2d = pca.transform(centroids)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels,
                           cmap="tab10", s=50, alpha=0.7, edgecolors="k", linewidths=0.3)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c="red", marker="X",
                s=250, edgecolors="black", label="Centroids")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title("ผลลัพธ์การจัดกลุ่มเมล็ดลูกเกดด้วย K-Means (PCA 2D)")
    plt.legend()
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"บันทึกกราฟ clusters ที่: {save_path}")