import os
import pandas as pd

from data_loader import prepare_data
from kmeans_np import KMeansNP               # <-- เปลี่ยนจาก kmeans_tf import KMeansTF
from knn_tools import (
    find_nearest_animal_to_centroid,
    majority_label_per_cluster,
    clustering_purity,
)
from visualize import plot_elbow, plot_clusters_2d, OUTPUT_DIR

K_RANGE = range(2, 8)
FINAL_K = 2  # Raisin dataset มี 2 พันธุ์จริง (Kecimen, Besni)


def main():
    print("=" * 60)
    print("LAB 4: K-Means Clustering บน Raisin Dataset (NumPy)")
    print("=" * 60)

    X_scaled, df, feature_cols, scaler = prepare_data()
    sample_names = [f"raisin_{i+1}" for i in range(len(df))]
    true_types = df["Class"].values

    print(f"จำนวนข้อมูล: {X_scaled.shape[0]} เมล็ด, {X_scaled.shape[1]} features\n")

    inertias = []
    for k in K_RANGE:
        model = KMeansNP(k=k).fit(X_scaled)    # <-- เปลี่ยนชื่อคลาส
        inertias.append(model.inertia_)
        print(f"k = {k:2d} -> Inertia = {model.inertia_:.2f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_elbow(K_RANGE, inertias)

    final_model = KMeansNP(k=FINAL_K).fit(X_scaled)
    cluster_labels = final_model.labels_

    plot_clusters_2d(X_scaled, cluster_labels, final_model.centroids)

    nearest_samples = find_nearest_animal_to_centroid(
        X_scaled, final_model.centroids, sample_names
    )
    majority = majority_label_per_cluster(cluster_labels, true_types)
    purity = clustering_purity(cluster_labels, true_types)

    print(f"\nPurity score (k={FINAL_K}): {purity:.4f}")

    summary_rows = []
    for cluster_id in range(FINAL_K):
        maj_type, count, total = majority.get(cluster_id, ("-", 0, 0))
        summary_rows.append({
            "cluster_id": cluster_id,
            "representative_sample": nearest_samples[cluster_id],
            "majority_class": maj_type,
            "majority_count": count,
            "total_in_cluster": total,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "cluster_summary.csv"), index=False)
    print("\n", summary_df)

    result_df = df.copy()
    result_df["cluster"] = cluster_labels
    result_df.to_csv(os.path.join(OUTPUT_DIR, "clustered_animals.csv"), index=False)

    print("\nเสร็จสิ้น! ผลลัพธ์อยู่ในโฟลเดอร์ clustering/outputs/")


if __name__ == "__main__":
    main()