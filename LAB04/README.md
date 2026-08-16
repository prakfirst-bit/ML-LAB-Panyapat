# 🍇 Machine Learning Lab 4: KNN & K-Means Implementation from Scratch

โปรเจกต์นี้เป็นการพัฒนากระบวนการเรียนรู้ของเครื่อง (Machine Learning) สองรูปแบบ ได้แก่ **K-Nearest Neighbors (KNN Classifier)** สำหรับงาน Classification และ **K-Means Clustering** สำหรับงาน Clustering โดยเขียนอัลกอริทึมขึ้นเองตั้งแต่ต้นด้วย **NumPy Vectorization Matrix Computation** (ไม่ใช้โมเดลสำเร็จรูปจาก `scikit-learn` หรือ `TensorFlow`) ประมวลผลบน **Raisin Dataset**

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB04/
├── classification/
│   ├── data_loader.py       # โหลดและแปลงข้อมูล (StandardScaler, LabelEncoder)
│   ├── evaluate.py          # คำนวณ Accuracy, Report และวาดกราฟ
│   ├── knn_np.py            # อัลกอริทึม KNN จาก Scratch ด้วย NumPy
│   ├── main.py              # สคริปต์หลักสำหรับเทรนและประเมินผล KNN
│   └── outputs/             # ผลลัพธ์กราฟ k-curve, confusion matrix และ predictions.csv
├── clustering/
│   ├── data_loader.py       # โหลดและแปลงข้อมูลสำหรับ Clustering
│   ├── kmeans_np.py         # อัลกอริทึม K-Means จาก Scratch ด้วย NumPy
│   ├── knn_tools.py         # เครื่องมือช่วยตีความผลลัพธ์ (Centroid distance, Purity score)
│   ├── main.py              # สคริปต์หลักสำหรับหา k (Elbow) และประมวลผล K-Means
│   ├── visualize.py         # การลดมิติข้อมูลด้วย PCA และวาดกราฟ 2D
│   └── outputs/             # ผลลัพธ์กราฟ Elbow, PCA Cluster และไฟล์สรุปการจัดกลุ่ม
├── .gitignore
├── README.md
└── requirements.txt