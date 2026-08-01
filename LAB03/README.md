# LAB03: Regression & Classification

รายวิชา Machine Learning (04-624-201) | มหาวิทยาลัยเทคโนโลยีราชมงคลธัญบุรี

## เป้าหมาย
ทำนาย **อายุ** (Regression) และ **เพศ** (Classification) จากภาพใบหน้า โดยใช้ Dataset UTKFace เปรียบเทียบประสิทธิภาพระหว่างโมเดลแบบต่างๆ

## Dataset
- **UTKFace** (https://www.kaggle.com/datasets/jangedoo/utkface-new)
- สุ่มใช้ทั้งหมด 4,000 ภาพจากทั้งหมด 23,700+ ภาพ (random_state=42 เพื่อผลลัพธ์ที่ทำซ้ำได้)
- Label (อายุ, เพศ) ฝังอยู่ในชื่อไฟล์รูปแบบ `age_gender_race_date.jpg`

## Pipeline
1. โหลดภาพ + parse label จากชื่อไฟล์
2. Preprocessing: resize เป็น 64x64, แปลง grayscale, flatten เป็นเวกเตอร์ 4096 มิติ, normalize (0-1)
3. แบ่ง Train/Test 80/20
4. PCA ลดมิติจาก 4096 เหลือ 50 มิติ (เก็บ variance ได้ ~88.8%)

## LAB1: Regression (ทำนายอายุ)
เปรียบเทียบ Simple Linear Regression (ใช้ 1 PCA component) กับ Multiple Linear Regression (ใช้ 50 PCA components)

| Model | MAE | MSE | R2 |
|---|---|---|---|
| Simple Linear Reg (Test) | 5.66 | 61.89 | 0.147 |
| Multiple Linear Reg (Test) | 4.59 | 41.31 | 0.430 |

**สรุป:** Multiple Linear Regression แม่นยำกว่า Simple ชัดเจนในทุกตัวชี้วัด เนื่องจากมีฟีเจอร์ให้เรียนรู้มากกว่า

## LAB2: Classification (ทำนายเพศ)
ใช้ Logistic Regression บน PCA 50 มิติ พร้อม Decision Boundary Visualization บน 2 มิติแรก

| Set | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Train | 0.810 | 0.817 | 0.847 | 0.832 |
| Test | 0.794 | 0.794 | 0.846 | 0.819 |

- **AUC = 0.874** (เกณฑ์ดี)
- Confusion Matrix และ ROC Curve อยู่ในไฟล์ notebook

## LAB3: Model Comparison
- **Simple vs Multiple Regression:** Multiple ชนะทุกตัวชี้วัด (R² 0.43 vs 0.147)
- **Train vs Test:** ทั้งสองโมเดล (Regression และ Classification) ไม่มี overfitting ชัดเจน ค่าระหว่าง Train/Test ใกล้เคียงกัน
- **Regression vs Classification:** Classification (ทายเพศ) แม่นยำกว่า Regression (ทายอายุ) อย่างชัดเจน เนื่องจากเป็นปัญหา binary ที่ง่ายกว่าการทายค่าต่อเนื่อง ประกอบกับข้อมูลอายุมีการกระจายตัวไม่สม่ำเสมอ (กลุ่มอายุน้อยมีมากกว่ากลุ่มผู้สูงอายุ)

## เทคโนโลยีที่ใช้
Python, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn (PCA, Linear Regression, Logistic Regression)

## ไฟล์ในโฟลเดอร์นี้
- `LAB03_Regression_Classification.ipynb` — โค้ดฉบับเต็มพร้อมผลลัพธ์
- `README.md` — เอกสารสรุปฉบับนี้
- ตัวของข้อมูล
