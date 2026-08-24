# LAB05: Image Classification using Support Vector Machine (SVM)

โปรเจกต์นี้เป็นการจำแนกภาพถ่ายรังสีอก (Chest X-Ray Images) เพื่อตรวจวินิจฉัยผู้ป่วย **COVID-19** เปรียบเทียบกับปอดคนปกติ (**Normal**) โดยใช้แบบจำลอง **Support Vector Machine (SVM)** ร่วมกับการทดสอบความคงทนของโมเดลด้วยการเติมสัญญาณรบกวน (Gaussian Noise)

---

## 📁 โครงสร้างไดเรกทอรี (Project Structure)

```text
LAB05/
├── classification/
│   ├── data_loader.py          # ดึงและปรับขนาดภาพ (Resize)
│   ├── preprocessing.py        # Flatten ภาพ และทำ Feature Standardization
│   ├── split_data.py           # แบ่งข้อมูลเป็น Train / Test set
│   ├── svm_model.py            # สร้างและฝึกสอนโมเดล SVM
│   ├── evaluate.py             # วัดประสิทธิภาพโมเดล (Accuracy & Confusion Matrix)
│   ├── main.py                 # สคริปต์หลักรันทั้ง Workflow
│   └── outputs/                # ผลลัพธ์จากการรัน (Models, Metrics, Plots)
├── Covid19-dataset/            # ชุดข้อมูลภาพถ่ายรังสีอก (Normal vs COVID)
├── link-data.txt               # แหล่งที่มาของ Dataset
├── requirements.txt            # รายการ ไลบรารีที่จำเป็น
└── README.md                   # เอกสารอธิบายโปรเจกต์