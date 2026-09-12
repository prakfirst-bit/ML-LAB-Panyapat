# LAB06: การจำแนกโรคปอดบวมจากภาพถ่ายรังสีทรวงอกด้วยโครงข่ายประสาทเทียม (Neural Network / MLP)

โปรเจกต์นี้เป็นส่วนหนึ่งของ **LAB06: Chest X-Ray Pneumonia Classification** โดยมุ่งเน้นการใช้โครงข่ายประสาทเทียมแบบ Multi-Layer Perceptron (MLP / Fully Connected Neural Network) ในการจำแนกภาพถ่ายรังสีทรวงอกออกเป็น 2 คลาส ได้แก่ **ปกติ (NORMAL)** และ **ปอดบวม (PNEUMONIA)**

---

## 📁 โครงสร้างโปรเจกต์ (Project Structure)

```text
LAB06/
├── classification/
│   ├── data_loader.py       # การโหลดข้อมูล การปรับขนาดภาพ และทำ Flattening/Normalization
│   ├── model.py             # การกำหนดสถาปัตยกรรมโครงข่ายประสาทเทียม (Neural Network Architecture)
│   ├── train.py             # ลูปการเทรนโมเดลและการปรับค่า Weight/Bias
│   ├── evaluate.py          # การประเมินผลโมเดล (Accuracy, Loss, Confusion Matrix)
│   ├── main.py              # สคริปต์หลักสำหรับสั่งรันโปรเซสทั้งหมด
│   └── outputs/             # โฟลเดอร์เก็บผลลัพธ์ กราฟประสิทธิภาพ และ Log การประเมินผล
├── requirements.txt         # รายการ ไลบรารีที่จำเป็นต้องติดตั้ง
└── README.md                # เอกสารอธิบายโปรเจกต์และรายงานผลการทดลอง
