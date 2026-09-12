import os
import cv2
import json
import numpy as np

def load_data(data_dir="../chest_xray"):
    features, labels = [], []
    
    # ตรวจสอบคลาส (NORMAL, PNEUMONIA)
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()
    
    label_map = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"Found classes: {label_map}")
    
    for cls_name in classes:
        cls_path = os.path.join(data_dir, cls_name)
        img_names = os.listdir(cls_path)
        print(f"Loading {len(img_names)} images from class: {cls_name}...")
        
        for img_name in img_names:
            img_path = os.path.join(cls_path, img_name)
            # ข้ามไฟล์ที่ไม่ใช่รูปภาพ หรือไฟล์ซ่อนระบบ
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    features.append(img_path)
                    labels.append(label_map[cls_name])
            except Exception as e:
                continue
                
    os.makedirs("outputs", exist_ok=True)
    np.save("outputs/features.npy", np.array(features))
    np.save("outputs/labels.npy", np.array(labels))
    with open("outputs/classes.json", "w") as f:
        json.dump(classes, f)
        
    return np.array(features), np.array(labels), classes