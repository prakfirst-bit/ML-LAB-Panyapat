import cv2
import numpy as np
from tqdm import tqdm

def preprocess_images(img_paths, target_size=(64, 64)):
    processed_imgs = []
    print("Preprocessing and Standardizing Chest X-Ray Images...")
    
    for path in tqdm(img_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        # แปลงเป็น RGB เพื่อความสม่ำเสมอ
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize ภาพ X-Ray ให้เท่ากัน
        img_resized = cv2.resize(img_rgb, target_size)
        processed_imgs.append(img_resized)
        
    # Standardize Pixel Values (0-255 -> 0.0-1.0)[cite: 1]
    X = np.array(processed_imgs, dtype='float32') / 255.0  
    return X