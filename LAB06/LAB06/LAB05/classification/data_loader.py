import os
import cv2
import numpy as np

def load_images_from_folder(folder, label, img_size=(64, 64)):
    images, labels = [], []
    if not os.path.isdir(folder):
        return images, labels
    files = sorted(os.listdir(folder))
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        images.append(img)
        labels.append(label)
    return images, labels

def load_dataset(base_dir, img_size=(64, 64)):
    """
    รวมข้อมูลจากทั้ง train/ และ test/ ของ dataset เดิมเข้าด้วยกัน
    เพราะรูปมีน้อย เราจะเอามา split เอง (80/20) ในขั้นตอนถัดไปแทน
    ใช้แค่ 2 คลาส: Covid กับ Normal (ไม่เอา Viral Pneumonia)
    """
    covid_imgs, covid_labels = [], []
    normal_imgs, normal_labels = [], []

    for split in ["train", "test"]:
        covid_dir = os.path.join(base_dir, split, "Covid")
        normal_dir = os.path.join(base_dir, split, "Normal")

        c_imgs, c_labels = load_images_from_folder(covid_dir, 1, img_size)
        n_imgs, n_labels = load_images_from_folder(normal_dir, 0, img_size)

        covid_imgs += c_imgs
        covid_labels += c_labels
        normal_imgs += n_imgs
        normal_labels += n_labels

    print(f"COVID images: {len(covid_imgs)}, Normal images: {len(normal_imgs)}")

    X = np.array(covid_imgs + normal_imgs)
    y = np.array(covid_labels + normal_labels)
    return X, y