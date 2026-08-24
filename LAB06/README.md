# LAB06: Chest X-Ray Pneumonia Classification using Deep Learning (CNN)

This repository contains the full implementation and experimental analysis for **LAB06: Chest X-Ray Pneumonia Classification**. The goal is to build, train, and evaluate a Deep Learning model (Convolutional Neural Network) to automatically classify chest X-ray images into **NORMAL** or **PNEUMONIA**.

---

## 📁 Project Structure

```text
LAB06/
├── classification/
│   ├── data_loader.py       # Data loading, augmentation, and preprocessing
│   ├── model.py             # CNN architecture definitions (Custom CNN / Transfer Learning)
│   ├── train.py             # Model training loop and learning rate scheduling
│   ├── evaluate.py          # Metrics evaluation, confusion matrix & ROC visualization
│   ├── main.py              # Main execution script
│   └── outputs/             # Saved metrics, plots, confusion matrix, and prediction logs
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation and experimental report
