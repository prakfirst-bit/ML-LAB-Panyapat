import numpy as np
import matplotlib.pyplot as plt
import json

def test_random_samples(model, X_test, y_test):
    with open("outputs/classes.json") as f:
        classes = json.load(f)
        
    # สุ่มเลือกภาพจาก Test set มา 4 รูป[cite: 1]
    indices = np.random.choice(len(X_test), 4, replace=False)
    plt.figure(figsize=(12, 4))
    
    for i, idx in enumerate(indices):
        img = X_test[idx]
        true_label = classes[y_test[idx]]
        
        # ทำนายผล
        pred_prob = model.predict(np.expand_dims(img, axis=0), verbose=0)
        pred_idx = np.argmax(pred_prob)
        pred_label = classes[pred_idx]
        confidence = pred_prob[0][pred_idx] * 100
        
        plt.subplot(1, 4, i+1)
        plt.imshow(img)
        # แสดงผลสีเขียวหากทายถูก สีแดงหากทายผิด
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)", color=color)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig("outputs/prediction_sample.png")
    plt.close()
    print("Saved sample predictions to 'outputs/prediction_sample.png'")