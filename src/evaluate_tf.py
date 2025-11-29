import pandas as pd
import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# ----------------- Load Classes -----------------
CLASSES_PATH = "models/classes.npy"
classes = np.load(CLASSES_PATH, allow_pickle=True)
LABELS = list(classes)

def preprocess_frame(frame, img_size=224):
    frame = cv2.resize(frame, (img_size, img_size))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # ensure RGB
    frame = frame.astype("float32") / 255.0
    return frame

def predict_video(model, video_path, img_size=224, stride=5):
    cap = cv2.VideoCapture(video_path)
    preds = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % stride == 0:
            x = preprocess_frame(frame, img_size)
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x, verbose=0)
            preds.append(np.argmax(pred))
        count += 1
    cap.release()

    if preds:
        final_pred = max(set(preds), key=preds.count)  # majority vote
        return str(classes[final_pred])
    else:
        return "unknown"

def plot_confusion(cm, labels, out_path="outputs/confusion_matrix.png"):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"✅ Saved confusion matrix at {out_path}")

if __name__ == "__main__":
    # load model
    model = tf.keras.models.load_model("models/inception_model.h5", compile=False)

    # load ground truth
    df = pd.read_csv("ground_truth.csv")
    y_true, y_pred = [], []

    for _, row in df.iterrows():
        gt_label = row["label"]
        video = row["video"]
        pred_label = predict_video(model, video)
        y_true.append(gt_label)
        y_pred.append(pred_label)
        print(f"Video: {video} | True: {gt_label} | Pred: {pred_label}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plot_confusion(cm, LABELS)

    # classification report
    report = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)
    print("\nClassification Report:\n", report)
    with open("outputs/metrics.txt", "w") as f:
        f.write(report)
    print("✅ Saved metrics to outputs/metrics.txt")
