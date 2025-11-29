import cv2
import numpy as np
import tensorflow as tf
import os

# class labels – update if your team used different order
LABELS = ["violence", "panic", "fall", "normal"]

def preprocess_frame(frame, img_size=224):
    # Resize and normalize frame
    frame = cv2.resize(frame, (img_size, img_size))
    frame = frame.astype("float32") / 255.0
    return frame

def predict_video(model_path, video_path, img_size=224, stride=5):
    # Load model
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frame_preds = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # process every Nth frame (stride)
        if frame_count % stride == 0:
            x = preprocess_frame(frame, img_size)
            x = np.expand_dims(x, axis=0)  # add batch dimension
            preds = model.predict(x, verbose=0)
            pred_label = LABELS[np.argmax(preds)]
            frame_preds.append(pred_label)
        frame_count += 1

    cap.release()

    # majority vote
    final_pred = max(set(frame_preds), key=frame_preds.count)
    return final_pred, frame_preds

if __name__ == "__main__":
    model_path = "models/inception_model.h5"
    video_path = "data/test_clips/fall_01.mp4"

    final_pred, frame_preds = predict_video(model_path, video_path)
    print("✅ Video:", video_path)
    print("Predicted class:", final_pred)
    print("Sample frame predictions:", frame_preds[:10])
