import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import os

# -------------------- Model & Classes --------------------
MODEL_PATH = "models/inception_model.h5"
CLASSES_PATH = "models/classes.npy"

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

classes = np.load(CLASSES_PATH, allow_pickle=True)
IMG_SIZE = 224

# -------------------- Preprocessing --------------------
def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # ensure RGB
    frame = frame.astype("float32") / 255.0
    return np.expand_dims(frame, axis=0)             # (1,224,224,3)

# -------------------- Prediction --------------------
def predict_video(video_path, stride=10):
    cap = cv2.VideoCapture(video_path)
    preds = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % stride == 0:
            x = preprocess_frame(frame)
            try:
                pred = model.predict(x, verbose=0)
                preds.append(np.argmax(pred))
            except Exception as e:
                print("Prediction error:", e)
        frame_count += 1

    cap.release()

    if preds:
        final_idx = max(set(preds), key=preds.count)   # majority vote
        return str(classes[final_idx])
    else:
        return "unknown"

# -------------------- Streamlit UI --------------------
st.title("🎥 Crowd Anomaly Detection Demo")
st.write("Upload a test video and the model will predict the anomaly type.")

uploaded_file = st.file_uploader("Upload a .mp4 video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    st.video(video_path)

    if st.button("Run Prediction"):
        st.write("⏳ Running prediction...")
        pred = predict_video(video_path)
        st.success(f"✅ Predicted Class: **{pred}**")
