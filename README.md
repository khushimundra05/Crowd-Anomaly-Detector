# 📌 **Crowd Anomaly Detection**

This project detects **abnormal events in crowd videos**—such as **violence, stampedes, sudden collapses, and vehicle intrusions**—using a combination of **spatial + temporal deep learning models**.
The system works in real-time and highlights video segments where anomalies occur.

---

## 🚀 **Tech Stack**

- **Python** – Core programming
- **TensorFlow / Keras** – Deep learning (InceptionNet + LSTM)
- **OpenCV** – Video processing
- **NumPy / Pandas** – Data preprocessing
- **Streamlit** – Web interface

---

## 🧠 **Model Overview**

This system uses a **hybrid deep learning architecture**:

- **InceptionNet** → Spatial feature extraction
- **LSTM** → Temporal sequence modeling
- Combined model → **Classifies normal vs anomalous behavior**

### 📊 **Model Performance**

- **Training Accuracy:** 84.31%
- **Test Accuracy:** 62.11%
- **Loss:** 0.5326

---

## ⭐ **Features**

- Real-time anomaly detection
- Detects multiple types of crowd anomalies
- Upload any video for testing
- Visual output showing anomaly predictions
- Modular, extendable architecture

---

## 🔧 **Installation**

### **1. Clone the repository**

```bash
git clone https://github.com/khushimundra05/Crowd-Anomaly-Detector.git
cd Crowd-Anomaly-Detector
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the app**

```bash
streamlit run app.py
```

---

## 📁 **Project Structure**

```
Crowd-Anomaly-Detector/
│── app.py
│── model/
│── utils/
│── data/
│── README.md
│── requirements.txt
```

---

## 🔍 **How It Works**

1. User uploads a video through the Streamlit UI
2. Frames extracted using OpenCV
3. InceptionNet extracts spatial features
4. LSTM analyzes temporal dependencies
5. Model predicts normal/anomalous behavior
6. Results displayed with highlighted segments

---

## 🚀 **Future Improvements**

- Expand dataset for better generalization
- Add attention mechanisms
- Improve real-time performance
- Deploy as cloud API
- Add live CCTV feed support

---

## 👥 **Contributors**

- **Khushi Mundra**
- **Tanvi Chhaparia**
- **Richa Doshi**
