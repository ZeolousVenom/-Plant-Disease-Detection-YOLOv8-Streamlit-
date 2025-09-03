# 🌱 Plant Disease Detection (YOLOv8 + Streamlit)

A machine learning app that detects plant diseases using YOLOv8, OpenCV, and Streamlit.  

## 📌 Overview
This project is an AI-powered web application that detects **plant leaf diseases** using a custom-trained **YOLOv8 model** integrated with **Streamlit**.  
It helps farmers, researchers, and agriculture professionals identify plant health issues quickly and accurately.

---

## 🚀 Features
- ✅ Upload plant leaf images and get instant disease detection results.
- ✅ Built using **YOLOv8 (Ultralytics)** for object detection.
- ✅ Interactive **Streamlit web interface** for easy use.
- ✅ Provides **confidence scores** for predictions.
- ✅ Lightweight and works on local systems.

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **YOLOv8 (Ultralytics)**
- **Streamlit**
- **OpenCV**
- **NumPy & Pandas**

---

## 📂 Project Structure
Plant-Disease-Detection-YOLOv8-Streamlit/
│── dataset/ # Training dataset
│── runs/ # YOLOv8 training results
│── best.pt # Trained YOLOv8 model weights
│── app.py # Streamlit application
│── requirements.txt # Python dependencies
│── README.txt # Project documentation

---

## ⚡ Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Plant-Disease-Detection-YOLOv8-Streamlit.git
   cd Plant-Disease-Detection-YOLOv8-Streamlit
Create a virtual environment (recommended):

python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows
Install dependencies:

---
2. install dependency :
pip install -r requirements.txt

---
3. Run the Streamlit app:
streamlit run app.py

---
4. Train a model :
If you want to retrain the model on a new dataset:

yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640

---
## 📦 Installation
```bash
git clone https://github.com/username/plant-disease-detection.git
cd plant-disease-detection
pip install -r requirements.txt
