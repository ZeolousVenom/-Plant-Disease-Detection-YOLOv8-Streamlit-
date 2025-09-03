# app.py
import streamlit as st
import os
from PIL import Image
import cv2
import numpy as np

from detect import RiceDetector
from utils import cv2_from_uploaded, crop_with_padding, draw_detections
from disease_info import DISEASE_INFO

# ----------------------------
# Load YOLO model only
# ----------------------------
@st.cache_resource
def load_detector():
    yolo_weights = os.getenv("YOLO_WEIGHTS", "models/yolov8n-rice.pt")
    detector = RiceDetector(yolo_weights)
    return detector

detector = load_detector()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="🌾 Rice Plant Disease Detection", layout="wide")
st.title("🌾 Rice Plant Disease Detection (YOLOv8 Only)")

uploaded_file = st.file_uploader("Upload a rice leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load & show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert for OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect diseases with YOLOv8
    results = detector.detect(img_cv)

    if results:
        st.subheader("Detection Results")

        for det in results:
            disease = det["name"]
            conf = det["confidence"]

            st.markdown(
                f"**Disease:** {disease}  \n"
                f"**Confidence:** {conf*100:.2f}%"
            )

            # Show treatment info if available
            if disease in DISEASE_INFO:
                st.info(DISEASE_INFO[disease])
            else:
                st.warning("No treatment information available for this class.")

        # Display image with bounding boxes
        st.subheader("Detected Regions")
        st.image(detector.last_result_plot, caption="Detection Output", use_column_width=True)

    else:
        st.warning("No disease detected.")
