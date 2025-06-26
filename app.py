import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import time

st.set_page_config(page_title="Face Mask Detector", layout="centered")

# Load model and Haar cascade
model = load_model("Real_World_Mask_Detection.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
IMG_SIZE = 224  # Must match model input

# UI Setup
st.markdown("<h1 style='text-align: center;'>😷 Real-Time Face Mask Detector</h1>", unsafe_allow_html=True)
start_button = st.button("Start Detection")
frame_window = st.image([])

# Main detection function
def detect_and_display():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("⚠️ Failed to access webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            try:
                # Convert to RGB for model
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
                normalized = resized / 255.0
                input_tensor = np.expand_dims(normalized, axis=0)

                # Predict
                prediction = model.predict(input_tensor)[0][0]
                if prediction < 0.5:
                    label = "Mask"
                    confidence = 1 - prediction
                    color = (0, 255, 0)  # Green
                else:
                    label = "No Mask"
                    confidence = prediction
                    color = (0, 0, 255)  # Red

                label_text = f"{label} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print("Error during prediction:", e)
                continue

        # Convert BGR → RGB for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame)

        # Small delay to avoid CPU overuse
        time.sleep(0.05)

    cap.release()

# Trigger detection
if start_button:
    detect_and_display()
