import streamlit as st
import cv2
import onnxruntime as ort
import numpy as np
import time
import requests
import os

# Paths to SSD model files
PROTOTXT_PATH = "Custom_VGG13/RFB-320/RFB-320.prototxt"
CAFFEMODEL_PATH = "Custom_VGG13/RFB-320/RFB-320.caffemodel"

# Verify that model files exist
assert os.path.isfile(PROTOTXT_PATH), f"File not found: {PROTOTXT_PATH}"
assert os.path.isfile(CAFFEMODEL_PATH), f"File not found: {CAFFEMODEL_PATH}"

# Initialize ONNX model for emotion detection
session = ort.InferenceSession("Custom_VGG13/emotion-ferplus-8.onnx")

# Function to send chat requests to LM Studio
def query_lmstudio(messages, model="llama-3.2-1b-instruct"):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]

# Emotion labels
emotion_labels = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt"
]

# Function to detect emotion using SSD face detector
def detect_emotion():
    # Load the face detector
    net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, CAFFEMODEL_PATH)

    cap = cv2.VideoCapture(0)
    time.sleep(2)  # Warm up the camera

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "No face detected"

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )

    net.setInput(blob)
    detections = net.forward()

    # Select highest confidence detection
    i = np.argmax(detections[0, 0, :, 2])
    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY, startX:endX]

        # Preprocess face for emotion model
        face_blob = cv2.resize(
            cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (64, 64)
        ).astype(np.float32)
        face_blob = face_blob[np.newaxis, np.newaxis, :, :]  # (1,1,64,64)

        outputs = session.run(None, {"Input3": face_blob})
        emotion_idx = np.argmax(outputs[0])
        return emotion_labels[emotion_idx]

    return "No face detected"

# Streamlit interface
st.title("Emotional Chatbot Therapist")

# Detect emotion once per session
if "emotion" not in st.session_state:
    st.session_state.emotion = detect_emotion()

# Show detected emotion
st.info(f"Detected Emotion: **{st.session_state.emotion}**")

# Compose system prompt
emotion_prompt = (
    f"You are a helpful and empathetic therapist. "
    f"The user seems to be feeling {st.session_state.emotion}. "
    "Adjust your tone accordingly."
)

# User input field
user_input = st.text_input("You:", "")

# On user message, query LM Studio and display response
if user_input:
    response = query_lmstudio(
        messages=[
            {"role": "system", "content": emotion_prompt},
            {"role": "user", "content": user_input}
        ]
    )
    st.markdown(f"**Therapist:** {response}")
