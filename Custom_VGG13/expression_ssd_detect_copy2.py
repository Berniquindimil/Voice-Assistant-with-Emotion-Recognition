import streamlit as st
import lmstudio as lms
import cv2
import numpy as np
import os
from PIL import Image

# --- Emotion detection setup ---
def load_emotion_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, 'emotion-ferplus-8.onnx')
    emotion_net = cv2.dnn.readNetFromONNX(onnx_path)

    # Load the Haar Cascade classifier for face detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        st.error(f"Haar Cascade face detection model not found!")
        st.stop()

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return emotion_net, face_cascade

emotion_dict = {
    0: 'neutral', 1: 'happiness', 2: 'surprise',
    3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'
}

# Precompute SSD priors (kept as is for emotion detection)
from math import ceil
image_std = 128.0
strides = [8.0, 16.0, 32.0, 64.0]
min_boxes = [[10.0,16.0,24.0],[32.0,48.0],[64.0,96.0],[128.0,192.0,256.0]]

def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    priors = []
    for idx in range(len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][idx]
        scale_h = image_size[1] / shrinkage_list[1][idx]
        for j in range(feature_map_list[1][idx]):
            for i in range(feature_map_list[0][idx]):
                x_center = (i + .5) / scale_w
                y_center = (j + .5) / scale_h
                for mb in min_boxes[idx]:
                    w_ = mb / image_size[0]
                    h_ = mb / image_size[1]
                    priors.append([x_center, y_center, w_, h_])
    return np.clip(priors, 0.0, 1.0)

def define_priors(image_size):
    fmap_sizes = [[ceil(image_size[0]/s) for s in strides],
                  [ceil(image_size[1]/s) for s in strides]]
    shrink = [strides, strides]
    return generate_priors(fmap_sizes, shrink, image_size, min_boxes)

priors = define_priors([320,240])

# Emotion detection using Haar Cascade
def detect_emotion_haar(img_pil, emotion_net, face_cascade, conf_threshold=0.3):
    # Convert and inspect frame
    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        st.warning("No faces detected.")
        return None
    
    # Loop through the faces and choose the first one
    for (x, y, w, h) in faces:
        # Crop and process face
        face = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, (64, 64))
        blob2 = cv2.dnn.blobFromImage(gray_face, scalefactor=1 / 255.0)
        emotion_net.setInput(blob2)
        out = emotion_net.forward()
        label = int(np.argmax(out[0]))
        emotion = emotion_dict.get(label)
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return emotion, frame

# --- Streamlit app ---
st.set_page_config(page_title="Emotional Assistance")
st.title("🗣️ Emotional Assistance Chatbot")

# Load models
if 'emotion_models' not in st.session_state:
    st.session_state.emotion_net, st.session_state.face_cascade = load_emotion_models()

# Camera capture and emotion detection
def capture_and_detect():
    img_file = st.camera_input("Capture your face for emotion detection")
    if not img_file:
        return None
    img = Image.open(img_file)
    st.image(img, caption="Input from camera", use_container_width=True)
    emotion, frame_with_box = detect_emotion_haar(img,
                                                 st.session_state.emotion_net,
                                                 st.session_state.face_cascade,
                                                 conf_threshold=0.3)
    st.image(frame_with_box, caption="Face detection with emotion", use_container_width=True)
    return emotion

emotion = capture_and_detect()
if emotion:
    st.success(f"Detected emotion: {emotion}")

# Initialize chatbot model
if 'model' not in st.session_state:
    st.session_state.model = lms.llm("llama-3.2-1b-instruct")

# Build initial context
if 'full_context' not in st.session_state:
    base = "You are a professional therapist specializing in mental health."
    if emotion:
        base += f" The user appears to be feeling {emotion}."
    base += " You respond with empathy, validate emotions, and offer guidance non-judgmentally."
    st.session_state.full_context = [{"role":"system","content":base}]
    st.session_state.messages = []

# Chat interface
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if user_input := st.chat_input("Say something..."):
    st.session_state.full_context.append({"role":"user","content":user_input})
    st.session_state.messages.append({"role":"user","content":user_input})
    conv = ''
    for m in st.session_state.full_context:
        prefix = 'User:' if m['role']=='user' else 'Therapist:'
        conv += f"{prefix} {m['content']}\n"
    conv += "Therapist:"
    resp = st.session_state.model.respond(conv)
    st.session_state.messages.append({"role":"assistant","content":resp})
    with st.chat_message("assistant"):
        st.markdown(resp)
