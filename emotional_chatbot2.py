import streamlit as st
import lmstudio as lms
import cv2
import numpy as np
import os
from PIL import Image
import tensorflow as tf

# --- Emotion detection setup ---
def load_emotion_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'CNN/model_weights.weights.h5')
    
    # Load the Keras model
    if not os.path.exists(model_path):
        st.error(f"Emotion model file not found: {model_path}")
        st.stop()
    
    emotion_model = tf.keras.models.load_model(model_path)

    # Load the Haar Cascade classifier for face detection
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        st.error(f"Haar Cascade face detection model not found!")
        st.stop()

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    return emotion_model, face_cascade

emotion_dict = {
    0: 'neutral', 1: 'happiness', 2: 'surprise',
    3: 'sadness', 4: 'anger', 5: 'disgust', 6: 'fear'
}

# Preprocess face for Keras model
def preprocess_image_for_model(face):
    # Convert to grayscale and resize to (48, 48) - size for FER-2013 model
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    gray_face = cv2.resize(gray_face, (48, 48))  # FER-2013 size
    # Normalize pixel values and add batch/channel dimension
    gray_face = np.expand_dims(gray_face, axis=0)  # Add batch dimension
    gray_face = np.expand_dims(gray_face, axis=-1)  # Add channel dimension
    gray_face = gray_face.astype('float32') / 255.0  # Normalize to [0, 1] range
    return gray_face

# Emotion detection using Haar Cascade and Keras model
def detect_emotion_haar(frame, emotion_model, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        st.warning("No faces detected.")
        return None, frame
    
    for (x, y, w, h) in faces:
        # Crop and process face
        face = frame[y:y+h, x:x+w]
        preprocessed_face = preprocess_image_for_model(face)
        
        # Use the Keras model to predict the emotion
        emotion_probabilities = emotion_model.predict(preprocessed_face)
        emotion_label = np.argmax(emotion_probabilities[0])  # Get the index of the highest probability
        emotion = emotion_dict.get(emotion_label)
        
        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return emotion, frame

# --- Streamlit app ---
st.set_page_config(page_title="Emotional Assistance")
st.title("🗣️ Emotional Assistance Chatbot")

# Load models
if 'emotion_models' not in st.session_state:
    st.session_state.emotion_model, st.session_state.face_cascade = load_emotion_models()

# Initialize chatbot model
if 'model' not in st.session_state:
    st.session_state.model = lms.llm("llama-3.2-1b-instruct")

# Build initial context
if 'full_context' not in st.session_state:
    base = "You are a professional therapist specializing in mental health."
    st.session_state.full_context = [{"role":"system","content":base}]
    st.session_state.messages = []

# Camera capture and emotion detection
def capture_and_detect():
    # Capture the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect emotion and draw face bounding box
        emotion, frame_with_box = detect_emotion_haar(frame, st.session_state.emotion_model, st.session_state.face_cascade)

        # Show the frame with bounding box and emotion
        frame_rgb = cv2.cvtColor(frame_with_box, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        st.image(img_pil, caption="Detected Emotion", use_container_width=True)
        
        if emotion:
            return emotion

# Capture emotion from webcam and update the context dynamically
emotion = capture_and_detect()

if emotion:
    st.success(f"Detected emotion: {emotion}")

    # Update the context with detected emotion
    if 'full_context' in st.session_state:
        base = f"You are a professional therapist specializing in mental health. The user appears to be feeling {emotion}."
        st.session_state.full_context = [{"role": "system", "content": base}]
    
    # Chat interface
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    if user_input := st.chat_input("Say something..."):
        st.session_state.full_context.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Combine conversation context
        conv = ''
        for m in st.session_state.full_context:
            prefix = 'User:' if m['role'] == 'user' else 'Therapist:'
            conv += f"{prefix} {m['content']}\n"
        conv += "Therapist:"
        
        # Generate response using the model
        resp = st.session_state.model.respond(conv)
        st.session_state.messages.append({"role": "assistant", "content": resp})
        
        # Display assistant's message
        with st.chat_message("assistant"):
            st.markdown(resp)

cap.release()
cv2.destroyAllWindows()
