import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from CNN.CNN_model import model
import lmstudio as lms
import tensorflow as tf

st.title("Emotional Assistance")

# --- EmotionAgent Definition ---
class EmotionAgent:
    def __init__(self, llm_model, emotion="neutral"):
        self.model = llm_model
        self.emotion = emotion
        self.system_prompt = self.build_system_prompt()
        self.full_context = [{"role": "system", "content": self.system_prompt}]
        self.messages = []

    def build_system_prompt(self):
        base = (
            "You are a professional therapist specializing in mental health. "
            "You listen with empathy, validate emotions, and offer guidance without judgment. "
            "You respond in a kind, clear, and emotionally adaptive manner focused on emotional well-being."
        )
        if self.emotion and self.emotion.lower() != "neutral":
            base += f" The user appears to be feeling {self.emotion.lower()}, so respond appropriately for someone who feels this way."
        else:
            base += " The user's emotion appears to be neutral; ask gentle, open-ended questions to explore how they feel."
        return base

    def update_emotion(self, new_emotion):
        self.emotion = new_emotion
        self.system_prompt = self.build_system_prompt()
        # reset context with updated system prompt
        self.full_context = [{"role": "system", "content": self.system_prompt}] + self.messages

    def add_user_input(self, user_input):
        self.messages.append({"role": "user", "content": user_input})
        self.full_context.append({"role": "user", "content": user_input})

    def get_response(self):
        # build the complete prompt text
        prompt_text = ''
        for msg in self.full_context:
            if msg['role'] == 'system':
                prompt_text += msg['content'] + "\n"
            elif msg['role'] == 'user':
                prompt_text += f"User: {msg['content']}\n"
            elif msg['role'] == 'assistant':
                prompt_text += f"Therapist: {msg['content']}\n"
        prompt_text += "Therapist:"
        # get model response
        response = self.model.respond(prompt_text)
        # store assistant response
        self.messages.append({"role": "assistant", "content": response})
        self.full_context.append({"role": "assistant", "content": response})
        return response

# --- Emotion Detection Setup ---
# Load emotion model weights once
if "emotion_model" not in st.session_state:
    model.load_weights(
        "/Users/bernardoquindimil/Code/Berniquindimil/Proyect/CNN/model_weights.weights.h5"
    )
    st.session_state.emotion_model = model

# Load OpenCV's pre-trained face detector once
if "face_cascade" not in st.session_state:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    st.session_state.face_cascade = face_cascade

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# --- Streamlit UI: Emotion Capture ---
img_file = st.camera_input("Take a photo to analyze your emotion")

detected_emotion = "neutral"
if img_file is not None:
    pil_img = Image.open(img_file)
    rgb_img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(rgb_img)
    gray = np.array(pil_img.convert("L"))
    faces = st.session_state.face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.expand_dims(face, axis=(0, -1))
        pred = st.session_state.emotion_model.predict(face)
        detected_emotion = emotion_labels[int(np.argmax(pred))].lower()
        draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=3)
        draw.text((x, y-15), detected_emotion.capitalize(), fill="red")
    st.success(f"Detected emotion: {detected_emotion}")
    st.image(rgb_img, use_container_width=True)

# --- Initialize LLM and Agent ---
if "chat_model" not in st.session_state:
    st.session_state.chat_model = lms.llm("llama-3.2-1b-instruct")

if "agent" not in st.session_state:
    st.session_state.agent = EmotionAgent(
        llm_model=st.session_state.chat_model,
        emotion=detected_emotion
    )
else:
    st.session_state.agent.update_emotion(detected_emotion)

# Display conversation history
for msg in st.session_state.agent.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Chat input
if user_input := st.chat_input("What's on your mind?"):
    st.chat_message("user").markdown(user_input)
    st.session_state.agent.add_user_input(user_input)
    response = st.session_state.agent.get_response()
    with st.chat_message("assistant"):
        st.markdown(response)
