import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from CNN.CNN_model import model
import lmstudio as lms

st.title("Emotional Assistance")

# --- Emotion Detection Setup ---
# Load emotion model weights once
if "emotion_model" not in st.session_state:
    model.load_weights(
        "/Users/bernardoquindimil/Code/Berniquindimil/Proyect/CNN/model_weights.weights.h5"
    )  # Update path if needed
    st.session_state.emotion_model = model

# Load OpenCV's pre-trained face detector once
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Define emotion labels
emotion_labels = [
    'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'
]

# Camera input widget
img_file = st.camera_input("Take a photo to analyze your emotion")

# Default emotion and annotated image
detected_emotion = "Neutral"
annotated_image = None

if img_file is not None:
    # Open original image as PIL and prepare for drawing
    pil_img = Image.open(img_file)
    rgb_img = pil_img.convert("RGB")
    draw = ImageDraw.Draw(rgb_img)

    # Convert to grayscale for face detection
    gray = np.array(pil_img.convert("L"))

    # Detect face in the image
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) > 0:
        # Use the first detected face
        x, y, w, h = faces[0]
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (48, 48))  # FER-2013 input size
        face = face / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # Shape: (1, 48, 48, 1)

        # Predict emotion
        prediction = st.session_state.emotion_model.predict(face)
        detected_emotion = emotion_labels[np.argmax(prediction)]

        # Draw bounding box and label on RGB image
        draw.rectangle(
            [(x, y), (x + w, y + h)], outline="red", width=3
        )
        draw.text((x, y - 15), detected_emotion, fill="red")

    # Display detected emotion and annotated photo
    st.success(f"Detected emotion: {detected_emotion}")
    annotated_image = rgb_img
    st.image(annotated_image, use_container_width=True)

# --- Chatbot Setup ---
# Load LLM once
if "chat_model" not in st.session_state:
    st.session_state.chat_model = lms.llm("llama-3.2-1b-instruct")

# System prompt for therapist
SYSTEM_PROMPT = (
    "You are a professional therapist specializing in mental health. "
    "You listen with empathy, validate emotions, and offer guidance without judgment. "
    "You respond in a kind, clear, and focused manner focused on emotional well-being."
)

# Session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
if "full_context" not in st.session_state:
    st.session_state.full_context = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What's on your mind?"):
    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.full_context.append({"role": "user", "content": prompt})

    # Build prompt text for LLM
    prompt_text = ""
    # Add system prompt
    for msg in st.session_state.full_context:
        if msg["role"] == "system":
            prompt_text += msg["content"] + "\n"

    # Insert detected emotion context
    prompt_text += f"(Detected emotion: {detected_emotion})\n"

    # Add conversation history
    for msg in st.session_state.full_context:
        if msg["role"] == "user":
            prompt_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            prompt_text += f"Therapist: {msg['content']}\n"

    prompt_text += "Therapist:"

    # Get LLM response
    response = st.session_state.chat_model.respond(prompt_text)

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)

    # Update session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.full_context.append({"role": "assistant", "content": response})
