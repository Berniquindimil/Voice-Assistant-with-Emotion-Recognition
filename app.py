# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import joblib
from tensorflow.keras.models import load_model
import lmstudio as lms

# Set page config first before any other Streamlit commands
st.set_page_config(page_title="Emotional Assistant", layout="wide")

from Computer_vision.CNN.CNN_model import model
from NLP.STT_TTS import speech_to_text, text_to_speech, extract_features
from Intelligent_system.emotion_agent import EmotionAgent
from Intelligent_system.mental_health_questionnaire import MentalHealthQuestionnaire

# Cache the SER model with its scaler/encoder
@st.cache_resource
def load_ser_objects():
    ser_model = load_model('NLP/SER/SER_model.h5')
    scaler    = joblib.load('NLP/SER/scaler.pkl')
    encoder   = joblib.load('NLP/SER/encoder.pkl')
    return ser_model, scaler, encoder

ser_model, scaler, encoder = load_ser_objects()

model.load_weights('Computer_vision/CNN/model_weights.weights.h5')
fer_model = model

# Retrieve the emotion labels in the training order
class_list = encoder.categories_[0].tolist()

# Initialize face detector once in session state
if "face_cascade" not in st.session_state:
    st.session_state.face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

# Initialize chat model and history if not already present
if "chat_model" not in st.session_state:
    st.session_state.chat_model = lms.llm("llama-3.2-1b-instruct")

# Initialize or get the questionnaire module
if "questionnaire" not in st.session_state:
    st.session_state.questionnaire = MentalHealthQuestionnaire()

# Initialize interface state
if "show_section" not in st.session_state:
    st.session_state.show_section = "emotion_detection" # Default view

st.title("Emotional Assistance")

# Create tabs for different sections
tabs = st.tabs(["Emotion Detection", "Mental Health Assessment", "Conversation"])

# Tab 1: Emotion Detection
with tabs[0]:
    st.header("Emotion Detection")
    st.write("Let's detect your emotional state through your facial expression and voice.")
    
    # Predefined emotion labels for facial recognition
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # Column layout for face and voice detection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Facial Emotion Detection")
        # Capture user photo for facial emotion detection
        img_file = st.camera_input("Capture a photo to analyze your facial emotion")
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
                pred = fer_model.predict(face)
                detected_emotion = emotion_labels[int(np.argmax(pred))]
                draw.rectangle([(x, y), (x+w, y+h)], outline="red", width=3)
                draw.text((x, y-15), detected_emotion.capitalize(), fill="red")
            else:
                detected_emotion = 'neutral'
            st.success(f"Detected facial emotion: {detected_emotion}")
            st.image(rgb_img, use_container_width=True)
            
            # Save the detected emotion to session state
            st.session_state.face_emotion = detected_emotion
    
    with col2:
        st.subheader("Voice Emotion Detection")
        if st.button("Record Voice for Emotion Analysis"):
            # Record up to a maximum duration
            max_duration = 5  # seconds
            sr = 22050
            st.info(f"Recording audio for up to {max_duration} seconds...")
            audio = sd.rec(int(sr * max_duration), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            
            # Save recorded audio to WAV file
            wav_path = "emotion_audio.wav"
            write(wav_path, sr, (audio * 32767).astype('int16'))
            st.success("Recording complete!")
            
            # Segment audio and predict emotion for each segment
            sig = audio.flatten()
            segment_length = int(2.5 * sr)
            probabilities = []
            num_segments = int(np.ceil(len(sig) / segment_length))
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                chunk = sig[start:end]
                if len(chunk) < segment_length:
                    chunk = np.pad(chunk, (0, segment_length - len(chunk)))
                feats = extract_features(chunk, sr)
                feats_scaled = scaler.transform([feats])
                input_data = np.expand_dims(feats_scaled, axis=2)
                probabilities.append(ser_model.predict(input_data)[0])
            avg_probs = np.mean(probabilities, axis=0)
            voice_emotion = class_list[int(np.argmax(avg_probs))]
            st.success(f"Detected voice emotion: **{voice_emotion}**")
            
            # Save the detected emotion to session state
            st.session_state.speech_emotion = voice_emotion
    
    # Update or initialize the agent with detected emotions
    if "face_emotion" in st.session_state and "speech_emotion" in st.session_state:
        if "agent" not in st.session_state:
            st.session_state.agent = EmotionAgent(
                llm_model=st.session_state.chat_model,
                face_emotion=st.session_state.face_emotion,
                speech_emotion=st.session_state.speech_emotion
            )
        else:
            st.session_state.agent.update_emotions(
                face_emotion=st.session_state.face_emotion,
                speech_emotion=st.session_state.speech_emotion
            )
        
        st.success("Emotion detection complete. You can now proceed to the assessment or conversation tabs.")

# Tab 2: Mental Health Assessment
with tabs[1]:
    st.header("Mental Health Assessment")
    st.write("""Complete these standardized questionnaires to help me better understand your current mental state. 
             This will allow me to provide more personalized support and guidance during our conversation.""")
    
    # Render the questionnaire using the MentalHealthQuestionnaire class
    st.session_state.questionnaire.render_questionnaire_section()

# Tab 3: Conversation
with tabs[2]:
    st.header("Therapeutic Conversation")
    
    # Check if agent is initialized
    if "agent" not in st.session_state:
        st.warning("Please complete the emotion detection first in the Emotion Detection tab.")
    else:
        # Display current emotional state
        current_emotion = st.session_state.agent.final_emotion
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Current detected emotional state: **{current_emotion}**")
        
        # Display assessment insights if available
        if st.session_state.agent.assessment_results:
            with col2:
                with st.expander("View Assessment Insights"):
                    insights = st.session_state.agent.get_therapeutic_insights()
                    if insights:
                        st.write(insights)
        
        # Chat interface
        st.subheader("Chat with your Therapeutic Assistant")
        
        # Display conversation history
        for msg in st.session_state.agent.messages:
            role = msg['role']
            content = msg['content']
            
            # Special handling for system messages (make them info blocks)
            if role == 'system':
                st.info(content)
            else:
                with st.chat_message(role):
                    st.markdown(content)
        
        # Text-based chat input
        if user_input := st.chat_input("What's on your mind?"):
            st.chat_message("user").markdown(user_input)
            st.session_state.agent.add_user_input(user_input)
            response = st.session_state.agent.get_response()
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Convert assistant's response to speech
            text_to_speech(response, current_emotion)
        
        # Voice-based interaction
        st.subheader("Or speak to your therapist")
        if st.button("Record Voice Message", key="voice_msg"):
            # Record up to a maximum duration
            max_duration = 10  # seconds
            sr = 22050
            st.info(f"Recording audio for up to {max_duration} seconds...")
            audio = sd.rec(int(sr * max_duration), samplerate=sr, channels=1, dtype='float32')
            sd.wait()
            
            # Save recorded audio to WAV file
            wav_path = "user_prompt.wav"
            write(wav_path, sr, (audio * 32767).astype('int16'))
            st.success("Recording complete!")
        
            # Transcribe audio to text
            transcript = speech_to_text(wav_path)
            st.chat_message("user").markdown(transcript)
        
            # Segment audio and predict emotion for each segment
            sig = audio.flatten()
            segment_length = int(2.5 * sr)
            probabilities = []
            num_segments = int(np.ceil(len(sig) / segment_length))
            for i in range(num_segments):
                start = i * segment_length
                end = start + segment_length
                chunk = sig[start:end]
                if len(chunk) < segment_length:
                    chunk = np.pad(chunk, (0, segment_length - len(chunk)))
                feats = extract_features(chunk, sr)
                feats_scaled = scaler.transform([feats])
                input_data = np.expand_dims(feats_scaled, axis=2)
                probabilities.append(ser_model.predict(input_data)[0])
            avg_probs = np.mean(probabilities, axis=0)
            voice_emotion = class_list[int(np.argmax(avg_probs))]
            
            # Update the session speech emotion
            st.session_state.speech_emotion = voice_emotion
            # Update agent emotions
            st.session_state.agent.update_emotions(
                face_emotion=st.session_state.face_emotion,
                speech_emotion=st.session_state.speech_emotion
            )
            # Add transcript
            st.session_state.agent.add_user_input(transcript)
        
            # Generate response
            response = st.session_state.agent.get_response()
            with st.chat_message("assistant"):
                st.markdown(response)
        
            current_emotion = st.session_state.agent.decide_emotion()
        
            # Convert assistant's response to speech
            text_to_speech(response, current_emotion)