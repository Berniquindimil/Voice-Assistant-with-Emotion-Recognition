import whisper
import pyttsx3
import streamlit as st
import speech_recognition as sr
import tempfile

def recognize_audio_from_file(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError:
            return "Speech recognition service error."

def speech_to_text():
    model = whisper.load_model("base")  # Load the Whisper model
    st.write("Listening...")
    audio = st.audio(st.audio_file, format="audio/wav")
    result = model.transcribe(audio)
    user_input = result["text"]
    st.write(f"You said: {user_input}")
    return user_input

def text_to_speech(text, emotion):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # Change tone or voice based on detected emotion
    if emotion == "happy":
        engine.setProperty('rate', 150)  # Faster speech
        engine.setProperty('volume', 1)  # Full volume
        engine.setProperty('voice', voices[1].id)  # Female voice
    elif emotion == "sad":
        engine.setProperty('rate', 120)  # Slower speech
        engine.setProperty('volume', 0.8)  # Lower volume
        engine.setProperty('voice', voices[0].id)  # Male voice
    else:
        engine.setProperty('rate', 130)
        engine.setProperty('volume', 0.9)
    
    engine.say(text)
    engine.runAndWait()
