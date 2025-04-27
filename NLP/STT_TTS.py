# STT_TTS.py
import whisper
import pyttsx3
import streamlit as st

import numpy as np
import librosa

@st.cache_resource
def get_whisper_model():
    """Cache and return the Whisper model instance."""
    return whisper.load_model("base")


def speech_to_text(audio_file_path: str) -> str:
    """Transcribe the given WAV file to text using Whisper."""
    model = get_whisper_model()
    result = model.transcribe(audio_file_path)
    return result.get("text", "").strip()

def extract_features(data, sampling_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sampling_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

@st.cache_resource
def get_tts_engine():
    """Cache and return the text-to-speech engine."""
    engine = pyttsx3.init()
    return engine


def text_to_speech(text: str, emotion: str):
    """Convert text to speech with voice adjustments based on emotion."""
    engine = get_tts_engine()
    voices = engine.getProperty('voices')
    # Adjust rate, volume, and voice based on detected emotion
    if emotion == "happy":
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.setProperty('voice', voices[1].id)
    elif emotion == "sad":
        engine.setProperty('rate', 120)
        engine.setProperty('volume', 0.8)
        engine.setProperty('voice', voices[0].id)
    else:
        engine.setProperty('rate', 130)
        engine.setProperty('volume', 0.9)
    engine.say(text)
    engine.runAndWait()
