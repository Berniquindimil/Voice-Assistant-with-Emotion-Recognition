# STT_TTS.py
import whisper
import streamlit as st

import sounddevice as sd
import tempfile
import os
from gtts import gTTS
from pydub import AudioSegment

import numpy as np
import librosa

@st.cache_resource
def get_whisper_model():
    """Cache and return the Whisper model instance."""
    return whisper.load_model("base.en")


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

# gTTS + pydub “same voice, different tone”
EMOTION_SPEED = {
    "happy":    1.1,
    "sad":      0.85,
    "angry":    0.90,
    "fear":     0.85,
    "surprise": 1.08,
    "neutral":  1.00
}

def text_to_speech(text, emotion: str):
    """
    1) Coerce text to native str if needed.
    2) Generate MP3 via gTTS into a temp file.
    3) Load audio with pydub.
    4) Adjust speed/pitch by frame rate.
    5) Play with sounddevice.
    """
    # Ensure we have a Python string
    if not isinstance(text, str):
        text = getattr(text, 'text', None) or str(text)

    # Write to a temporary file to avoid write_to_fp issues
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
        tts = gTTS(text=text, lang="en")
        tts.save(tf.name)
        temp_path = tf.name

    try:
        seg = AudioSegment.from_file(temp_path, format="mp3")
        speed = EMOTION_SPEED.get(emotion, 1.0)
        new_rate = int(seg.frame_rate * speed)
        seg2 = seg._spawn(seg.raw_data, overrides={"frame_rate": new_rate})
        seg2 = seg2.set_frame_rate(44100)

        samples = np.array(seg2.get_array_of_samples(), dtype="float32")
        samples /= np.iinfo(seg2.array_type).max
        sd.play(samples, samplerate=seg2.frame_rate)
        sd.wait()
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
