import streamlit as st
import whisper
import numpy as np
import soundfile as sf

st.title("Conversation Transcriber")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    model = whisper.load_model("base")  # Use 'medium' or 'large' for better accuracy
    result = model.transcribe("temp_audio.wav")
    
    transcript = result["text"]
    st.write("### Transcription:")
    st.write(transcript)
    
    with open("transcript.txt", "w") as f:
        f.write(transcript)
    
    st.success("Transcript saved as transcript.txt")
