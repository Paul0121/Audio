import sounddevice as sd
import numpy as np
import whisper
import wave
import scipy.io.wavfile as wav

# Recording parameters
SAMPLE_RATE = 44100
DURATION = 30  # Adjust as needed
OUTPUT_FILE = "conversation.wav"

def record_audio():
    print("Recording...")
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype=np.int16)
    sd.wait()
    wav.write(OUTPUT_FILE, SAMPLE_RATE, audio_data)
    print("Recording complete.")

def transcribe_audio():
    print("Transcribing...")
    model = whisper.load_model("base")  # Use 'medium' or 'large' for better accuracy
    result = model.transcribe(OUTPUT_FILE)
    return result['text']

def save_transcript(transcript):
    with open("transcript.txt", "w") as f:
        f.write(transcript)
    print("Transcript saved as transcript.txt")

if __name__ == "__main__":
    record_audio()
    transcript = transcribe_audio()
    save_transcript(transcript)
