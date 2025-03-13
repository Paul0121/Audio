import pyaudio
import wave
import whisper
import numpy as np
from scipy.spatial.distance import cdist

# Recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 30  # Adjust as needed
OUTPUT_FILE = "conversation.wav"

def record_audio():
    print("Recording...")
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(OUTPUT_FILE, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

def transcribe_audio():
    print("Transcribing...")
    model = whisper.load_model("base")  # Use 'medium' or 'large' for better accuracy
    result = model.transcribe(OUTPUT_FILE, word_timestamps=True)
    return result['segments']

def speaker_diarization(segments):
    embeddings = np.array([seg['embedding'] for seg in segments if 'embedding' in seg])
    if embeddings.shape[0] < 2:
        print("Not enough data for speaker separation.")
        return segments

    centroids = np.array([embeddings[0], embeddings[-1]])  # Assume two different speakers at start & end
    distances = cdist(embeddings, centroids, metric='cosine')
    labels = np.argmin(distances, axis=1)  # Assign each segment to closest centroid

    for i, seg in enumerate(segments):
        seg['speaker'] = f"Caller {labels[i] + 1}"
    return segments

def save_transcript(segments):
    with open("transcript.txt", "w") as f:
        for seg in segments:
            f.write(f"{seg['speaker']}: {seg['text']}\n")
    print("Transcript saved as transcript.txt")

if __name__ == "__main__":
    record_audio()
    segments = transcribe_audio()
    segments = speaker_diarization(segments)
    save_transcript(segments)
