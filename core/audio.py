# === core/audio.py ===
import pyaudio
import wave
import tempfile
from openai import OpenAI, AsyncOpenAI
from openai.helpers import LocalAudioPlayer
import os

CLIENT = OpenAI()
CLIENT_ASYNC = AsyncOpenAI()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def record_audio(duration=5):
    print("\nEnregistrement en cours... (parlez maintenant)")
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(int(RATE / CHUNK * duration))]
    print("Enregistrement terminé.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return frames, RATE

def save_wav(frames, sample_rate, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(frames, sample_rate):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        save_wav(frames, sample_rate, temp_file.name)
        with open(temp_file.name, "rb") as audio_file:
            transcript = CLIENT.audio.transcriptions.create(model="whisper-1", file=audio_file)
    os.unlink(temp_file.name)
    return transcript.text

async def text_to_speech(text):
    async with CLIENT_ASYNC.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="ash",
        input=text,
        instructions="Serviable et sympathique",
        response_format="wav"
    ) as response:
        await LocalAudioPlayer().play(response)

