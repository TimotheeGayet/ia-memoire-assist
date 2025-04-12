# === core/audio.py ===

from openai.helpers import LocalAudioPlayer
from openai import OpenAI, AsyncOpenAI
import collections
import webrtcvad
import tempfile
import pyaudio
import time
import wave
import os

CLIENT = OpenAI()
CLIENT_ASYNC = AsyncOpenAI()

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000  # cohérent avec webrtcvad
FRAME_DURATION_MS = 30  # durée d'une frame en ms
FRAME_SIZE = 480  # taille fixe requise par webrtcvad

class VoiceActivityRecorder:
    def __init__(self, aggressiveness=3, sample_rate=SAMPLE_RATE, frame_duration=FRAME_DURATION_MS):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = FRAME_SIZE  # utilisation de la taille fixe
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=sample_rate,
                                input=True,
                                frames_per_buffer=self.frame_size)

    def read_frame(self):
        return self.stream.read(self.frame_size)

    def is_speech(self, frame):
        return self.vad.is_speech(frame, self.sample_rate)

    def record_until_silence(self, silence_limit=1.0, max_duration=10.0):
        print("🎙️ En attente de la parole...")
        frames = []
        ring_buffer = collections.deque(maxlen=int(silence_limit * 1000 / self.frame_duration))
        start_time = time.time()
        speaking = False

        while True:
            frame = self.read_frame()
            is_speaking = self.is_speech(frame)

            if is_speaking:
                if not speaking:
                    print("🗣️ Début de la parole détecté.")
                    speaking = True
                ring_buffer.clear()
                frames.append(frame)
            elif speaking:
                ring_buffer.append(frame)
                if len(ring_buffer) == ring_buffer.maxlen:
                    print("🤐 Fin de la parole détectée.")
                    frames.extend(ring_buffer)
                    break

            if time.time() - start_time > max_duration:
                print("⏱️ Durée maximale atteinte.")
                break

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        return frames

def save_wav(frames, sample_rate, filename):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(frames, sample_rate=SAMPLE_RATE):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        save_wav(frames, sample_rate, temp_file.name)
        with open(temp_file.name, "rb") as audio_file:
            transcript = CLIENT.audio.transcriptions.create(model="whisper-1", file=audio_file)
    os.unlink(temp_file.name)
    return transcript.text

async def text_to_speech(text):
    async with CLIENT_ASYNC.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions="Parle en français avec une voix naturelle et chaleureuse. Adapte ton rythme et ton intonation pour une conversation fluide et agréable. Utilise des pauses naturelles et des variations de ton pour transmettre les émotions de manière authentique.",
        response_format="wav"
    ) as response:
        await LocalAudioPlayer().play(response)

def vad_record_and_transcribe():
    recorder = VoiceActivityRecorder()
    frames = recorder.record_until_silence()
    return transcribe_audio(frames, sample_rate=recorder.sample_rate)
