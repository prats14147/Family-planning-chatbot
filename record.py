# record_audio.py
import sounddevice as sd
import soundfile as sf
import datetime

def record_audio(filename="input.wav", duration=10, samplerate=16000):
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
    sd.wait()
    sf.write(filename, audio, samplerate)
    print(f"✅ Recording complete: saved as {filename}")

if __name__ == "__main__":
    duration = 10  # seconds
    filename = "input.wav"
    record_audio(filename, duration)
