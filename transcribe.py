# transcribe_to_txt.py
import whisper
import datetime
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

def transcribe_and_save(audio_path="input.wav", output_file="transcription_text.txt"):
    try:
        # ------------------- CHECK AUDIO FILE -------------------
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return

        # ------------------- LANGUAGE DETECTION -------------------
        print("🧠 Detecting language using Whisper...")
        whisper_model = whisper.load_model("small")
        lang_detect = whisper_model.transcribe(audio_path, task="transcribe")
        detected_language = lang_detect.get("language", "unknown")
        print(f"🌐 Detected Language: {detected_language}")

        # ------------------- TRANSCRIPTION -------------------
        if detected_language in ["en", "english"]:
            print("🔠 Using Whisper for English transcription...")
            result = whisper_model.transcribe(audio_path, language="en")
            transcription = result["text"].strip()
            language = "English"

        else:
            print("🇳🇵 Using Hugging Face model for Nepali transcription...")
            model_name = "iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR"
            processor = Wav2Vec2Processor.from_pretrained(model_name)
            model = Wav2Vec2ForCTC.from_pretrained(model_name)

            speech_array, sampling_rate = torchaudio.load(audio_path)
            if speech_array.shape[0] > 1:
                speech_array = torch.mean(speech_array, dim=0, keepdim=True)
            speech = speech_array.squeeze()

            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
                speech = resampler(speech)

            inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            language = "Nepali"

        print(f"📝 Transcription ({language}): {transcription}")

        # ------------------- APPEND TO TEXT FILE -------------------
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(output_file, "a", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Language: {language}\n")
            f.write(f"Transcription:\n{transcription}\n\n")

        print(f"✅ Transcription appended to '{output_file}'")

    except Exception as e:
        print("❌ ERROR:", str(e))


if __name__ == "__main__":
    transcribe_and_save("input.wav", "transcription_text.txt")
