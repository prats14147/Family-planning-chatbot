# transcribe_audio.py
import whisper
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import os

def transcribe_audio(audio_path):
    try:
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return ""

        # --- Detect language using Whisper ---
        whisper_model = whisper.load_model("small")
        lang_detect = whisper_model.transcribe(audio_path, task="transcribe")
        detected_language = lang_detect.get("language", "unknown")

        # --- Transcription ---
        if detected_language in ["en", "english"]:
            result = whisper_model.transcribe(audio_path, language="en")
            transcription = result["text"].strip()
        else:
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

        print(f"📝 Transcription: {transcription}")
        return transcription

    except Exception as e:
        print("❌ Transcription Error:", e)
        return ""
