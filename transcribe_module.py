# transcribe_module.py
import os
import whisper
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# ---------- Load models ONCE at import time ----------
print("Loading Whisper model (small) ...")
WHISPER_MODEL = whisper.load_model("small")  # consider "base" for speed or "small" for accuracy

# Lazy-load Nepali HF model only if needed (we'll initialize to None)
HF_NEPALI_MODEL = None
HF_NEPALI_PROCESSOR = None
HF_NEPALI_NAME = "iamTangsang/Wav2Vec2_XLS-R-300m_Nepali_ASR"

def _ensure_nepali_model_loaded():
    global HF_NEPALI_MODEL, HF_NEPALI_PROCESSOR
    if HF_NEPALI_MODEL is None or HF_NEPALI_PROCESSOR is None:
        print("Loading Nepali HuggingFace model ...")
        HF_NEPALI_PROCESSOR = Wav2Vec2Processor.from_pretrained(HF_NEPALI_NAME)
        HF_NEPALI_MODEL = Wav2Vec2ForCTC.from_pretrained(HF_NEPALI_NAME)

def transcribe_audio(wav_path: str) -> (str, str):
    """
    Returns: (transcription_text, detected_language_code)
    detected_language_code is e.g. "en" or "ne" (or whisper's reported language)
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Use Whisper to detect language first
    detected = WHISPER_MODEL.transcribe(wav_path, task="transcribe")
    detected_lang = detected.get("language", "unknown")
    # Whisper may return language codes like 'ne' or 'english' depending on versions;
    # normalize to lower-case
    detected_lang = str(detected_lang).lower()

    # If English -> use Whisper transcription (fast path)
    if detected_lang.startswith("en"):
        res = WHISPER_MODEL.transcribe(wav_path, language="en")
        text = res.get("text", "").strip()
        return text, "en"

    # Otherwise, use the Nepali HF model (or fallback to Whisper's text if needed)
    try:
        _ensure_nepali_model_loaded()
        speech_array, sampling_rate = torchaudio.load(wav_path)
        # Convert to mono
        if speech_array.shape[0] > 1:
            speech_array = torch.mean(speech_array, dim=0, keepdim=True)
        speech = speech_array.squeeze()

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            speech = resampler(speech)

        inputs = HF_NEPALI_PROCESSOR(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = HF_NEPALI_MODEL(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = HF_NEPALI_PROCESSOR.batch_decode(predicted_ids)[0].strip()
        return transcription, "ne"
    except Exception as e:
        # Fallback: return Whisper text if HF model fails
        fallback = detected.get("text", "").strip()
        return fallback, detected_lang
