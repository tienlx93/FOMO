from transformers import pipeline
import soundfile as sf
import time
import streamlit as st
from langdetect import detect

LANG_MAP = {
    "en": "eng", "vi": "vie", "es": "spa", "fr": "fra", "de": "deu",
    "zh-cn": "zho", "zh": "zho", "pt": "por", "it": "ita", "id": "ind",
    "ja": "jpn", "ko": "kor", "ar": "arb"
}

def detect_tts_language(text):
    """Detect language and map to MMS-TTS code"""
    try:
        lang_code = detect(text)
        return LANG_MAP.get(lang_code, "eng")
    except Exception:
        return "eng"

def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper

@st.cache_resource
def load_tts(lang="eng"):
    return pipeline("text-to-speech", model=f"facebook/mms-tts-{lang}")

@timeit
def speak(text, lang=None, save_path=None):
    if not lang:
        lang = detect_tts_language(text);
    print (lang);
    tts = load_tts(lang)
    speech = tts(text)
    audio = speech["audio"].squeeze()
    if save_path:
        sf.write(save_path, audio, samplerate=speech["sampling_rate"])
        return
    return audio, speech["sampling_rate"]

# Example use:
if __name__ == "__main__":
    load_tts(lang="eng")
    speak("""1️⃣ Using transformers (automatic download & caching)

If you just do:

from transformers import pipeline

tts = pipeline("text-to-speech", model="facebook/mms-tts-eng")


Hugging Face will automatically:

download the model once,

store it under your local cache (usually ~/.cache/huggingface/hub),

and reuse it later offline.

You can check the actual path via:

tts.model.config._name_or_path""", lang="eng")
