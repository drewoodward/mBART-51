import io
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import torch
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

WHISPER_MODEL_ID = "drewoodward/whisper-small-spanglish"
WHISPER_BASE_ID = "openai/whisper-small"
MBART_MODEL_ID = "drewoodward/mBART-51"
TTS_VOICES = ["af_heart", "af_bella", "af_nicole", "am_adam", "am_michael", "bf_emma", "bm_george"]

st.set_page_config(page_title="mBART-51 Translator", layout="centered")


@st.cache_resource
def load_whisper_model():
    device = 0 if torch.cuda.is_available() else -1
    processor = WhisperProcessor.from_pretrained(WHISPER_BASE_ID)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_ID)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device=device,
        chunk_length_s=30,
    )


@st.cache_resource
def load_mbart():
    tokenizer = MBart50TokenizerFast.from_pretrained(MBART_MODEL_ID)
    model = MBartForConditionalGeneration.from_pretrained(MBART_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return tokenizer, model.to(device).eval(), device


@st.cache_resource
def load_tts():
    from kokoro import KPipeline
    return KPipeline(lang_code="a")


def transcribe(audio_path: str) -> str:
    asr = load_whisper_model()
    return asr(audio_path)["text"]


def translate(text: str) -> str:
    tokenizer, model, device = load_mbart()
    tokenizer.src_lang = "es_XX"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_length=512,
            num_beams=4,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def synthesize(text: str, voice: str) -> io.BytesIO:
    pipeline = load_tts()
    chunks = [audio for _, _, audio in pipeline(text, voice=voice, speed=1.0)]
    audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)
    return buf


# --- UI ---

st.title("mBART-51 Translator")
st.caption("Whisper Small (Spanglish-tuned) → mBART-51 → Kokoro TTS | Accent & code-switching aware")

input_method = st.radio("Audio input", ["Upload file", "Record"], horizontal=True)

audio_source = None
if input_method == "Record":
    audio_source = st.audio_input("Click to record")
else:
    audio_source = st.file_uploader(
        "Upload audio file", type=["mp3", "wav", "m4a", "ogg", "flac"]
    )

voice = st.selectbox("TTS voice", TTS_VOICES)

if audio_source:
    suffix = ".wav"
    if hasattr(audio_source, "name") and audio_source.name:
        suffix = Path(audio_source.name).suffix or ".wav"

    if hasattr(audio_source, "getvalue"):
        audio_bytes = audio_source.getvalue()
    else:
        if hasattr(audio_source, "seek"):
            audio_source.seek(0)
        audio_bytes = audio_source.read()

    audio_key = hash(audio_bytes)

    st.caption("Original audio")
    st.audio(audio_bytes)

    if st.session_state.get("audio_key") != audio_key:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            with st.spinner("Transcribing with Whisper Small..."):
                st.session_state.transcript = transcribe(tmp_path)
            st.session_state.audio_key = audio_key
        finally:
            os.unlink(tmp_path)

    st.subheader("Transcription")
    transcript = st.text_area(
        "Edit if needed:", value=st.session_state.transcript, height=100
    )

    if st.button("Translate & Synthesize", type="primary"):
        with st.spinner("Translating with mBART-51..."):
            translation = translate(transcript)

        st.subheader("Translation")
        st.write(translation)

        with st.spinner("Synthesizing with Kokoro TTS..."):
            audio_buf = synthesize(translation, voice)

        st.subheader("Translated Audio")
        st.audio(audio_buf, format="audio/wav")
        st.download_button("Download WAV", audio_buf, "translation.wav", "audio/wav")
