import base64
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Union

from fastapi import UploadFile
from dotenv import load_dotenv
import openai

load_dotenv()



# Speech-to-Text (OpenAI Whisper)

async def transcribe_audio(file: UploadFile) -> str:
    """Transcribe an UploadFile with Whisper and return plain text."""
    # Persist to a temporary file – Whisper requires a real file handle
    suffix = Path(file.filename).suffix or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Reset file pointer (in case caller wants to re-use it later)
    file.file.seek(0)

    # Call Whisper
    try:
        with open(tmp_path, "rb") as audio_in:
            transcription = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_in,
                response_format="text",
                temperature=0.0,
            )
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

    # The SDK returns a str for response_format="text"
    text = transcription.strip()
    print(f"[Whisper] Transcript (first 120 chars): {text[:120]}{'…' if len(text)>120 else ''}")
    return text



# Text-to-Speech (OpenAI TTS)

async def synthesize_speech(text: str, voice: str = "alloy") -> str:
    """Convert `text` to speech (mp3) and return a base64-encoded string."""
    if not text:
        return ""

  
    print(f"[TTS] Synthesizing {len(text)} characters with voice='{voice}' using model 'tts-1'")

    tts_response = openai.audio.speech.create(
        model="tts-1",           
        voice=voice,
        input=text,
        response_format="mp3",   
    )

    if hasattr(tts_response, "audio"):
        audio_bytes: Union[bytes, bytearray] = tts_response.audio.data  
    else:
        audio_bytes = tts_response.content

    b64_audio = base64.b64encode(audio_bytes).decode()
  
    return f"data:audio/mp3;base64,{b64_audio}" 