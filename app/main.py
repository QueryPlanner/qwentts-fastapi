"""
Qwen3-TTS FastAPI Server

A FastAPI server for Qwen3-TTS 0.6B model with:
- CustomVoice: Predefined speakers with style instructions
- VoiceClone: Clone voice from reference audio
"""

import io
import time
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from contextlib import asynccontextmanager

# Model will be loaded at startup
model = None


class Language(str, Enum):
    AUTO = "Auto"
    CHINESE = "Chinese"
    ENGLISH = "English"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    GERMAN = "German"
    FRENCH = "French"
    RUSSIAN = "Russian"
    PORTUGUESE = "Portuguese"
    SPANISH = "Spanish"
    ITALIAN = "Italian"


class Speaker(str, Enum):
    VIVIAN = "Vivian"
    RYAN = "Ryan"
    AIDEN = "Aiden"
    JOSH = "Josh"
    DREW = "Drew"
    MIA = "Mia"
    EMMA = "Emma"
    AMY = "Amy"
    BRIAN = "Brian"


class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"


class TTSRequest(BaseModel):
    """TTS generation request."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    language: Language = Field(Language.AUTO, description="Output language")
    speaker: Speaker = Field(Speaker.RYAN, description="Voice speaker")
    instructions: Optional[str] = Field("", description="Style/emotion instructions")
    audio_format: AudioFormat = Field(AudioFormat.WAV, description="Output audio format")


class VoiceCloneRequest(BaseModel):
    """Voice clone TTS request."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    language: Language = Field(Language.AUTO, description="Output language")
    ref_text: Optional[str] = Field("", description="Reference audio transcript")


class TTSResponse(BaseModel):
    """TTS generation response with stats."""
    generation_time: float
    audio_duration: float
    real_time_factor: float
    characters: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    global model

    print("=" * 60)
    print("Qwen3-TTS FastAPI Server")
    print("=" * 60)

    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda:0"
        print("Device: CUDA GPU")
    else:
        device = "cpu"
        print("Device: CPU")

    # Load model
    print("\nLoading Qwen3-TTS 0.6B model...")
    from qwen_tts import Qwen3TTSModel

    load_start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        device_map=device,
        dtype=torch.float32,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f} seconds")
    print("=" * 60)
    print("\nServer ready!")

    yield

    # Cleanup
    print("\nShutting down...")
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="Qwen3-TTS API",
    description="Text-to-Speech API using Qwen3-TTS 0.6B model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """API root with available endpoints."""
    return {
        "name": "Qwen3-TTS API",
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "endpoints": {
            "/tts": "Generate speech with predefined speaker",
            "/tts/clone": "Generate speech with cloned voice",
            "/voices": "List available speakers",
            "/languages": "List supported languages",
            "/health": "Health check",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/voices")
async def list_voices():
    """List available speakers."""
    return {
        "voices": [s.value for s in Speaker],
        "default": Speaker.RYAN.value,
    }


@app.get("/languages")
async def list_languages():
    """List supported languages."""
    return {
        "languages": [l.value for l in Language],
        "default": Language.AUTO.value,
    }


@app.post("/tts", response_class=StreamingResponse)
async def generate_speech(request: TTSRequest):
    """
    Generate speech from text using predefined speaker.

    - **text**: Text to synthesize (required)
    - **language**: Output language (default: Auto)
    - **speaker**: Voice speaker (default: Ryan)
    - **instructions**: Style/emotion instructions (optional)
    - **audio_format**: Output format (default: wav)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        gen_start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=request.text,
            language=request.language.value,
            speaker=request.speaker.value,
            instruct=request.instructions or "",
        )
        gen_time = time.time() - gen_start

        # Calculate stats
        audio_duration = len(wavs[0]) / sr

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format=request.audio_format.value.upper())
        buffer.seek(0)

        # Set content type
        content_types = {
            AudioFormat.WAV: "audio/wav",
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.FLAC: "audio/flac",
        }

        headers = {
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Audio-Duration": f"{audio_duration:.2f}",
            "X-Real-Time-Factor": f"{audio_duration/gen_time:.2f}",
            "X-Characters": str(len(request.text)),
        }

        return StreamingResponse(
            buffer,
            media_type=content_types[request.audio_format],
            headers=headers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/tts/batch", response_class=StreamingResponse)
async def generate_speech_batch(
    texts: List[str] = Form(...),
    languages: Optional[str] = Form("Auto"),
    speakers: Optional[str] = Form("Ryan"),
    instructions: Optional[str] = Form(""),
    audio_format: AudioFormat = Form(AudioFormat.WAV),
):
    """
    Batch generate speech from multiple texts.

    - **texts**: List of texts to synthesize (form field, comma-separated for multiple)
    - **languages**: Comma-separated languages or single language for all
    - **speakers**: Comma-separated speakers or single speaker for all
    - **instructions**: Comma-separated instructions or single for all
    - **audio_format**: Output format (default: wav)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse inputs
        lang_list = [l.strip() for l in languages.split(",")] if languages else ["Auto"] * len(texts)
        speaker_list = [s.strip() for s in speakers.split(",")] if speakers else ["Ryan"] * len(texts)
        instruct_list = [i.strip() for i in instructions.split("|")] if instructions else [""] * len(texts)

        # Expand single values to match text count
        if len(lang_list) == 1:
            lang_list = lang_list * len(texts)
        if len(speaker_list) == 1:
            speaker_list = speaker_list * len(texts)
        if len(instruct_list) == 1:
            instruct_list = instruct_list * len(texts)

        gen_start = time.time()
        wavs, sr = model.generate_custom_voice(
            text=texts,
            language=lang_list,
            speaker=speaker_list,
            instruct=instruct_list,
        )
        gen_time = time.time() - gen_start

        # Concatenate all audio
        import numpy as np
        combined = np.concatenate(wavs)
        audio_duration = len(combined) / sr

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, combined, sr, format=audio_format.value.upper())
        buffer.seek(0)

        content_types = {
            AudioFormat.WAV: "audio/wav",
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.FLAC: "audio/flac",
        }

        headers = {
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Audio-Duration": f"{audio_duration:.2f}",
            "X-Real-Time-Factor": f"{audio_duration/gen_time:.2f}",
            "X-Characters": str(sum(len(t) for t in texts)),
        }

        return StreamingResponse(
            buffer,
            media_type=content_types[audio_format],
            headers=headers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/tts/clone", response_class=StreamingResponse)
async def generate_speech_clone(
    text: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(""),
    language: Language = Form(Language.AUTO),
    audio_format: AudioFormat = Form(AudioFormat.WAV),
):
    """
    Generate speech with cloned voice from reference audio.

    - **text**: Text to synthesize
    - **ref_audio**: Reference audio file for voice cloning
    - **ref_text**: Transcript of reference audio (improves quality)
    - **language**: Output language (default: Auto)
    - **audio_format**: Output format (default: wav)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read reference audio
        ref_audio_bytes = await ref_audio.read()

        # Save temporarily for processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(ref_audio_bytes)
            tmp_path = tmp.name

        gen_start = time.time()
        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language.value,
            ref_audio=tmp_path,
            ref_text=ref_text or None,
        )
        gen_time = time.time() - gen_start

        # Cleanup temp file
        import os
        os.unlink(tmp_path)

        audio_duration = len(wavs[0]) / sr

        # Convert to bytes
        buffer = io.BytesIO()
        sf.write(buffer, wavs[0], sr, format=audio_format.value.upper())
        buffer.seek(0)

        content_types = {
            AudioFormat.WAV: "audio/wav",
            AudioFormat.MP3: "audio/mpeg",
            AudioFormat.FLAC: "audio/flac",
        }

        headers = {
            "X-Generation-Time": f"{gen_time:.2f}",
            "X-Audio-Duration": f"{audio_duration:.2f}",
            "X-Real-Time-Factor": f"{audio_duration/gen_time:.2f}",
            "X-Characters": str(len(text)),
        }

        return StreamingResponse(
            buffer,
            media_type=content_types[audio_format],
            headers=headers,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
