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
import subprocess
import re
import numpy as np
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


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    audio_format: AudioFormat,
) -> io.BytesIO:
    """
    Encode audio array to bytes in the specified format.

    Args:
        audio: NumPy array of audio samples
        sample_rate: Sample rate in Hz
        audio_format: Output format (WAV, MP3, or FLAC)

    Returns:
        BytesIO buffer containing encoded audio

    Note:
        MP3 encoding uses ffmpeg (must be installed).
        WAV and FLAC use soundfile directly.
    """
    buffer = io.BytesIO()

    if audio_format == AudioFormat.MP3:
        # MP3 requires ffmpeg transcoding
        # First write to WAV, then convert to MP3
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sample_rate, format='WAV')
        wav_buffer.seek(0)

        # Use ffmpeg to convert WAV to MP3
        process = subprocess.run(
            [
                'ffmpeg', '-y',
                '-i', 'pipe:0',
                '-codec:a', 'libmp3lame',
                '-qscale:a', '2',  # ~190kbps, good quality for speech
                '-f', 'mp3',
                'pipe:1'
            ],
            input=wav_buffer.read(),
            capture_output=True,
        )

        if process.returncode != 0:
            raise RuntimeError(f"FFmpeg MP3 encoding failed: {process.stderr.decode()}")

        buffer.write(process.stdout)
        buffer.seek(0)

    else:
        # WAV and FLAC are natively supported by soundfile
        sf.write(buffer, audio, sample_rate, format=audio_format.value.upper())
        buffer.seek(0)

    return buffer


def smart_chunk_text(
    text: str,
    max_chars: int = 500,
    min_sentences: int = 2,
    max_sentences: int = 3,
) -> List[str]:
    """
    Split text into chunks respecting natural boundaries.

    Groups 2-3 sentences together for optimal:
    - Memory usage (prevents OOM on long texts)
    - Audio quality (better prosody on shorter segments)
    - Natural boundaries (respects paragraphs and sentences)

    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk (default 500)
        min_sentences: Minimum sentences per chunk (default 2)
        max_sentences: Maximum sentences per chunk (default 3)

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # If text is small enough, return as-is
    text_stripped = text.strip()
    if len(text_stripped) <= max_chars:
        return [text_stripped]

    # Split into paragraphs first (prefer paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = []
    current_sentences = 0
    current_length = 0

    for para in paragraphs:
        # Normalize whitespace within each paragraph (preserves paragraph structure)
        para = ' '.join(para.split())
        if not para:
            continue

        # Split paragraph into sentences
        # Regex handles: . ! ? followed by space or end, respects quotes
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z"\'\u4e00-\u9fff])|(?<=[.!?])$',
            para
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        for sentence in sentences:
            sent_len = len(sentence)

            # Fallback for sentences longer than max_chars to prevent OOM
            if sent_len > max_chars:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_sentences = 0
                    current_length = 0

                # Split the long sentence into max_chars chunks
                for i in range(0, sent_len, max_chars):
                    chunks.append(sentence[i:i+max_chars])
                continue

            # Check if adding this sentence would exceed limits
            would_exceed_chars = current_length + sent_len + (1 if current_chunk else 0) > max_chars
            would_exceed_sentences = current_sentences >= max_sentences

            # Decide whether to flush current chunk
            should_flush = False

            if would_exceed_chars and current_chunk:
                should_flush = True
            elif would_exceed_sentences and current_sentences >= min_sentences:
                should_flush = True

            if should_flush:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_sentences = 0
                current_length = 0

            current_chunk.append(sentence)
            current_sentences += 1
            current_length += sent_len + (1 if len(current_chunk) > 1 else 0)

        # After each paragraph, consider flushing if we have min_sentences
        if current_sentences >= min_sentences:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_sentences = 0
            current_length = 0

    # Don't forget remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


class TTSRequest(BaseModel):
    """TTS generation request."""
    text: str = Field(..., description="Text to synthesize", min_length=1)
    language: Language = Field(Language.AUTO, description="Output language")
    speaker: Speaker = Field(Speaker.RYAN, description="Voice speaker")
    instructions: Optional[str] = Field("", description="Style/emotion instructions")
    audio_format: AudioFormat = Field(AudioFormat.MP3, description="Output audio format")


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
            "/tts": "Generate speech with predefined speaker (auto-chunking for long text)",
            "/tts/batch": "Batch generate multiple texts",
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

    Automatically handles long texts by chunking into 2-3 sentence
    segments for optimal memory usage and audio quality.

    - **text**: Text to synthesize (required, any length supported)
    - **language**: Output language (default: Auto)
    - **speaker**: Voice speaker (default: Ryan)
    - **instructions**: Style/emotion instructions (optional)
    - **audio_format**: Output format (default: mp3)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Smart chunk text for long inputs
        chunks = smart_chunk_text(request.text)

        if not chunks:
            raise HTTPException(status_code=400, detail="No text to synthesize")

        gen_start = time.time()

        # Generate audio for each chunk
        all_wavs = []
        sr = None

        for chunk in chunks:
            wavs, sr = model.generate_custom_voice(
                text=chunk,
                language=request.language.value,
                speaker=request.speaker.value,
                instruct=request.instructions or "",
            )
            all_wavs.extend(wavs)

        gen_time = time.time() - gen_start

        # Concatenate all audio
        combined = np.concatenate(all_wavs)
        audio_duration = len(combined) / sr

        # Convert to bytes (supports WAV, MP3, FLAC)
        buffer = encode_audio(combined, sr, request.audio_format)

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
            "X-Chunks": str(len(chunks)),
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
    audio_format: AudioFormat = Form(AudioFormat.MP3),
):
    """
    Batch generate speech from multiple texts.

    - **texts**: List of texts to synthesize (form field, comma-separated for multiple)
    - **languages**: Comma-separated languages or single language for all
    - **speakers**: Comma-separated speakers or single speaker for all
    - **instructions**: Comma-separated instructions or single for all
    - **audio_format**: Output format (default: mp3)
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
        combined = np.concatenate(wavs)
        audio_duration = len(combined) / sr

        # Convert to bytes (supports WAV, MP3, FLAC)
        buffer = encode_audio(combined, sr, audio_format)

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
    audio_format: AudioFormat = Form(AudioFormat.MP3),
):
    """
    Generate speech with cloned voice from reference audio.

    - **text**: Text to synthesize
    - **ref_audio**: Reference audio file for voice cloning
    - **ref_text**: Transcript of reference audio (improves quality)
    - **language**: Output language (default: Auto)
    - **audio_format**: Output format (default: mp3)
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

        # Convert to bytes (supports WAV, MP3, FLAC)
        buffer = encode_audio(wavs[0], sr, audio_format)

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
