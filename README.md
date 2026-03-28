# Qwen3-TTS FastAPI Server

A production-ready FastAPI server for Qwen3-TTS 0.6B text-to-speech model with Docker support.

## Features

- **CustomVoice**: Generate speech with 9 predefined speakers and style instructions
- **Voice Cloning**: Clone voice from reference audio
- **Batch Processing**: Generate multiple texts in one request
- **Multiple Output Formats**: WAV, MP3, FLAC
- **Docker Ready**: Easy deployment with Docker Compose
- **Auto Device Detection**: Works on CUDA, MPS (Apple Silicon), and CPU

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwentts-fastapi.git
cd qwentts-fastapi

# Start the server (first run will download the model ~2GB)
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/qwentts-fastapi.git
cd qwentts-fastapi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

## API Documentation

Once the server is running, access the interactive API docs at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### `GET /`
Root endpoint with API information.

### `GET /health`
Health check endpoint.

### `GET /voices`
List available speakers.

**Response:**
```json
{
  "voices": ["Vivian", "Ryan", "Aiden", "Josh", "Drew", "Mia", "Emma", "Amy", "Brian"],
  "default": "Ryan"
}
```

### `GET /languages`
List supported languages.

**Response:**
```json
{
  "languages": ["Auto", "Chinese", "English", "Japanese", "Korean", "German", "French", "Russian", "Portuguese", "Spanish", "Italian"],
  "default": "Auto"
}
```

### `POST /tts`
Generate speech with a predefined speaker.

**Request Body:**
```json
{
  "text": "Hello, how are you today?",
  "language": "English",
  "speaker": "Ryan",
  "instructions": "Speak with enthusiasm and energy.",
  "audio_format": "wav"
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| text | string | required | Text to synthesize |
| language | string | "Auto" | Output language |
| speaker | string | "Ryan" | Voice speaker |
| instructions | string | "" | Style/emotion instructions |
| audio_format | string | "wav" | Output format: wav, mp3, flac |

**Response Headers:**
- `X-Generation-Time`: Time taken to generate (seconds)
- `X-Audio-Duration`: Duration of generated audio (seconds)
- `X-Real-Time-Factor`: Audio duration / generation time
- `X-Characters`: Number of characters processed

### `POST /tts/batch`
Generate speech from multiple texts (form-data).

**Form Parameters:**
- `texts`: List of texts (required)
- `languages`: Comma-separated languages or single for all
- `speakers`: Comma-separated speakers or single for all
- `instructions`: Pipe-separated instructions or single for all
- `audio_format`: Output format (default: wav)

### `POST /tts/clone`
Generate speech with a cloned voice from reference audio.

**Form Parameters:**
- `text`: Text to synthesize (required)
- `ref_audio`: Reference audio file (required)
- `ref_text`: Transcript of reference audio (improves quality)
- `language`: Output language
- `audio_format`: Output format

## Usage Examples

### cURL

```bash
# Simple TTS
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you today?",
    "speaker": "Ryan",
    "instructions": "Speak with enthusiasm."
  }' \
  --output output.wav

# With style instructions
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is absolutely amazing!",
    "speaker": "Vivian",
    "instructions": "Very excited and happy."
  }' \
  --output excited.wav

# Voice cloning
curl -X POST "http://localhost:8000/tts/clone" \
  -F "text=Hello, this is a cloned voice" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=This is the original voice sample" \
  --output cloned.wav
```

### Python

```python
import httpx

# Simple TTS
response = httpx.post(
    "http://localhost:8000/tts",
    json={
        "text": "Hello, how are you today?",
        "speaker": "Ryan",
        "instructions": "Speak with enthusiasm."
    },
    timeout=60.0
)

with open("output.wav", "wb") as f:
    f.write(response.content)

print(f"Generation time: {response.headers['X-Generation-Time']}s")
print(f"Audio duration: {response.headers['X-Audio-Duration']}s")
```

### Using OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Note: Create a compatible wrapper if needed
response = httpx.post(
    "http://localhost:8000/tts",
    json={"text": "Hello world!", "speaker": "Ryan"}
)
```

## Available Speakers

| Speaker | Description |
|---------|-------------|
| Vivian | Female voice |
| Ryan | Male voice (default) |
| Aiden | Male voice |
| Josh | Male voice |
| Drew | Male voice |
| Mia | Female voice |
| Emma | Female voice |
| Amy | Female voice |
| Brian | Male voice |

## Supported Languages

- Auto (automatic detection)
- Chinese
- English
- Japanese
- Korean
- German
- French
- Russian
- Portuguese
- Spanish
- Italian

## Style Instructions

Use the `instructions` parameter to control emotion and style:

```json
{
  "instructions": "Speak with enthusiasm and energy."
}
```

**Examples:**
- `"Speak with enthusiasm and energy."`
- `"Very excited and happy."`
- `"Calm and relaxed tone."`
- `"Professional and formal."`
- `"用特别愤怒的语气说"` (Chinese: "Speak with an angry tone")

## Configuration

Environment variables can be set in `.env` file:

```env
# Model settings
MODEL_NAME=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
DEVICE=auto
DTYPE=float32

# Server settings
HOST=0.0.0.0
PORT=8000
```

### Device Options

| Device | Description |
|--------|-------------|
| `auto` | Automatically detect best device |
| `cuda:0` | NVIDIA GPU |
| `mps` | Apple Silicon GPU |
| `cpu` | CPU only |

## Docker Commands

```bash
# Build the image
docker build -t qwentts-fastapi .

# Run container
docker run -p 8000:8000 qwentts-fastapi

# Run with GPU support
docker run --gpus all -p 8000:8000 qwentts-fastapi

# Using docker-compose
docker-compose up -d        # Start in background
docker-compose logs -f      # View logs
docker-compose down         # Stop and remove
docker-compose down -v      # Stop and remove volumes (clears model cache)
```

## Performance

| Device | Model | Speed (chars/sec) | Real-time Factor |
|--------|-------|-------------------|------------------|
| Apple M4 (MPS) | 0.6B | ~4.2 | 0.40x |
| NVIDIA RTX 3080 (CUDA) | 0.6B | ~15+ | 1.5x+ |
| CPU | 0.6B | ~1.5 | 0.15x |

## Model Size

- **Model**: Qwen3-TTS-0.6B-CustomVoice
- **Download Size**: ~2GB
- **VRAM Required**: ~3GB (FP32), ~2GB (FP16)
- **Disk Space**: ~2.5GB (with cache)

## Troubleshooting

### Model download fails
```bash
# Pre-download model manually
python -c "from qwen_tts import Qwen3TTSModel; Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice')"
```

### Out of memory
- Use smaller batch sizes
- Set `DTYPE=float16` (may cause issues on MPS)
- Use CPU if GPU memory is insufficient

### Port already in use
```bash
# Change port in docker-compose.yml or run with:
uvicorn app.main:app --port 8001
```

## License

This project uses the Qwen3-TTS model. Please refer to the model's license for usage terms.

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Qwen Team
- Model: [Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice)
