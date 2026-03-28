"""
Qwen3-TTS API Test Client

Usage:
    python test_client.py
"""

import httpx
import sys


API_BASE = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = httpx.get(f"{API_BASE}/health", timeout=10.0)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    return response.status_code == 200


def test_voices():
    """Test voices endpoint."""
    print("\nTesting /voices endpoint...")
    response = httpx.get(f"{API_BASE}/voices", timeout=10.0)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    return response.status_code == 200


def test_languages():
    """Test languages endpoint."""
    print("\nTesting /languages endpoint...")
    response = httpx.get(f"{API_BASE}/languages", timeout=10.0)
    print(f"  Status: {response.status_code}")
    print(f"  Response: {response.json()}")
    return response.status_code == 200


def test_tts():
    """Test TTS generation."""
    print("\nTesting /tts endpoint...")
    response = httpx.post(
        f"{API_BASE}/tts",
        json={
            "text": "Hello, this is a test of the Qwen3 TTS system.",
            "language": "English",
            "speaker": "Ryan",
            "instructions": "Speak with enthusiasm.",
            "audio_format": "wav"
        },
        timeout=120.0
    )

    if response.status_code == 200:
        with open("test_output.wav", "wb") as f:
            f.write(response.content)
        print(f"  Status: {response.status_code}")
        print(f"  Generation Time: {response.headers.get('X-Generation-Time')}s")
        print(f"  Audio Duration: {response.headers.get('X-Audio-Duration')}s")
        print(f"  Real-time Factor: {response.headers.get('X-Real-Time-Factor')}x")
        print(f"  Saved to: test_output.wav")
        return True
    else:
        print(f"  Error: {response.status_code} - {response.text}")
        return False


def test_tts_chinese():
    """Test Chinese TTS."""
    print("\nTesting Chinese TTS...")
    response = httpx.post(
        f"{API_BASE}/tts",
        json={
            "text": "你好，这是一个测试。",
            "language": "Chinese",
            "speaker": "Vivian",
            "instructions": "用开心的语气说"
        },
        timeout=120.0
    )

    if response.status_code == 200:
        with open("test_output_chinese.wav", "wb") as f:
            f.write(response.content)
        print(f"  Status: {response.status_code}")
        print(f"  Saved to: test_output_chinese.wav")
        return True
    else:
        print(f"  Error: {response.status_code} - {response.text}")
        return False


def main():
    print("=" * 50)
    print("Qwen3-TTS API Test Client")
    print("=" * 50)

    # Check if server is running
    try:
        httpx.get(f"{API_BASE}/", timeout=5.0)
    except Exception as e:
        print(f"\nError: Cannot connect to server at {API_BASE}")
        print("Make sure the server is running:")
        print("  docker-compose up -d")
        print("  OR")
        print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
        sys.exit(1)

    results = []

    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Voices List", test_voices()))
    results.append(("Languages List", test_languages()))
    results.append(("TTS Generation (English)", test_tts()))
    results.append(("TTS Generation (Chinese)", test_tts_chinese()))

    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
