import io
import json
import math
import wave
from http.server import ThreadingHTTPServer
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from src.detector import analyze_wav_bytes, build_demo_wav
from src.webapp import VoiceReviewHandler


def make_wav(sample_width=2, channels=1, sample_rate=16000, duration=1.0):
    count = int(sample_rate * duration)
    t = np.arange(count, dtype=np.float64) / sample_rate
    signal = 0.45 * np.sin(2 * np.pi * 330 * t)
    if sample_width == 1:
        mono = np.clip(signal * 127 + 128, 0, 255).astype(np.uint8)
    elif sample_width == 2:
        mono = (signal * 32767).astype("<i2")
    elif sample_width == 3:
        values = (signal * 8388607).astype(np.int32)
        packed = np.empty((count, 3), dtype=np.uint8)
        packed[:, 0] = values & 0xFF
        packed[:, 1] = (values >> 8) & 0xFF
        packed[:, 2] = (values >> 16) & 0xFF
        mono = packed.reshape(-1)
    elif sample_width == 4:
        mono = (signal * 2147483647).astype("<i4")
    else:
        raise ValueError("unsupported fixture width")

    if channels == 2:
        if sample_width == 3:
            mono = mono.reshape(-1, 3)
            mono = np.repeat(mono[:, None, :], 2, axis=1).reshape(-1)
        else:
            mono = np.repeat(mono[:, None], 2, axis=1).reshape(-1)

    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(mono.tobytes())
    return output.getvalue()


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), VoiceReviewHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_analysis_returns_signal_features_and_waveform():
    result = analyze_wav_bytes(build_demo_wav("reference"))

    assert result["duration_seconds"] == 1.5
    assert result["sample_rate"] == 16000
    assert result["channels"] == 1
    assert len(result["waveform"]) == 240
    assert set(result["features"]) == {
        "zcr", "energy", "spectral_centroid_proxy", "flatness"
    }
    assert 0 <= result["score_percent"] <= 100
    assert result["risk_level"] in {"low", "review", "high"}
    assert "not a validated probability" in result["notes"]
    assert all(math.isfinite(value) for value in result["features"].values())


@pytest.mark.parametrize("sample_width", [1, 2, 3, 4])
def test_analysis_accepts_supported_pcm_sample_widths(sample_width):
    result = analyze_wav_bytes(make_wav(sample_width=sample_width))
    assert result["sample_width_bits"] == sample_width * 8
    assert result["sample_count"] == 16000


def test_analysis_downmixes_stereo_wav():
    result = analyze_wav_bytes(make_wav(channels=2))
    assert result["channels"] == 2
    assert result["sample_count"] == 16000


def test_analysis_rejects_non_wav_data():
    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        analyze_wav_bytes(b"not an audio file")


def test_analysis_rejects_empty_data():
    with pytest.raises(ValueError, match="Choose a WAV"):
        analyze_wav_bytes(b"")


def test_analysis_rejects_short_audio():
    with pytest.raises(ValueError, match="at least 0.25 seconds"):
        analyze_wav_bytes(make_wav(duration=0.1))


def test_analysis_rejects_oversized_upload():
    with pytest.raises(ValueError, match="12 MB or smaller"):
        analyze_wav_bytes(b"x" * (12 * 1024 * 1024 + 1))


def test_analysis_rejects_truncated_wav():
    with pytest.raises(ValueError):
        analyze_wav_bytes(build_demo_wav("reference")[:-24])


def test_demo_builder_rejects_unknown_fixture():
    with pytest.raises(ValueError, match="reference or processed"):
        build_demo_wav("spoken")


def test_home_and_health_endpoint(local_server):
    with urlopen(local_server + "/") as response:
        page = response.read().decode("utf-8")
        assert response.status == 200
        assert "Soniq" in page
        assert "Content-Security-Policy" in response.headers

    with urlopen(local_server + "/api/health") as response:
        health = json.loads(response.read())
        assert health["status"] == "ok"
        assert health["mode"] == "local-only"


def test_upload_endpoint_analyzes_wav_in_memory(local_server):
    request = Request(
        local_server + "/api/analyze",
        data=build_demo_wav("processed"),
        headers={"Content-Type": "application/octet-stream", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        result = json.loads(response.read())
        assert response.status == 200
        assert result["source"] == "uploaded"
        assert result["duration_seconds"] == 1.5
        assert len(result["waveform"]) > 0


def test_upload_endpoint_rejects_wrong_content_type(local_server):
    request = Request(
        local_server + "/api/analyze",
        data=b"not a wave",
        headers={"Content-Type": "text/plain"},
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request)
    assert error.value.code == 415
    assert "PCM WAV" in error.value.read().decode("utf-8")
