"""Defensive voice-clone / deepfake audio heuristics (CPU-friendly demo)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np

@dataclass
class DetectionResult:
    is_synthetic_likely: bool
    confidence: float
    features: dict
    alert: str
    notes: str

def _read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    data = path.read_bytes()
    if data[:4] != b"RIFF":
        arr = np.frombuffer(data[:8000], dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        return arr, 16000
    sr = struct.unpack_from("<I", data, 24)[0]
    idx = data.find(b"data")
    if idx < 0:
        raise ValueError("No data chunk")
    size = struct.unpack_from("<I", data, idx + 4)[0]
    pcm = data[idx + 8: idx + 8 + size]
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr

def extract_features(samples: np.ndarray, sr: int) -> dict:
    if len(samples) == 0:
        return {"zcr": 0, "energy": 0, "spectral_centroid_proxy": 0, "flatness": 0}
    zcr = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2)
    energy = float(np.mean(samples ** 2))
    spec = np.abs(np.fft.rfft(samples[: min(len(samples), sr)]))
    freqs = np.fft.rfftfreq(min(len(samples), sr), 1 / sr)
    centroid = float(np.sum(freqs * spec) / (np.sum(spec) + 1e-9))
    flatness = float(np.exp(np.mean(np.log(spec + 1e-9))) / (np.mean(spec) + 1e-9))
    return {"zcr": round(zcr, 5), "energy": round(energy, 5), "spectral_centroid_proxy": round(centroid, 2), "flatness": round(flatness, 5)}

def detect(path: Path) -> DetectionResult:
    samples, sr = _read_wav_mono(Path(path))
    feats = extract_features(samples, sr)
    score = 0.0
    if feats["flatness"] > 0.4:
        score += 0.4
    if feats["zcr"] < 0.05 or feats["zcr"] > 0.3:
        score += 0.3
    if feats["energy"] < 0.001:
        score += 0.2
    score = min(1.0, score)
    likely = score >= 0.5
    return DetectionResult(
        is_synthetic_likely=likely,
        confidence=round(score, 3),
        features=feats,
        alert="POSSIBLE_SYNTHETIC_VOICE" if likely else "LIKELY_NATURAL",
        notes="Heuristic CPU demo only — not a production deepfake detector. Model uncertainty is high.",
    )


def analyze_wav_bytes(raw: bytes) -> dict:
    """Analyze an uploaded PCM WAV in memory and return explainable signal cues.

    This is an illustrative heuristic, not a validated voice-clone probability model.
    """
    import io
    import wave

    if not raw:
        raise ValueError("Choose a WAV audio file first.")
    if len(raw) > 12 * 1024 * 1024:
        raise ValueError("Audio files must be 12 MB or smaller.")
    if not raw.startswith(b"RIFF") or raw[8:12] != b"WAVE":
        raise ValueError("Unsupported audio format. Upload an uncompressed PCM WAV file.")

    try:
        with wave.open(io.BytesIO(raw), "rb") as reader:
            channels = reader.getnchannels()
            sample_width = reader.getsampwidth()
            sample_rate = reader.getframerate()
            frame_count = reader.getnframes()
            compression = reader.getcomptype()
            if compression != "NONE":
                raise ValueError("Compressed WAV is not supported. Export as PCM WAV.")
            if channels not in (1, 2):
                raise ValueError("Upload mono or stereo WAV audio.")
            if sample_width not in (1, 2, 3, 4):
                raise ValueError("Unsupported PCM sample width.")
            if not 8000 <= sample_rate <= 96000:
                raise ValueError("Sample rate must be between 8 kHz and 96 kHz.")
            duration = frame_count / sample_rate if sample_rate else 0
            if duration < 0.25:
                raise ValueError("Audio must be at least 0.25 seconds long.")
            if duration > 120:
                raise ValueError("Audio must be 120 seconds or shorter.")
            pcm = reader.readframes(frame_count)
            expected_size = frame_count * channels * sample_width
            if len(pcm) != expected_size:
                raise ValueError("The WAV file appears truncated or malformed.")
    except (wave.Error, EOFError, struct.error) as error:
        raise ValueError("Could not parse the WAV header. Try exporting as PCM WAV.") from error

    if sample_width == 1:
        decoded = (np.frombuffer(pcm, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        decoded = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        triplets = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        decoded_int = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        decoded_int = np.where(decoded_int & 0x800000, decoded_int - 0x1000000, decoded_int)
        decoded = decoded_int.astype(np.float32) / 8388608.0
    else:
        decoded = np.frombuffer(pcm, dtype="<i4").astype(np.float32) / 2147483648.0

    if channels == 2:
        samples = decoded.reshape(-1, 2).mean(axis=1)
    else:
        samples = decoded
    samples = np.clip(samples, -1.0, 1.0)
    if not np.all(np.isfinite(samples)):
        raise ValueError("The WAV file contains invalid sample values.")

    feats = extract_features(samples, sample_rate)
    score = 0.0
    if feats["flatness"] > 0.4:
        score += 0.4
    if feats["zcr"] < 0.05 or feats["zcr"] > 0.3:
        score += 0.3
    if feats["energy"] < 0.001:
        score += 0.2
    score = min(1.0, score)
    risk_level = "high" if score >= 0.5 else ("review" if score >= 0.2 else "low")

    point_count = min(240, len(samples))
    indexes = np.linspace(0, len(samples) - 1, num=point_count, dtype=int)
    waveform = [round(float(value), 4) for value in samples[indexes]]
    rms = float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0

    return {
        "file_bytes": len(raw),
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width_bits": sample_width * 8,
        "sample_count": int(len(samples)),
        "rms": round(rms, 5),
        "peak": round(peak, 5),
        "features": feats,
        "score": round(score, 3),
        "score_percent": round(score * 100),
        "risk_level": risk_level,
        "is_synthetic_likely": score >= 0.5,
        "alert": "POSSIBLE_SYNTHETIC_VOICE" if score >= 0.5 else ("REVIEW_SIGNAL" if risk_level == "review" else "NO_STRONG_CUES"),
        "waveform": waveform,
        "notes": (
            "Transparent signal heuristics only. This score is not a validated probability of cloned speech; "
            "real-world accuracy has not been established."
        ),
    }


def build_demo_wav(kind: str = "reference") -> bytes:
    """Create a deterministic test tone for interface testing; it is not speech."""
    import io
    import wave

    if kind not in {"reference", "processed"}:
        raise ValueError("Choose the reference or processed demo tone.")
    sample_rate = 16000
    duration = 1.5
    count = int(sample_rate * duration)
    t = np.arange(count, dtype=np.float64) / sample_rate
    if kind == "reference":
        envelope = 0.55 + 0.3 * np.sin(2 * np.pi * 3.7 * t)
        signal = envelope * (
            0.50 * np.sin(2 * np.pi * 180 * t)
            + 0.28 * np.sin(2 * np.pi * 360 * t)
            + 0.15 * np.sin(2 * np.pi * 540 * t)
        )
    else:
        signal = 0.35 * np.sin(2 * np.pi * 740 * t)
    pcm = (np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes()
    stream = io.BytesIO()
    with wave.open(stream, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(pcm)
    return stream.getvalue()
