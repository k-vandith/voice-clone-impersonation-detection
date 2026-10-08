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
