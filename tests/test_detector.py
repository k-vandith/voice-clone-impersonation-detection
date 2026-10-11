from pathlib import Path

import numpy as np
import pytest

from src.detector import analyze_wav_bytes, build_demo_wav, detect, extract_features


def test_detect():
    p = Path(__file__).resolve().parents[1] / "data" / "sample" / "natural_tone.wav"
    if not p.exists():
        import runpy
        runpy.run_path(str(p.parents[2] / "scripts" / "generate_demo_data.py"))
    result = detect(p)
    assert 0 <= result.confidence <= 1
    assert result.alert in ("HEURISTIC_REVIEW_ONLY", "INSUFFICIENT_AUDIO_SIGNAL")


def test_detect_uses_same_validated_decoder_as_upload_analysis(tmp_path: Path):
    raw = build_demo_wav("processed")
    path = tmp_path / "tone.wav"
    path.write_bytes(raw)

    direct = detect(path)
    upload = analyze_wav_bytes(raw)

    assert direct.features == upload["features"]
    assert direct.confidence == upload["threshold_weight_score"]
    assert direct.is_synthetic_likely is None
    assert upload["is_synthetic_likely"] is None
    assert upload["decision_status"] == "NOT_VALIDATED_HUMAN_REVIEW_ONLY"


def test_detect_rejects_non_wav_bytes_instead_of_scoring_them_as_audio(tmp_path: Path):
    path = tmp_path / "not-audio.bin"
    path.write_bytes(b"not a wav file")

    with pytest.raises(ValueError, match="uncompressed PCM WAV"):
        detect(path)


@pytest.mark.parametrize("sample_rate", [0, -1, True, 1.5])
def test_feature_extractor_rejects_invalid_sample_rate(sample_rate):
    with pytest.raises(ValueError, match="positive integer"):
        extract_features(np.ones(16, dtype=np.float32), sample_rate)


def test_feature_extractor_rejects_nonfinite_or_multichannel_arrays():
    with pytest.raises(ValueError, match="finite"):
        extract_features(np.array([0.0, np.nan, 0.1]), 16000)
    with pytest.raises(ValueError, match="one-dimensional"):
        extract_features(np.ones((4, 2)), 16000)

@pytest.mark.parametrize("amplitude", [1.01, -1.01, 1e30])
def test_feature_extractor_rejects_out_of_range_finite_samples(amplitude):
    samples = np.full(32, amplitude, dtype=np.float32)

    with pytest.raises(ValueError, match="normalized to the range"):
        extract_features(samples, 16000)
