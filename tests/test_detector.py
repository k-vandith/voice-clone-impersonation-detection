from pathlib import Path
from src.detector import detect
def test_detect():
    p = Path(__file__).resolve().parents[1] / "data" / "sample" / "natural_tone.wav"
    if not p.exists():
        import runpy; runpy.run_path(str(p.parents[2] / "scripts" / "generate_demo_data.py"))
    r = detect(p)
    assert 0 <= r.confidence <= 1
    assert r.alert in ("POSSIBLE_SYNTHETIC_VOICE", "LIKELY_NATURAL")
