# Voice Clone Impersonation Detection

Defensive CPU-friendly heuristics for synthetic voice indicators. **Not** a production deepfake system; confidence is explicitly uncertain.

## Installation

```bash
# Linux/macOS
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Demo
```bash
python scripts/generate_demo_data.py
python -c "from src.detector import detect; print(detect('data/sample/natural_tone.wav'))"
pytest -v
```

## License
MIT
