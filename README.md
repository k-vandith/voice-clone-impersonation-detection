# Voice Clone Impersonation Detection

Heuristic detector for synthetic / cloned voice indicators using signal features (e.g. MFCC-style summaries) for defensive authentication research.

## Problem Statement

Generative voice cloning increases impersonation risk in call centres and executive channels. Organisations need a local, explainable detector that flags anomalous voice characteristics without cloud audio upload.

## Overview

Extract lightweight acoustic features from audio samples (or synthetic feature vectors in demo mode), score clone likelihood, and present results in Streamlit.

## Features

- **Feature extraction** – spectral / cepstral summaries
- **Clone likelihood score** – heuristic model
- **Streamlit review UI**
- **Demo mode** – works without real microphone capture
- **Batch scoring** of sample sets

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Streamlit  │────▶│   Detector   │────▶│  Feature +  │
│     UI      │     │              │     │  Score layer│
└─────────────┘     └──────┬───────┘     └─────────────┘
                           │
                    ┌──────▼───────┐
                    │  Audio / CSV │
                    └──────────────┘
```

## Tech Stack

- Python 3.11+
- NumPy / Pandas
- Streamlit
- pytest

## Repository Structure

```
voice-clone-impersonation-detection/
├── README.md
├── requirements.txt
├── src/
│   └── detector.py
├── tests/
│   └── test_detector.py
├── data/
├── scripts/
│   ├── setup_env.py
│   ├── setup.sh
│   ├── setup.ps1
│   └── generate_demo_data.py
└── docs/
```

## System Requirements

| Mode | CPU | RAM | Disk | GPU |
|------|-----|-----|------|-----|
| Demo | Any | 1 GB | 500 MB | Not needed |

## Installation

### Recommended (all platforms) — automated bootstrap

Handles missing `ensurepip`, symlink restrictions, and installs dependencies into `.venv`:

```bash
git clone https://github.com/k-vandith/voice-clone-impersonation-detection.git
cd voice-clone-impersonation-detection
python3 scripts/setup_env.py    # or:  python scripts/setup_env.py
```

Then activate:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

### Manual setup

#### Windows (PowerShell)

```powershell
git clone https://github.com/k-vandith/voice-clone-impersonation-detection.git
cd voice-clone-impersonation-detection
python -m venv .venv --copies
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux / macOS

```bash
git clone https://github.com/k-vandith/voice-clone-impersonation-detection.git
cd voice-clone-impersonation-detection
# If venv fails with ensurepip errors:
#   sudo apt install python3-venv python3-pip
python3 -m venv .venv --copies
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Why `--copies`?

Some environments cannot create symlinks inside a venv (`Operation not permitted` on `lib64 → lib`). Using `--copies` avoids that. `scripts/setup_env.py` tries `--copies` first automatically.

## Environment Variables

None required.

## Dataset / Demo Mode

```bash
python scripts/generate_demo_data.py
```

## Running the Application

```bash
streamlit run src/detector.py
```

## API Usage

```python
from src.detector import detect
print(detect("data/sample_features.json"))
```

## Testing

```bash
pytest -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: src` | Run from project root; ensure `PYTHONPATH=.` |
| `venv` / ensurepip fails | Run `python3 scripts/setup_env.py` or install `python3-venv` |
| `Operation not permitted` on lib64 | Use `python3 -m venv .venv --copies` |
| Missing dependency | Activate `.venv` and re-run `pip install -r requirements.txt` |

## Limitations

- Research / demo-grade heuristics — not a certified biometric system.
- Real-time telephony integration is out of scope.
- Optional deep models (if added later) need extra dependencies.

## Security / Privacy

- Defensive detection only — do not use to forge or attack voice systems.
- Keep real user audio offline and consent-managed.

## Future Improvements

- Optional torchaudio / librosa backends
- Streaming inference
- Enrollment vs probe workflows

## License

MIT

## Interface

```bash
python run.py
```

Opens the local Streamlit workspace on port 8501. Demo paths work without GPU, webcam, or a paid API. `streamlit run src/app.py` is equivalent.
