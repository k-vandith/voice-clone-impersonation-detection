<p align="center">
  <img src="web/mark.svg" alt="Soniq logo" width="70">
</p>
<h1 align="center">Soniq — Voice Integrity Studio</h1>
<p align="center">Local-first WAV review with a waveform preview and explainable acoustic signal checks.</p>

<p align="center">
  <img src="https://img.shields.io/github/actions/workflow/status/k-vandith/voice-clone-impersonation-detection/tests.yml?branch=main&label=CI" alt="CI status">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-MIT-777777" alt="MIT License">
</p>

## Quick start

    git clone https://github.com/k-vandith/voice-clone-impersonation-detection.git
    cd voice-clone-impersonation-detection
    python -m venv .venv

Activate the environment (Windows: \`.venv\Scripts\Activate.ps1\`; macOS/Linux: \`source .venv/bin/activate\`), then run:

    python -m pip install -r requirements.txt
    python run.py

Open http://127.0.0.1:8501.

## Use it

1. Upload or drop a PCM WAV file (mono/stereo, 8–96 kHz, 0.25–120 seconds, maximum 12 MB).
2. Inspect the waveform and acoustic measurements, then export a JSON review report.
3. Use the included test tones to try the interface; they are generated tones, not speech.

## Tests

    python -m pip install -r requirements-dev.txt
    pytest -q

## Privacy and limitations

Audio is processed in memory by the local server and is not saved to disk or sent to an external service. The score uses simple signal heuristics; it is **not** a calibrated probability and has not been validated as a real-world voice-clone detector. A low score does not prove authenticity.

## License

MIT
