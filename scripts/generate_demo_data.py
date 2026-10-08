from pathlib import Path
import struct, math
ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
SAMPLE.mkdir(parents=True, exist_ok=True)
def write_tone(path, freq=440, sr=16000, dur=1.0):
    n = int(sr * dur)
    samples = [int(16000 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    pcm = b"".join(struct.pack("<h", max(-32767, min(32767, s))) for s in samples)
    header = struct.pack("<4sI4s4sIHHIIHH4sI", b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1, sr, sr * 2, 2, 16, b"data", len(pcm))
    path.write_bytes(header + pcm)
write_tone(SAMPLE / "natural_tone.wav", 220)
write_tone(SAMPLE / "synth_tone.wav", 880)
print("Wrote demo wavs")
