"""Voice risk console. Synthetic audio by default."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from src.voice_features import build_synthetic_dataset, extract_features_from_array, score_risk, train_spoof_classifier
from src.ui_theme import theme_css

@st.cache_resource
def _model():
    df = build_synthetic_dataset(180)
    return train_spoof_classifier(df)

def main() -> None:
    st.set_page_config(page_title="Voice defense", layout="wide")
    st.markdown(theme_css("#b79bff"), unsafe_allow_html=True)
    info = _model()
    st.markdown(f'<div class="top"><div><div class="kicker">Audio security</div><p class="title">Voice clone check</p></div><div class="pill">{info.get("backend")} · accuracy {info.get("accuracy")}</div></div>', unsafe_allow_html=True)
    kind = st.radio("Sample", ["Likely genuine", "Likely spoof"], horizontal=True)
    rng = np.random.default_rng(1 if kind.startswith("Likely g") else 2)
    samples = rng.normal(0.0 if kind.startswith("Likely g") else 0.4, 1.0 if kind.startswith("Likely g") else 0.5, size=8000)
    feats = extract_features_from_array(samples)
    risk = score_risk(feats, info)
    st.markdown(f'<div class="panel"><div class="kicker">Decision</div><p class="title">{risk["risk_level"]} · {risk["spoof_probability"]:.0%} spoof probability</p><p class="muted">This is a CPU demo on synthetic audio, not a certified detector. A high score means the local model is unsure the clip is genuine.</p></div>', unsafe_allow_html=True)
    fig = go.Figure(go.Scatter(y=samples[::8], line=dict(color="#b79bff", width=1)))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e7ecf3", height=280, title="Waveform (downsampled)")
    st.plotly_chart(fig, use_container_width=True)
    st.json({k: round(v, 4) if isinstance(v, float) else v for k, v in feats.items()})

if __name__ == "__main__":
    main()
