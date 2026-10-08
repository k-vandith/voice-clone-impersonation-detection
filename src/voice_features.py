"""Voice clone detection: synthetic MFCC features, classifier, risk meter."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def extract_features_from_array(samples: np.ndarray, sr: int = 16000) -> dict[str, float]:
    samples = np.asarray(samples, dtype=float)
    if samples.size == 0: samples = np.zeros(sr)
    frame = 512
    energies = [float(np.mean(samples[i:i+frame]**2)) for i in range(0, max(1, len(samples)-frame), frame)] or [0.0]
    energies = np.array(energies)
    window = samples[:min(len(samples), 2048)]
    if len(window) < 2048: window = np.pad(window, (0, 2048-len(window)))
    spec = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    logspec = np.log(spec + 1e-8)
    mfcc = [float(np.sum(logspec * np.cos(np.pi*k*(np.arange(len(logspec))+0.5)/len(logspec)))) for k in range(min(13, len(logspec)))]
    return {"energy_mean": float(np.mean(energies)), "energy_std": float(np.std(energies)), "zcr": float(np.mean(np.abs(np.diff(np.sign(samples))))/2), "mfcc0": mfcc[0] if mfcc else 0.0, "mfcc1": mfcc[1] if len(mfcc)>1 else 0.0, "mfcc2": mfcc[2] if len(mfcc)>2 else 0.0, "spectral_centroid": float(np.sum(np.arange(len(spec))*spec)/(np.sum(spec)+1e-8))}

def build_synthetic_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        spoof = i % 2 == 1
        samples = rng.normal(0.4 if spoof else 0.0, 0.5 if spoof else 1.0, size=8000)
        feats = extract_features_from_array(samples); feats["label"] = 1 if spoof else 0; rows.append(feats)
    return pd.DataFrame(rows)

def train_spoof_classifier(df: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [c for c in df.columns if c != "label"]
    X, y = df[feature_cols].values, df["label"].values
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=40, random_state=42); clf.fit(Xtr, ytr)
        return {"backend": "sklearn", "model": clf, "feature_cols": feature_cols, "accuracy": float(accuracy_score(yte, clf.predict(Xte)))}
    except ImportError:
        return {"backend": "rules", "model": None, "feature_cols": feature_cols, "accuracy": None}

def score_risk(feats: dict[str, float], model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    if model_info and model_info.get("backend") == "sklearn" and model_info.get("model") is not None:
        X = np.array([[feats.get(c, 0.0) for c in model_info["feature_cols"]]])
        proba = model_info["model"].predict_proba(X)[0]
        p = float(proba[1]) if len(proba) > 1 else float(proba[0])
    else:
        p = float(1.0 / (1.0 + feats.get("energy_std", 1.0)))
    level = "high" if p >= 0.7 else ("medium" if p >= 0.4 else "low")
    return {"spoof_probability": round(p, 4), "risk_level": level, "alert": level == "high"}
