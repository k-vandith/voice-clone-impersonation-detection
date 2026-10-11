"""Voice clone detection: compact signal features, classifier and risk meter."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def extract_features_from_array(samples: np.ndarray, sr: int = 16000) -> dict[str, float]:
    if not isinstance(sr, int) or isinstance(sr, bool) or sr <= 0:
        raise ValueError("Sample rate must be a positive integer.")
    values = np.asarray(samples, dtype=float)
    if values.ndim != 1:
        raise ValueError("Audio samples must be a one-dimensional mono array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Audio samples must contain only finite values.")
    if values.size == 0:
        values = np.zeros(2048, dtype=float)

    frame = 512
    energy_values = [
        float(np.mean(values[index:index + frame] ** 2))
        for index in range(0, max(1, len(values) - frame + 1), frame)
    ] or [0.0]
    energies = np.asarray(energy_values, dtype=float)
    window = values[:min(len(values), 2048)]
    if len(window) < 2048:
        window = np.pad(window, (0, 2048 - len(window)))
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window))))
    log_spectrum = np.log(spectrum + 1e-8)
    coefficients = [
        float(np.sum(
            log_spectrum
            * np.cos(np.pi * coefficient * (np.arange(len(log_spectrum)) + 0.5) / len(log_spectrum))
        ))
        for coefficient in range(min(13, len(log_spectrum)))
    ]
    result = {
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "zcr": float(np.mean(np.abs(np.diff(np.sign(values)))) / 2) if len(values) > 1 else 0.0,
        "mfcc0": coefficients[0] if coefficients else 0.0,
        "mfcc1": coefficients[1] if len(coefficients) > 1 else 0.0,
        "mfcc2": coefficients[2] if len(coefficients) > 2 else 0.0,
        "spectral_centroid": float(np.sum(np.arange(len(spectrum)) * spectrum) / (np.sum(spectrum) + 1e-8)),
    }
    if not np.isfinite(np.asarray(list(result.values()), dtype=float)).all():
        raise ValueError("Feature extraction produced non-finite values.")
    return result


def build_synthetic_dataset(n: int = 200, seed: int = 42) -> pd.DataFrame:
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("Synthetic dataset size must be a positive integer.")
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        spoof = index % 2 == 1
        samples = rng.normal(0.4 if spoof else 0.0, 0.5 if spoof else 1.0, size=8000)
        features = extract_features_from_array(samples)
        features["label"] = 1 if spoof else 0
        rows.append(features)
    frame = pd.DataFrame(rows)
    frame.attrs["synthetic_demo_data"] = True
    return frame


def train_spoof_classifier(df: pd.DataFrame) -> dict[str, Any]:
    if "label" not in df.columns:
        raise ValueError("Training data must include a binary 'label' column.")
    if df.empty:
        raise ValueError("Training data must contain at least 8 labelled examples.")
    feature_cols = [column for column in df.columns if column != "label"]
    if not feature_cols:
        raise ValueError("Training data must include at least one feature column.")
    if len(df) < 8:
        raise ValueError("Training data must contain at least 8 labelled examples.")

    labels = pd.to_numeric(df["label"], errors="coerce").to_numpy()
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be numeric 0 (natural) or 1 (spoof).")
    classes, counts = np.unique(labels.astype(int), return_counts=True)
    if not np.array_equal(classes, np.array([0, 1])) or np.any(counts < 2):
        raise ValueError("Training data must include at least two examples of each label (0 and 1).")

    try:
        X = df[feature_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("All model feature columns must contain numeric values.") from error
    if not np.isfinite(X).all():
        raise ValueError("Model feature columns must contain only finite values.")
    y = labels.astype(int)

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        classifier = RandomForestClassifier(n_estimators=40, random_state=42)
        classifier.fit(X_train, y_train)
        accuracy = float(accuracy_score(y_test, classifier.predict(X_test)))
        return {
            "backend": "sklearn",
            "model": classifier,
            "feature_cols": feature_cols,
            "accuracy": None if df.attrs.get("synthetic_demo_data") else accuracy,
            "calibrated": False,
            "validation_scope": "synthetic_demo_only" if df.attrs.get("synthetic_demo_data") else "holdout_only_unvalidated",
        }
    except ImportError:
        return {"backend": "rules", "model": None, "feature_cols": feature_cols, "accuracy": None, "calibrated": False, "validation_scope": "not_validated"}


def score_risk(feats: dict[str, float], model_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score only with an explicit model; never invent a spoof probability from energy."""
    model_score: float | None = None
    has_model = bool(model_info and model_info.get("backend") == "sklearn" and model_info.get("model") is not None)
    if has_model:
        feature_cols = model_info.get("feature_cols") or []
        missing = [column for column in feature_cols if column not in feats]
        if missing:
            raise ValueError(f"Missing model features: {', '.join(missing)}")
        values = np.asarray([feats[column] for column in feature_cols], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Model features must contain only finite values.")
        classifier = model_info["model"]
        probabilities = np.asarray(classifier.predict_proba(values.reshape(1, -1))[0], dtype=float)
        classes = np.asarray(classifier.classes_)
        if len(probabilities) != len(classes) or not np.isfinite(probabilities).all():
            raise ValueError("Classifier returned invalid probability values.")
        spoof_indexes = np.flatnonzero(classes == 1)
        if len(spoof_indexes):
            model_score = float(probabilities[int(spoof_indexes[0])])
        elif len(classes) == 1 and classes[0] == 0:
            model_score = 0.0
        elif len(classes) == 1 and classes[0] == 1:
            model_score = 1.0
        else:
            raise ValueError("Classifier must use numeric classes 0 (natural) and 1 (spoof).")
        model_score = min(1.0, max(0.0, model_score))

    calibrated = bool(model_info and model_info.get("calibrated") is True and model_score is not None)
    if not calibrated:
        return {
            "spoof_probability": None,
            "model_score": round(model_score, 4) if model_score is not None else None,
            "risk_level": "review",
            "alert": False,
            "status": "uncalibrated_model" if has_model else "no_validated_model",
            "interpretation": (
                "Uncalibrated model output; no authenticity decision is supported. Human review required."
                if has_model else
                "No trained and calibrated model is available; acoustic features alone do not establish voice cloning."
            ),
        }

    probability = float(model_score)
    level = "high" if probability >= 0.7 else ("medium" if probability >= 0.4 else "low")
    return {
        "spoof_probability": round(probability, 4),
        "model_score": round(probability, 4),
        "risk_level": level,
        "alert": level == "high",
        "status": "calibrated_model",
        "interpretation": "Model probability explicitly marked calibrated by caller; validate calibration on representative deployment data.",
    }
