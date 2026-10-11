import numpy as np
import pandas as pd
import pytest

from src.voice_features import (
    build_synthetic_dataset,
    extract_features_from_array,
    score_risk,
    train_spoof_classifier,
)


def test_voice():
    df = build_synthetic_dataset(40)
    info = train_spoof_classifier(df)
    risk = score_risk(extract_features_from_array(np.random.randn(4000)), info)
    assert "risk_level" in risk


def test_empty_array_produces_finite_silence_features():
    features = extract_features_from_array(np.array([]))

    assert all(np.isfinite(value) for value in features.values())
    assert features["energy_mean"] == 0
    assert features["zcr"] == 0


def test_feature_array_rejects_nonfinite_and_invalid_sample_rate():
    with pytest.raises(ValueError, match="finite"):
        extract_features_from_array(np.array([0.0, np.nan, 0.1]))
    with pytest.raises(ValueError, match="positive integer"):
        extract_features_from_array(np.ones(10), sr=0)


def test_spoof_training_rejects_missing_and_single_class_labels():
    with pytest.raises(ValueError, match="binary 'label'"):
        train_spoof_classifier(pd.DataFrame({"mfcc0": [0.1] * 10}))
    with pytest.raises(ValueError, match="at least two examples"):
        train_spoof_classifier(pd.DataFrame({"mfcc0": np.arange(10), "label": [0] * 10}))


def test_spoof_probability_uses_class_label_not_probability_position():
    class ReorderedClassesModel:
        classes_ = np.array([1, 0])

        def predict_proba(self, _features):
            return np.array([[0.82, 0.18]])

    output = score_risk(
        {"energy_std": 0.5},
        {"backend": "sklearn", "model": ReorderedClassesModel(), "feature_cols": ["energy_std"]},
    )

    assert output["spoof_probability"] == 0.82
    assert output["risk_level"] == "high"


def test_single_class_natural_model_has_zero_spoof_probability():
    class NaturalOnlyModel:
        classes_ = np.array([0])

        def predict_proba(self, _features):
            return np.array([[1.0]])

    output = score_risk(
        {"energy_std": 0.5},
        {"backend": "sklearn", "model": NaturalOnlyModel(), "feature_cols": ["energy_std"]},
    )

    assert output["spoof_probability"] == 0.0
    assert output["risk_level"] == "low"


def test_synthetic_dataset_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="positive integer"):
        build_synthetic_dataset(0)
