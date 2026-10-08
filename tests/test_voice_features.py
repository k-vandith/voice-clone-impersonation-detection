from src.voice_features import build_synthetic_dataset, train_spoof_classifier, score_risk, extract_features_from_array
import numpy as np
def test_voice():
    df = build_synthetic_dataset(40)
    info = train_spoof_classifier(df)
    risk = score_risk(extract_features_from_array(np.random.randn(4000)), info)
    assert "risk_level" in risk
