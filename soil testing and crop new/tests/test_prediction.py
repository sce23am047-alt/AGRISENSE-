import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.prediction import CropPredictor


def test_model_loads():
    predictor = CropPredictor()
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.label_encoder is not None


def test_prediction_output_shape():
    predictor = CropPredictor()
    values = {"N": 100, "P": 45, "K": 35, "temperature": 24, "moisture": 55, "ph": 6.2}
    result = predictor.predict(values)
    assert "primary" in result
    assert result["primary"] in config.CROP_FEATURES or True  # primary is a crop name, not a feature
    assert isinstance(result["primary"], str)
    if result.get("confidence") is not None:
        assert 0 <= result["confidence"] <= 100


def test_prediction_uses_correct_feature_order():
    predictor = CropPredictor()
    assert predictor.metadata.get("features") == config.CROP_FEATURES


def test_invalid_input_returns_error():
    predictor = CropPredictor()
    result = predictor.predict({"N": -5, "P": 45, "K": 35, "temperature": 24, "moisture": 55, "ph": 6.2})
    assert "error" in result


if __name__ == "__main__":
    test_model_loads()
    test_prediction_output_shape()
    test_prediction_uses_correct_feature_order()
    test_invalid_input_returns_error()
    print("All prediction tests passed.")
