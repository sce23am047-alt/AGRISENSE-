"""
src/prediction.py
Loads the trained crop model and produces predictions with confidence
and probability-ranked alternatives. No hardcoded or random results —
everything comes from model.predict_proba().
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.utils import load_pickle, load_json
from src.validation import validate_soil_input


class CropPredictor:
    def __init__(self):
        self.model = load_pickle(config.CROP_MODEL_PATH, "ML model")
        self.scaler = load_pickle(config.CROP_SCALER_PATH, "Feature scaler")
        self.label_encoder = load_pickle(config.CROP_LABEL_ENCODER_PATH, "Label encoder")
        self.metadata = load_json(config.CROP_METADATA_PATH) or {}

    def predict(self, values: dict) -> dict:
        """
        values must contain exactly config.CROP_FEATURES keys:
        N, P, K, temperature, moisture, ph
        """
        errors = validate_soil_input(values)
        if errors:
            return {"error": errors}

        row = {f: [float(values[f])] for f in config.CROP_FEATURES}
        X = pd.DataFrame(row, columns=config.CROP_FEATURES)
        X_scaled = self.scaler.transform(X)

        pred_encoded = self.model.predict(X_scaled)[0]
        primary_crop = self.label_encoder.inverse_transform([pred_encoded])[0]

        result = {
            "primary": primary_crop,
            "confidence": None,
            "alternatives": [],
            "low_confidence": False,
        }

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_scaled)[0]
            classes = self.label_encoder.inverse_transform(np.arange(len(proba)))
            ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

            confidence = round(float(ranked[0][1]) * 100, 1)
            alternatives = [
                {"crop": c, "match": round(float(p) * 100, 1)} for c, p in ranked[1:3]
            ]

            result["confidence"] = confidence
            result["alternatives"] = alternatives
            result["low_confidence"] = confidence < config.LOW_CONFIDENCE_THRESHOLD

        return result
