"""
src/fertilizer.py

Precision Fertilizer Optimization — two components:

1. Rule-based engine (PRIMARY, per the PPT methodology "Module 4:
   Rule-Based Recommendation"). Compares the user's N/P/K against the
   data-derived nutrient profile for the recommended crop
   (models/crop_profiles.json, computed from the actual training
   data — not invented) and maps the resulting deficiency pattern to
   a fertilizer type + application stage via config.FERTILIZER_RULES.

2. ML cross-check (OPTIONAL, EXPERIMENTAL). A RandomForestClassifier
   trained on data/raw/fertilizer_data.csv. IMPORTANT: on evaluation
   this dataset showed ~14% test accuracy for 7 classes — essentially
   chance level (1/7 ≈ 14.3%) — because the numeric features do not
   meaningfully correlate with the fertilizer label in the supplied
   data. This is reported honestly rather than hidden; the UI must
   label this module as low-confidence/experimental and should not
   present it as the primary recommendation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.utils import load_pickle, load_json


def rule_based_recommendation(crop: str, values: dict) -> dict:
    """
    crop: predicted/selected crop name (must be a key in crop_profiles.json)
    values: dict with N, P, K (measured soil values)
    """
    profiles = load_json(config.CROP_PROFILES_PATH)
    if not profiles or crop not in profiles.get("crop_profiles", {}):
        return {
            "status": "Unavailable",
            "reason": f"No data-derived nutrient profile available for '{crop}'.",
        }

    crop_profile = profiles["crop_profiles"][crop]

    status = {}
    for nutrient in ["N", "P", "K"]:
        required = crop_profile[nutrient]["mean"]
        tolerance = max(crop_profile[nutrient]["std"], required * 0.1)
        actual = float(values.get(nutrient, 0))
        if actual < required - tolerance:
            status[nutrient] = "Low"
        elif actual > required + tolerance:
            status[nutrient] = "High"
        else:
            status[nutrient] = "Optimal"

    deficient = tuple(sorted(n for n in ["N", "P", "K"] if status[n] == "Low"))
    excess = [n for n in ["N", "P", "K"] if status[n] == "High"]

    if not deficient:
        recommendation = {
            "fertilizer": "None required — maintain current nutrient levels",
            "stage": "-",
        }
        overall_status = "Nutrient Excess" if excess else "Optimal"
    else:
        recommendation = config.FERTILIZER_RULES.get(
            deficient,
            {"fertilizer": "Balanced NPK Complex", "stage": "Basal dose + Top dressing"},
        )
        overall_status = "Nutrient Deficient"

    return {
        "status": overall_status,
        "nutrient_status": status,
        "deficiencies": list(deficient),
        "excesses": excess,
        "recommended_fertilizer": recommendation["fertilizer"],
        "application_stage": recommendation["stage"],
        "crop_reference_profile": crop_profile,
        "guidance": (
            "Apply the recommended fertilizer at the indicated stage. "
            "This is a decision-support estimate derived from the training "
            "dataset's crop-wise nutrient averages, not a laboratory-certified "
            "prescription. Final dosage should consider a soil lab report, "
            "crop variety, soil type, local agricultural guidance, and field conditions."
        ),
    }


class FertilizerMLPredictor:
    """Optional ML cross-check — see module docstring for the accuracy caveat."""

    def __init__(self):
        self.model = load_pickle(config.FERTILIZER_MODEL_PATH, "Fertilizer ML model")
        self.encoders = load_pickle(config.FERTILIZER_ENCODERS_PATH, "Fertilizer encoders")
        self.metadata = load_json(config.FERTILIZER_METADATA_PATH) or {}

    @property
    def accuracy(self):
        return self.metadata.get("accuracy")

    @property
    def soil_types(self):
        return self.metadata.get("soil_types", [])

    @property
    def crop_types(self):
        return self.metadata.get("crop_types", [])

    def predict(self, temperature, humidity, moisture, nitrogen, potassium, phosphorous, soil_type, crop_type):
        cols = config.FERTILIZER_NUMERIC_FEATURES + config.FERTILIZER_CATEGORICAL_FEATURES
        row = [temperature, humidity, moisture, nitrogen, potassium, phosphorous]
        try:
            soil_enc = self.encoders["soil_type"].transform([soil_type])[0]
            crop_enc = self.encoders["crop_type"].transform([crop_type])[0]
        except ValueError as e:
            return {"error": str(e)}

        X = pd.DataFrame([row + [soil_enc, crop_enc]], columns=cols)
        pred = self.model.predict(X)[0]
        fertilizer = self.encoders["__target__"].inverse_transform([pred])[0]

        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = round(float(np.max(proba)) * 100, 1)

        return {
            "fertilizer": fertilizer,
            "confidence": confidence,
            "model_test_accuracy": self.accuracy,
        }
