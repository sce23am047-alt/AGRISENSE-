"""
src/soil_analysis.py

Rule-based / data-derived soil analysis:
  - Nutrient status (Low / Optimal / High) per parameter, using
    percentile thresholds computed from the actual training data
    (models/crop_profiles.json) rather than invented cutoffs.
  - A transparent "AI/Rule-Based Soil Health Indicator" score (0-100).

Explicitly NOT presented as a laboratory-certified measurement.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.utils import load_json


def _get_thresholds():
    profiles = load_json(config.CROP_PROFILES_PATH)
    if profiles is None:
        return None
    return profiles.get("general_thresholds")


def nutrient_status(feature: str, value: float, thresholds: dict = None) -> str:
    """Returns 'Low', 'Optimal', or 'High' for a given feature value."""
    thresholds = thresholds or _get_thresholds()

    if feature == "ph":
        low, high = config.IDEAL_PH_RANGE
        if value < low:
            return "Acidic"
        if value > high:
            return "Alkaline"
        return "Neutral"

    if not thresholds or feature not in thresholds:
        return "Unknown"

    band = thresholds[feature]
    if value < band["low_below"]:
        return "Low"
    if value > band["high_above"]:
        return "High"
    return "Optimal"


def analyze_soil(values: dict) -> dict:
    """
    values: dict with keys N, P, K, temperature, moisture, ph
    Returns per-parameter status + list of deficiencies (Low N/P/K).
    """
    thresholds = _get_thresholds()
    status = {}
    for feat in ["N", "P", "K", "temperature", "moisture", "ph"]:
        if feat in values and values[feat] is not None:
            status[feat] = nutrient_status(feat, float(values[feat]), thresholds)

    deficiencies = [f for f in ["N", "P", "K"] if status.get(f) == "Low"]
    excesses = [f for f in ["N", "P", "K"] if status.get(f) == "High"]

    return {
        "status": status,
        "deficiencies": deficiencies,
        "excesses": excesses,
    }


def soil_health_score(values: dict) -> dict:
    """
    Transparent weighted score (0-100):
      - N, P, K: 100 if Optimal, 55 if Low/High
      - pH: 100 if within IDEAL_PH_RANGE, decays with distance from the band
      - moisture: 100 if Optimal, 55 otherwise
    Weights come from config.SOIL_HEALTH_WEIGHTS.
    """
    analysis = analyze_soil(values)
    status = analysis["status"]
    weights = config.SOIL_HEALTH_WEIGHTS

    component_scores = {}
    for feat in ["N", "P", "K", "moisture"]:
        if feat in status:
            component_scores[feat] = 100.0 if status[feat] == "Optimal" else 55.0

    if "ph" in values and values["ph"] is not None:
        low, high = config.IDEAL_PH_RANGE
        ph = float(values["ph"])
        if low <= ph <= high:
            component_scores["ph"] = 100.0
        else:
            distance = min(abs(ph - low), abs(ph - high))
            component_scores["ph"] = max(30.0, 100.0 - distance * 15.0)

    total_weight = sum(weights[f] for f in component_scores if f in weights)
    if total_weight == 0:
        return {"score": 0, "label": "Unknown"}

    score = sum(component_scores[f] * weights[f] for f in component_scores if f in weights) / total_weight
    score = round(score)

    if score >= 80:
        label = "GOOD"
    elif score >= 60:
        label = "FAIR"
    else:
        label = "NEEDS ATTENTION"

    return {"score": score, "label": label, "components": component_scores}
