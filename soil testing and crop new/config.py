"""
config.py
Central configuration for AgriSense AI.
All paths, feature lists, and tunable thresholds live here so nothing
is hardcoded inside the dashboard or the ML code.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
CROP_DATASET_PATH = BASE_DIR / "data" / "raw" / "crop_data.csv"
FERTILIZER_DATASET_PATH = BASE_DIR / "data" / "raw" / "fertilizer_data.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Model artifact paths
# ---------------------------------------------------------------------------
MODEL_DIR = BASE_DIR / "models"

CROP_MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "crop_model.pkl"))
CROP_SCALER_PATH = Path(os.getenv("SCALER_PATH", MODEL_DIR / "scaler.pkl"))
CROP_LABEL_ENCODER_PATH = Path(
    os.getenv("LABEL_ENCODER_PATH", MODEL_DIR / "label_encoder.pkl")
)
CROP_METADATA_PATH = MODEL_DIR / "model_metadata.json"
CROP_PROFILES_PATH = MODEL_DIR / "crop_profiles.json"

FERTILIZER_MODEL_PATH = MODEL_DIR / "fertilizer_model.pkl"
FERTILIZER_ENCODERS_PATH = MODEL_DIR / "fertilizer_encoders.pkl"
FERTILIZER_METADATA_PATH = MODEL_DIR / "fertilizer_metadata.json"

# ---------------------------------------------------------------------------
# Database / reports / history
# ---------------------------------------------------------------------------
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "data" / "agrisense.db"))
REPORTS_DIR = BASE_DIR / "reports"

# ---------------------------------------------------------------------------
# Crop model — dataset-supported features only
# Source: data/raw/crop_data.csv (matches PPT sensors: NPK + pH + moisture + temperature)
# ---------------------------------------------------------------------------
CROP_FEATURES = ["N", "P", "K", "temperature", "moisture", "ph"]
CROP_TARGET = "crop"

# These parameters are mentioned in the project PPT/proposal but are NOT
# present in the supplied dataset. They are documented here as future
# IoT-sensor parameters rather than silently fabricated.
FUTURE_SENSOR_PARAMETERS = ["humidity", "rainfall"]

# ---------------------------------------------------------------------------
# Fertilizer model — dataset-supported features only
# Source: data/raw/fertilizer_data.csv
# ---------------------------------------------------------------------------
FERTILIZER_NUMERIC_FEATURES = [
    "temperature",
    "humidity",
    "moisture",
    "nitrogen",
    "potassium",
    "phosphorous",
]
FERTILIZER_CATEGORICAL_FEATURES = ["soil_type", "crop_type"]
FERTILIZER_TARGET = "fertilizer_name"

# Maps a predicted crop (from the crop model) to the closest matching
# "Crop Type" category available in the fertilizer dataset. This is a
# documented category mapping, not fabricated data — Ragi (a millet) is
# mapped to the Millets category and Rice to Paddy, its dataset label.
CROP_TO_FERTILIZER_CROP_TYPE = {
    "Maize": "Maize",
    "Wheat": "Wheat",
    "Rice": "Paddy",
    "Ragi": "Millets",
}

# ---------------------------------------------------------------------------
# Sowing calendar (rule-based, standard Indian agricultural seasons)
# Source: standard agronomic reference (Kharif/Rabi classification) —
# NOT derived from the training dataset, NOT an AI prediction.
# ---------------------------------------------------------------------------
CROP_SOWING_CALENDAR = {
    "Maize": {
        "season": "Kharif (also grown Rabi in irrigated areas)",
        "sowing_window": "June - July",
        "harvest_window": "September - October",
    },
    "Wheat": {
        "season": "Rabi",
        "sowing_window": "November - December",
        "harvest_window": "March - April",
    },
    "Rice": {
        "season": "Kharif",
        "sowing_window": "June - July",
        "harvest_window": "November - December",
    },
    "Ragi": {
        "season": "Kharif",
        "sowing_window": "June - July",
        "harvest_window": "October - November",
    },
}

# ---------------------------------------------------------------------------
# Soil health scoring weights (AI/Rule-Based Soil Health Indicator)
# Transparent, configurable — NOT a laboratory-certified measurement.
# ---------------------------------------------------------------------------
SOIL_HEALTH_WEIGHTS = {
    "N": 0.2,
    "P": 0.2,
    "K": 0.2,
    "ph": 0.25,
    "moisture": 0.15,
}

# Ideal pH band used for scoring (standard agronomic reference range)
IDEAL_PH_RANGE = (6.0, 7.5)

# ---------------------------------------------------------------------------
# Fertilizer rule-engine mapping (deficiency -> recommended fertilizer)
# Labeled in the UI as decision-support estimates, not lab prescriptions.
# ---------------------------------------------------------------------------
FERTILIZER_RULES = {
    ("N",): {"fertilizer": "Urea", "stage": "Top dressing"},
    ("P",): {"fertilizer": "DAP (Di-Ammonium Phosphate)", "stage": "Basal dose"},
    ("K",): {"fertilizer": "MOP (Muriate of Potash)", "stage": "Basal dose"},
    ("N", "P"): {"fertilizer": "NPK 20-20-0 / DAP + Urea", "stage": "Basal dose + Top dressing"},
    ("N", "K"): {"fertilizer": "NPK 20-0-20 / Urea + MOP", "stage": "Basal dose + Top dressing"},
    ("P", "K"): {"fertilizer": "NPK 0-20-20 / DAP + MOP", "stage": "Basal dose"},
    ("N", "P", "K"): {"fertilizer": "Balanced NPK Complex (e.g. 17-17-17)", "stage": "Basal dose + Top dressing"},
}

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
APP_ENV = os.getenv("APP_ENV", "development")
RANDOM_STATE = 42
TEST_SIZE = 0.2
LOW_CONFIDENCE_THRESHOLD = 60.0  # percent

# ---------------------------------------------------------------------------
# Live weather (real external API — OpenWeatherMap free tier)
# This is fetched data, NOT a trained model prediction. Always labeled as
# "Live Forecast" in the UI to keep it clearly separate from the ML outputs.
# ---------------------------------------------------------------------------
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"
WEATHER_DEFAULT_CITY = os.getenv("WEATHER_DEFAULT_CITY", "Bengaluru")