"""
src/utils.py
Small shared helpers used across the src/ and dashboard/ modules.
"""

import json
import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_pickle(path: Path, friendly_name: str):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{friendly_name} not found. Please run training/train_model.py."
        )
    return joblib.load(path)


def load_json(path: Path):
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)


def crop_model_ready() -> bool:
    return all(
        Path(p).exists()
        for p in [config.CROP_MODEL_PATH, config.CROP_SCALER_PATH, config.CROP_LABEL_ENCODER_PATH]
    )


def fertilizer_model_ready() -> bool:
    return all(
        Path(p).exists()
        for p in [config.FERTILIZER_MODEL_PATH, config.FERTILIZER_ENCODERS_PATH]
    )


def status_dot(ok: bool) -> str:
    return "🟢" if ok else "🔴"


def get_sensor_readings():
    """
    Placeholder abstraction for live IoT sensor input.

    No hardware (Arduino / ESP32 / Raspberry Pi) or MQTT/REST broker is
    connected in this deployment. This function intentionally raises
    rather than returning fabricated readings, so the dashboard never
    silently displays fake "live" sensor data. Wire this up to an
    MQTT subscriber or a REST poll once real hardware is available.
    """
    raise NotImplementedError(
        "No IoT sensor is connected. Wire get_sensor_readings() to your "
        "MQTT broker or REST endpoint once hardware is available."
    )
