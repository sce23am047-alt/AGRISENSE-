"""
src/validation.py
Validates soil-input values before they ever reach the ML model.
"""

VALID_RANGES = {
    "N": (0, 300, "Nitrogen (N)"),
    "P": (0, 300, "Phosphorus (P)"),
    "K": (0, 300, "Potassium (K)"),
    "temperature": (-10, 60, "Temperature (°C)"),
    "moisture": (0, 100, "Moisture (%)"),
    "ph": (0, 14, "Soil pH"),
    "humidity": (0, 100, "Humidity (%)"),
}


def validate_soil_input(values: dict) -> list:
    """
    Validates a dict of soil parameter values.
    Returns a list of user-friendly error strings (empty list = valid).
    """
    errors = []
    for key, value in values.items():
        if key not in VALID_RANGES:
            continue
        low, high, label = VALID_RANGES[key]

        if value is None or value == "":
            errors.append(f"⚠ {label} is required.")
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"⚠ {label} must be a numeric value.")
            continue

        if value < low or value > high:
            errors.append(f"⚠ Please enter a valid {label} between {low} and {high}.")

    return errors


def is_valid(values: dict) -> bool:
    return len(validate_soil_input(values)) == 0
