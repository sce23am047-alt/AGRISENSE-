import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.validation import validate_soil_input, is_valid


def valid_sample():
    return {"N": 80, "P": 45, "K": 40, "temperature": 25, "moisture": 60, "ph": 6.5}


def test_valid_input_has_no_errors():
    assert validate_soil_input(valid_sample()) == []
    assert is_valid(valid_sample())


def test_missing_field_is_rejected():
    values = valid_sample()
    values["N"] = ""
    errors = validate_soil_input(values)
    assert any("Nitrogen" in e for e in errors)


def test_invalid_ph_is_rejected():
    values = valid_sample()
    values["ph"] = 15
    errors = validate_soil_input(values)
    assert any("pH" in e for e in errors)


def test_negative_npk_is_rejected():
    values = valid_sample()
    values["K"] = -5
    errors = validate_soil_input(values)
    assert any("Potassium" in e for e in errors)


def test_non_numeric_value_is_rejected():
    values = valid_sample()
    values["temperature"] = "warm"
    errors = validate_soil_input(values)
    assert any("Temperature" in e for e in errors)


if __name__ == "__main__":
    test_valid_input_has_no_errors()
    test_missing_field_is_rejected()
    test_invalid_ph_is_rejected()
    test_negative_npk_is_rejected()
    test_non_numeric_value_is_rejected()
    print("All validation tests passed.")
