import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.fertilizer import rule_based_recommendation
from src.utils import load_json
import config


def get_a_known_crop():
    profiles = load_json(config.CROP_PROFILES_PATH)
    return list(profiles["crop_profiles"].keys())[0]


def test_low_nitrogen_triggers_deficiency():
    crop = get_a_known_crop()
    profile = load_json(config.CROP_PROFILES_PATH)["crop_profiles"][crop]
    values = {
        "N": max(profile["N"]["mean"] - profile["N"]["std"] * 5, 0),
        "P": profile["P"]["mean"],
        "K": profile["K"]["mean"],
    }
    result = rule_based_recommendation(crop, values)
    assert "N" in result["deficiencies"]
    assert result["status"] == "Nutrient Deficient"


def test_optimal_nutrients_require_no_fertilizer():
    crop = get_a_known_crop()
    profile = load_json(config.CROP_PROFILES_PATH)["crop_profiles"][crop]
    values = {"N": profile["N"]["mean"], "P": profile["P"]["mean"], "K": profile["K"]["mean"]}
    result = rule_based_recommendation(crop, values)
    assert result["deficiencies"] == []


def test_unknown_crop_returns_unavailable():
    result = rule_based_recommendation("NotARealCrop", {"N": 50, "P": 50, "K": 50})
    assert result["status"] == "Unavailable"


if __name__ == "__main__":
    test_low_nitrogen_triggers_deficiency()
    test_optimal_nutrients_require_no_fertilizer()
    test_unknown_crop_returns_unavailable()
    print("All fertilizer tests passed.")
