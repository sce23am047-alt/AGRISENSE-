"""
training/preprocessing.py
Modular data-loading, validation, cleaning, and preparation functions
used by the training pipeline. Nothing here is UI code.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset. Raises a clear error if the file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place the dataset in data/raw/."
        )
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame, required_columns: list) -> None:
    """Ensure the dataset actually contains the columns the model needs."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "Dataset validation failed. Missing columns: " + ", ".join(missing)
        )


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, then handle missing values and duplicates."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} row(s) with missing values.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate row(s).")
    return df


def prepare_crop_features(df: pd.DataFrame):
    """Rename raw crop-dataset columns to config.CROP_FEATURES and return X, y."""
    df = df.rename(columns={"pH": "ph"})
    validate_dataset(df, config.CROP_FEATURES + [config.CROP_TARGET])
    X = df[config.CROP_FEATURES].astype(float)
    y = df[config.CROP_TARGET].astype(str)
    return X, y


def prepare_fertilizer_features(df: pd.DataFrame):
    """Rename raw fertilizer-dataset columns to config's fertilizer schema."""
    rename_map = {
        "Temparature": "temperature",
        "Temperature": "temperature",
        "Humidity": "humidity",
        "Moisture": "moisture",
        "Soil Type": "soil_type",
        "Crop Type": "crop_type",
        "Nitrogen": "nitrogen",
        "Potassium": "potassium",
        "Phosphorous": "phosphorous",
        "Fertilizer Name": "fertilizer_name",
    }
    df = df.rename(columns=rename_map)
    required = (
        config.FERTILIZER_NUMERIC_FEATURES
        + config.FERTILIZER_CATEGORICAL_FEATURES
        + [config.FERTILIZER_TARGET]
    )
    validate_dataset(df, required)
    X = df[config.FERTILIZER_NUMERIC_FEATURES + config.FERTILIZER_CATEGORICAL_FEATURES].copy()
    y = df[config.FERTILIZER_TARGET].astype(str)
    return X, y


def split_dataset(X, y, test_size=None, stratify=True):
    test_size = test_size or config.TEST_SIZE
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=config.RANDOM_STATE,
        stratify=y if stratify else None,
    )


def scale_features(X_train, X_test=None):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    return X_train_scaled, scaler


def encode_labels(y_train, y_test=None):
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    if y_test is not None:
        y_test_enc = encoder.transform(y_test)
        return y_train_enc, y_test_enc, encoder
    return y_train_enc, encoder
