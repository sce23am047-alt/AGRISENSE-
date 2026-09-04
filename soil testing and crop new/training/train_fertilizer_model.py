"""
training/train_fertilizer_model.py

Trains a Random Forest classifier on data/raw/fertilizer_data.csv to
predict a fertilizer name from soil/crop conditions. This is the
data-driven half of the Precision Fertilizer Optimization module — it
runs alongside (not instead of) the transparent rule-based deficiency
engine in src/fertilizer.py.

Run:
    python training/train_fertilizer_model.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_fertilizer_features,
    split_dataset,
)


def main():
    print("=" * 60)
    print("AgriSense AI — Fertilizer Prediction Model Training")
    print("=" * 60)

    print(f"\n[1/5] Loading dataset from {config.FERTILIZER_DATASET_PATH}")
    df_raw = load_dataset(config.FERTILIZER_DATASET_PATH)
    print(f"  Loaded {len(df_raw)} rows, columns: {list(df_raw.columns)}")

    print("\n[2/5] Cleaning dataset")
    df = clean_dataset(df_raw)
    print(f"  {len(df)} rows remain after cleaning.")

    print("\n[3/5] Preparing features + encoding categoricals")
    X, y = prepare_fertilizer_features(df)

    cat_encoders = {}
    for col in config.FERTILIZER_CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        cat_encoders[col] = le

    target_encoder = LabelEncoder()
    y_enc = target_encoder.fit_transform(y)
    cat_encoders["__target__"] = target_encoder

    X_train, X_test, y_train, y_test = split_dataset(X, y_enc)

    print("\n[4/5] Training RandomForestClassifier")
    # NOTE: max_depth/min_samples_leaf are capped deliberately. With no real
    # signal between these features and the fertilizer label (see accuracy
    # below), an unconstrained Random Forest grows enormous, fully-overfit
    # trees that just memorize noise — that produced a 130+MB artifact for
    # no accuracy gain. Capping depth keeps the model small and honest
    # without changing its (already chance-level) real-world accuracy.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=config.RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
    }
    print(f"  accuracy={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    print("\n[5/5] Saving artifacts")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FERTILIZER_MODEL_PATH)
    joblib.dump(cat_encoders, config.FERTILIZER_ENCODERS_PATH)

    metadata = {
        "model_name": "RandomForestClassifier",
        "features_numeric": config.FERTILIZER_NUMERIC_FEATURES,
        "features_categorical": config.FERTILIZER_CATEGORICAL_FEATURES,
        "target": config.FERTILIZER_TARGET,
        "classes": list(target_encoder.classes_),
        "soil_types": list(cat_encoders["soil_type"].classes_),
        "crop_types": list(cat_encoders["crop_type"].classes_),
        **metrics,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(config.FERTILIZER_DATASET_PATH.name),
        "dataset_rows": len(df),
    }
    with open(config.FERTILIZER_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved:\n  {config.FERTILIZER_MODEL_PATH}\n  {config.FERTILIZER_ENCODERS_PATH}\n  {config.FERTILIZER_METADATA_PATH}")


if __name__ == "__main__":
    main()
