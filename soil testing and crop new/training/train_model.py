"""
training/train_model.py

Trains the AI crop-recommendation model on data/raw/crop_data.csv.

Run:
    python training/train_model.py

What it does:
  1. Loads + validates + cleans the dataset
  2. Splits / scales
  3. Trains Random Forest (project baseline, per the PPT methodology)
     plus Decision Tree, Gradient Boosting and Gaussian Naive Bayes for
     comparison (these are the model families referenced in the
     literature survey)
  4. Selects the best-performing model on the held-out test set while
     keeping Random Forest as the reported baseline
  5. Saves model, scaler, label encoder and metadata to models/
  6. Computes a data-derived per-crop nutrient profile (mean/std of
     N, P, K, pH, moisture, temperature per crop) from the TRAINING
     split only, and saves it to models/crop_profiles.json. This is
     what src/fertilizer.py uses instead of any invented thresholds.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.preprocessing import (
    load_dataset,
    clean_dataset,
    prepare_crop_features,
    split_dataset,
    scale_features,
    encode_labels,
)


def build_candidate_models():
    return {
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200, random_state=config.RANDOM_STATE
        ),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=config.RANDOM_STATE),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=config.RANDOM_STATE
        ),
        "GaussianNB": GaussianNB(),
    }


def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, preds, average="weighted", zero_division=0)), 4),
    }


def compute_crop_profiles(df):
    """Data-derived per-crop nutrient/parameter profile (mean + std)."""
    df = df.rename(columns={"pH": "ph"})
    profiles = {}
    for crop, group in df.groupby(config.CROP_TARGET):
        profiles[crop] = {
            feat: {
                "mean": round(float(group[feat].mean()), 2),
                "std": round(float(group[feat].std()), 2),
            }
            for feat in config.CROP_FEATURES
        }
    return profiles


def compute_general_thresholds(df):
    """33rd/66th percentile bands per feature -> Low / Optimal / High cutoffs."""
    df = df.rename(columns={"pH": "ph"})
    thresholds = {}
    for feat in config.CROP_FEATURES:
        low_cut = round(float(df[feat].quantile(0.33)), 2)
        high_cut = round(float(df[feat].quantile(0.66)), 2)
        thresholds[feat] = {"low_below": low_cut, "high_above": high_cut}
    return thresholds


def main():
    print("=" * 60)
    print("AgriSense AI — Crop Recommendation Model Training")
    print("=" * 60)

    print(f"\n[1/6] Loading dataset from {config.CROP_DATASET_PATH}")
    df_raw = load_dataset(config.CROP_DATASET_PATH)
    print(f"  Loaded {len(df_raw)} rows, columns: {list(df_raw.columns)}")

    print("\n[2/6] Cleaning dataset")
    df = clean_dataset(df_raw)
    print(f"  {len(df)} rows remain after cleaning.")

    print("\n[3/6] Preparing features + splitting")
    X, y = prepare_crop_features(df)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    print("\n[4/6] Scaling features + encoding labels")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    y_train_enc, y_test_enc, label_encoder = encode_labels(y_train, y_test)

    print("\n[5/6] Training + comparing candidate models")
    candidates = build_candidate_models()
    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(X_train_scaled, y_train_enc)
        metrics = evaluate(model, X_test_scaled, y_test_enc)
        results[name] = metrics
        fitted[name] = model
        print(f"  {name:28s} acc={metrics['accuracy']:.4f}  f1={metrics['f1_score']:.4f}")

    best_name = max(results, key=lambda n: results[n]["f1_score"])
    baseline_name = "RandomForestClassifier"
    print(f"\n  Best by F1-score: {best_name}")
    print(f"  Project baseline (per PPT methodology): {baseline_name}")

    # Keep Random Forest as the deployed baseline unless another model is
    # clearly and substantially better (per project spec, section 20).
    deployed_name = baseline_name
    if results[best_name]["f1_score"] - results[baseline_name]["f1_score"] > 0.03:
        deployed_name = best_name
        print(f"  {best_name} outperforms the baseline by >3% F1 — deploying {best_name} instead.")

    deployed_model = fitted[deployed_name]

    print(f"\n[6/6] Saving artifacts for deployed model: {deployed_name}")
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed_model, config.CROP_MODEL_PATH)
    joblib.dump(scaler, config.CROP_SCALER_PATH)
    joblib.dump(label_encoder, config.CROP_LABEL_ENCODER_PATH)

    metadata = {
        "model_name": deployed_name,
        "baseline_model": baseline_name,
        "features": config.CROP_FEATURES,
        "target": config.CROP_TARGET,
        "classes": list(label_encoder.classes_),
        "accuracy": results[deployed_name]["accuracy"],
        "precision": results[deployed_name]["precision"],
        "recall": results[deployed_name]["recall"],
        "f1_score": results[deployed_name]["f1_score"],
        "model_comparison": results,
        "training_date": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(config.CROP_DATASET_PATH.name),
        "dataset_rows": len(df),
        "future_sensor_parameters_not_in_dataset": config.FUTURE_SENSOR_PARAMETERS,
    }
    with open(config.CROP_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    profiles = {
        "crop_profiles": compute_crop_profiles(df.iloc[X_train.index]),
        "general_thresholds": compute_general_thresholds(df.iloc[X_train.index]),
        "note": "Derived from the training split of data/raw/crop_data.csv. "
        "Used as a transparent, data-driven reference — not a laboratory-certified standard.",
    }
    with open(config.CROP_PROFILES_PATH, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"\nSaved:\n  {config.CROP_MODEL_PATH}\n  {config.CROP_SCALER_PATH}\n  "
          f"{config.CROP_LABEL_ENCODER_PATH}\n  {config.CROP_METADATA_PATH}\n  {config.CROP_PROFILES_PATH}")
    print("\nDone. Run 'python training/evaluate_model.py' for a detailed report,")
    print("or 'streamlit run app.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
