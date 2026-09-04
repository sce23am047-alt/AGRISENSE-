"""
training/evaluate_model.py

Loads the already-trained crop model and prints a detailed evaluation
report (accuracy, precision, recall, F1, confusion matrix, per-class
report). Does NOT retrain — run training/train_model.py first.

Run:
    python training/evaluate_model.py
"""

import sys
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from training.preprocessing import load_dataset, clean_dataset, prepare_crop_features, split_dataset


def main():
    for p in [config.CROP_MODEL_PATH, config.CROP_SCALER_PATH, config.CROP_LABEL_ENCODER_PATH]:
        if not p.exists():
            print(f"ML model not found ({p.name}). Please run training/train_model.py.")
            sys.exit(1)

    model = joblib.load(config.CROP_MODEL_PATH)
    scaler = joblib.load(config.CROP_SCALER_PATH)
    label_encoder = joblib.load(config.CROP_LABEL_ENCODER_PATH)

    df = clean_dataset(load_dataset(config.CROP_DATASET_PATH))
    X, y = prepare_crop_features(df)
    _, X_test, _, y_test = split_dataset(X, y)

    X_test_scaled = scaler.transform(X_test)
    y_test_enc = label_encoder.transform(y_test)
    preds = model.predict(X_test_scaled)

    print("Model Evaluation")
    print("-" * 16)
    print(f"Accuracy:  {accuracy_score(y_test_enc, preds):.4f}")
    print(f"Precision: {precision_score(y_test_enc, preds, average='weighted', zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test_enc, preds, average='weighted', zero_division=0):.4f}")
    print(f"F1 Score:  {f1_score(y_test_enc, preds, average='weighted', zero_division=0):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_enc, preds))

    print("\nClassification Report:")
    print(classification_report(y_test_enc, preds, target_names=label_encoder.classes_, zero_division=0))


if __name__ == "__main__":
    main()
