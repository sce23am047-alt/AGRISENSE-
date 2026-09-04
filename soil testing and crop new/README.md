# AgriSense AI

**AI-Based Soil Testing and Crop Recommendation System with Precision Fertilizer Optimization**

Built from: your project PPT (*Soil Testing and Crop Recommendation Robot*), your master-prompt
specification, and three supplied datasets. Scoped to the PPT's actual proposed system —
Soil Data Acquisition → Soil Analysis → AI Crop Recommendation → Precision Fertilizer
Optimization → Dashboard & Report Generation — with no fabricated modules.

---

## 1. Project Overview

Traditional soil testing is slow, expensive, and often leads to the wrong crop or fertilizer
choice. AgriSense AI takes manually-entered (or future IoT-sensed) NPK, pH, moisture and
temperature readings and returns an AI-generated crop recommendation, a nutrient deficiency
analysis, and a fertilizer suggestion — all from models trained on your own datasets, not
hardcoded values.

## 2. Problem Statement

- Manual soil testing is slow and requires lab access.
- Farmers often select crops or fertilizer without data-backed guidance.
- Over- or under-fertilization wastes money and degrades soil over time.
- No simple, low-cost decision-support tool ties soil condition → crop choice → fertilizer
  action together.

## 3. Objectives

- Reduce dependence on slow manual soil testing.
- Recommend the most suitable crop from real soil parameters.
- Identify nutrient deficiencies transparently.
- Suggest an appropriate fertilizer response and application stage.
- Present all of this through a clear, non-technical dashboard.

## 4. Dataset Description — read this before touching the code

You supplied three CSV files. They do **not** share a common schema, so rather than merging
them into something that half-fabricates data, each is used for what it actually supports:

| File | Used as | Why |
|---|---|---|
| `data/raw/crop_data.csv` (from `maize_ragi_wheat_rice_crop_recommendation.csv`) | **Primary crop model training data** | Its columns — N, P, K, pH, moisture, temperature — are an exact match for the sensors your PPT describes. 4000 rows, 4 balanced classes (Maize, Ragi, Rice, Wheat). |
| `data/raw/fertilizer_data.csv` (from `data_core.csv`) | **Fertilizer model training data** | Real fertilizer-prediction dataset (Soil Type, Crop Type, N/P/K, Temp, Humidity, Moisture → Fertilizer Name). 8000 rows. |
| `data/raw/crop_data_alternate_UNUSED.csv` (from `crop_recommendation_20_281_29.csv`) | **Not used for training** | Uses humidity + rainfall instead of moisture — a genuine schema mismatch against the sensors in your PPT. Kept for reference; see the adjacent `.README.txt`. |

**Crops supported:** Maize, Ragi, Rice, Wheat — this is what your actual dataset supports.
If you later get a dataset with more crop labels on the same N/P/K/pH/moisture/temperature
schema, retraining is a one-command operation (`python training/train_model.py`).

## 5. Machine Learning Workflow

1. **Crop model** (`training/train_model.py`): loads → cleans (drops nulls/duplicates) →
   splits 80/20 → scales → trains **Random Forest** (your PPT's specified model) alongside
   Decision Tree, Gradient Boosting and Gaussian Naive Bayes for comparison → keeps Random
   Forest as the deployed model since it also won on the held-out test set → saves artifacts
   + a **data-derived per-crop nutrient profile** (`models/crop_profiles.json`) used by the
   fertilizer engine.
2. **Fertilizer model** (`training/train_fertilizer_model.py`): trains a Random Forest on
   `fertilizer_data.csv`. See the honest result in §7 below.

### Actual results (from this run — not projected)

| Model | Accuracy | F1 |
|---|---|---|
| **Random Forest (deployed)** | **88.6%** | **88.6%** |
| Gradient Boosting | 87.6% | 87.6% |
| Decision Tree | 83.4% | 83.3% |
| Gaussian Naive Bayes | 80.1% | 80.2% |

Re-run `python training/evaluate_model.py` any time for the full confusion matrix and
per-class report on your machine.

## 6. Fertilizer Optimization — how it actually works

Two components, clearly separated in the dashboard:

- **Rule-based engine (primary — matches your PPT's Module 4 methodology).** Compares your
  entered N/P/K against the *actual training-data average* for the recommended crop
  (`models/crop_profiles.json` — computed from real data, not invented thresholds), flags
  Low/Optimal/High, and maps the deficiency pattern to a fertilizer type (Urea / DAP / MOP /
  NPK blend) and an application stage (Basal dose / Top dressing) via `config.FERTILIZER_RULES`.
- **ML cross-check (optional, clearly labeled "experimental" in the UI).**

## 7. An honest limitation you should know about

The fertilizer *ML* model, trained on your `fertilizer_data.csv`, scores **~13.3% test
accuracy across 7 fertilizer classes** — essentially chance level (1/7 ≈ 14.3%). I checked
class-wise nutrient means and the Crop-Type × Fertilizer-Name distribution directly: this
particular dataset simply does not contain a learnable relationship between the numeric
features and the fertilizer label, for any model. This isn't a bug — it's reported here
instead of quietly showing an inflated number, per your own master prompt's "never fabricate
accuracy" rule. It's why the **rule-based engine is the primary fertilizer recommendation**,
and the ML model is boxed off as an experimental cross-check with its real accuracy displayed
next to every prediction.

## 8. Technology Stack

Python 3.12 · Streamlit · pandas · numpy · scikit-learn · joblib · plotly · matplotlib ·
reportlab (PDF reports) · SQLite (history) · python-dotenv

## 9. Folder Structure

```
AI_Soil_Testing_Crop_Recommendation/
├── app.py                     # Streamlit dashboard (entry point)
├── config.py                  # All paths, feature lists, thresholds — nothing hardcoded elsewhere
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/                   # crop_data.csv, fertilizer_data.csv, + unused alt. dataset
│   ├── processed/
│   └── sample/
├── models/                    # crop_model.pkl, fertilizer_model.pkl, metadata (generated)
├── training/
│   ├── preprocessing.py       # load / validate / clean / split / scale
│   ├── train_model.py         # crop model training + comparison
│   ├── train_fertilizer_model.py
│   └── evaluate_model.py      # detailed metrics + confusion matrix
├── src/
│   ├── validation.py          # input range checks
│   ├── prediction.py          # CropPredictor — load model, predict, confidence, alternatives
│   ├── soil_analysis.py       # nutrient status + soil health score (rule-based, data-derived)
│   ├── fertilizer.py          # rule engine + optional ML cross-check
│   ├── history.py             # SQLite read/write for the History page
│   ├── reports.py             # PDF report generation
│   └── utils.py                # shared helpers, get_sensor_readings() IoT placeholder
├── dashboard/
│   ├── styles.py               # CSS (forest green / emerald / amber theme)
│   └── components.py           # metric cards, badges, status dots
├── tests/
│   ├── test_validation.py
│   ├── test_prediction.py
│   └── test_fertilizer.py
└── reports/                    # generated PDFs land here at runtime
```

## 10. Installation & Setup (Windows / VS Code)

```bash
cd AI_Soil_Testing_Crop_Recommendation
python --version
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

macOS/Linux: replace `.venv\Scripts\activate` with `source .venv/bin/activate`.

## 11. Train the Models

```bash
python training/train_model.py
python training/train_fertilizer_model.py
python training/evaluate_model.py
```

This creates everything under `models/` (`.pkl` files + metadata JSON). The app will refuse
to start meaningfully without these — it checks and tells you exactly what's missing.

## 12. Run the Dashboard

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## 13. Run the Tests

```bash
python -m pytest tests/ -v
```

12 tests covering validation edge cases, model loading/prediction shape, and the fertilizer
rule engine. All passing against the trained artifacts in this build.

## 14. Example Workflow

1. Open **Soil Analysis**, enter N/P/K, temperature, moisture, pH (or use the defaults).
2. Click **Analyze Soil & Generate AI Recommendation**.
3. See soil health score, nutrient status, recommended crop + confidence + alternatives,
   and the fertilizer recommendation — all on one page.
4. Check **Dashboard** for the summary cards, **Crop Recommendation** for the detail view,
   **Fertilizer Optimization** for the rule-based result plus the optional ML cross-check.
5. Go to **Reports** to download a PDF, or **History** to see everything you've run so far.

## 15. Troubleshooting

| Message | Fix |
|---|---|
| `ML model not found` | Run `python training/train_model.py` |
| `Dataset not found` | Confirm the CSVs are present under `data/raw/` |
| `Fertilizer ML model not found` | Run `python training/train_fertilizer_model.py` |
| Validation errors on submit | Values must be within the ranges shown — see `src/validation.py` |

## 16. Limitations (stated plainly, not buried)

- Crop coverage is limited to Maize, Ragi, Rice, Wheat — bounded by the supplied dataset.
- No live IoT hardware is connected — `src/utils.py:get_sensor_readings()` is a documented
  stub ready for MQTT/REST wiring, not a live feed.
- No live weather API — `WEATHER_API_KEY` in `.env.example` is unused and reserved.
- No image-based soil classification — no image-labelled data was supplied.
- Fertilizer *dosage* (exact kg/acre) is not provided — the dataset has no dosage column, so
  the system recommends fertilizer *type* and *application stage* only.
- The fertilizer ML cross-check has ~14% accuracy (see §7) — treat it as experimental, not
  a recommendation source.

## 17. Future Enhancements (per the PPT's own future-work section)

Further IoT sensor integration, mobile application, drone/satellite monitoring,
multilingual support, market-price prediction, and government-scheme integration. None of
these are implemented in this build — they're listed here for continuity, not implied.

## License

Academic / educational project.
#   s o i l _ t e s t i n g  
 