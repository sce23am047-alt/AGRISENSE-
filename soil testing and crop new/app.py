"""
app.py
AgriSense AI — AI-Based Soil Testing and Crop Recommendation System
with Precision Fertilizer Optimization.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
import config
from dashboard.styles import CUSTOM_CSS
from dashboard.components import (
    metric_card, status_class, badge, warning_box, system_status_sidebar,
    icon_metric_card, nutrient_bar, confidence_gauge,
)
from src.validation import validate_soil_input, VALID_RANGES
from src.soil_analysis import analyze_soil, soil_health_score
from src.utils import crop_model_ready, fertilizer_model_ready, load_json
from src.history import save_analysis, get_history, clear_history
from src.reports import build_report
from src.weather import get_forecast, weather_configured
from src.translations import t, LANGUAGE_OPTIONS

st.set_page_config(page_title="AgriSense AI", page_icon="🌾", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

for key in ["soil_values", "crop_result", "soil_result", "fertilizer_result", "ai_summary", "location"]:
    if key not in st.session_state:
        st.session_state[key] = None


def get_predictor():
    if "predictor" not in st.session_state:
        from src.prediction import CropPredictor
        st.session_state["predictor"] = CropPredictor()
    return st.session_state["predictor"]


def get_ml_fertilizer():
    if "ml_fertilizer" not in st.session_state:
        from src.fertilizer import FertilizerMLPredictor
        st.session_state["ml_fertilizer"] = FertilizerMLPredictor()
    return st.session_state["ml_fertilizer"]


def build_ai_summary(soil_result, crop_result, fert_result):
    parts = []
    deficiencies = soil_result.get("deficiencies", [])
    excesses = soil_result.get("excesses", [])
    if deficiencies:
        parts.append(f"The soil analysis indicates that {', '.join(deficiencies)} "
                      f"{'is' if len(deficiencies) == 1 else 'are'} below the data-derived optimal range.")
    if excesses:
        parts.append(f"{', '.join(excesses)} {'is' if len(excesses) == 1 else 'are'} above the optimal range.")
    if not deficiencies and not excesses:
        parts.append("Nitrogen, phosphorus and potassium levels are within the optimal range for this soil sample.")

    if crop_result and crop_result.get("primary"):
        conf_text = f" with a confidence of {crop_result['confidence']}%" if crop_result.get("confidence") is not None else ""
        parts.append(f"Based on the trained Random Forest crop recommendation model, "
                      f"the most suitable crop is {crop_result['primary'].upper()}{conf_text}.")
        if crop_result.get("low_confidence"):
            parts.append("This is a low-confidence prediction — consider laboratory soil testing "
                          "or additional field measurements before acting on it.")

    if fert_result and fert_result.get("status"):
        if fert_result["status"] == "Nutrient Deficient":
            parts.append(f"The fertilizer recommendation engine suggests addressing the identified "
                          f"deficiencies with {fert_result.get('recommended_fertilizer', 'a suitable fertilizer')}.")
        elif fert_result["status"] == "Optimal":
            parts.append("No fertilizer intervention is currently indicated.")

    return " ".join(parts)


st.sidebar.markdown("## 🌾 AgriSense AI")
st.sidebar.caption("AI-Powered Soil Intelligence")

lang_choice = st.sidebar.selectbox("🌐 Language / ಭಾಷೆ / भाषा", list(LANGUAGE_OPTIONS.keys()))
lang = LANGUAGE_OPTIONS[lang_choice]

PAGES = [
    "Dashboard", "Soil Analysis", "Crop Recommendation",
    "Fertilizer Optimization", "Weather", "Reports", "History", "About Project",
]
PAGE_LABELS = {
    "Dashboard": f"📊 {t('nav_dashboard', lang)}",
    "Soil Analysis": f"🌱 {t('nav_soil_analysis', lang)}",
    "Crop Recommendation": f"🌾 {t('nav_crop_recommendation', lang)}",
    "Fertilizer Optimization": f"🧪 {t('nav_fertilizer', lang)}",
    "Weather": f"🌦️ {t('nav_weather', lang)}",
    "Reports": f"📄 {t('nav_reports', lang)}",
    "History": f"🕘 {t('nav_history', lang)}",
    "About Project": f"ℹ️ {t('nav_about', lang)}",
}
page = st.sidebar.radio("Navigation", PAGES, format_func=lambda p: PAGE_LABELS[p], label_visibility="collapsed")

system_status_sidebar(
    model_ok=crop_model_ready(),
    fert_ok=fertilizer_model_ready(),
    db_ok=True,
)

st.title(f"🌾 {t('app_title', lang)}")
st.caption(t("app_caption", lang))

if not crop_model_ready():
    st.error("ML model not found. Please run `python training/train_model.py` before using this dashboard.")
    st.stop()

metadata = load_json(config.CROP_METADATA_PATH) or {}

# ===========================================================================
# PAGE: DASHBOARD
# ===========================================================================
if page == "Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        icon_metric_card("🤖", "ML Model", "Ready", "teal")
    with c2:
        icon_metric_card("🌱", "Soil Analysis", "Ready", "green")
    with c3:
        icon_metric_card("🎯", "Recommendation Engine", "Ready", "amber")
    with c4:
        icon_metric_card("📊", "Model Accuracy", f"{metadata.get('accuracy', 0)*100:.1f}%", "slate")

    st.markdown("---")

    if st.session_state["soil_result"] is None:
        st.info(t("no_analysis_yet", lang))
    else:
        soil_result = st.session_state["soil_result"]
        crop_result = st.session_state["crop_result"]
        fert_result = st.session_state["fertilizer_result"]

        col1, col2, col3, col4 = st.columns(4)
        icon_metric_card("🌾", "Soil Health", soil_result["label"], "green", col1)
        icon_metric_card("🌽", "Recommended Crop", crop_result["primary"].upper() if crop_result else "-", "teal", col2)
        conf = f"{crop_result['confidence']}%" if crop_result and crop_result.get("confidence") is not None else "-"
        icon_metric_card("🎯", "Confidence", conf, "amber", col3)
        n_def = len(soil_result.get("deficiencies", []))
        icon_metric_card("⚠️", "Nutrient Status", f"{n_def} Deficienc{'y' if n_def == 1 else 'ies'}",
                          "coral" if n_def else "green", col4)

        st.markdown("### AI Decision Summary")
        st.write(st.session_state["ai_summary"])

# ===========================================================================
# PAGE: SOIL ANALYSIS
# ===========================================================================
elif page == "Soil Analysis":
    st.markdown(f"### {t('soil_analysis_heading', lang)}")

    input_mode = st.radio("Input Mode", ["Manual Input", "IoT Sensor (Simulated)"], horizontal=True)

    if input_mode == "IoT Sensor (Simulated)":
        st.info("No hardware sensor is connected in this deployment. Enter simulated sensor readings below; "
                "`src/utils.py` is structured so `get_sensor_readings()` can later be wired to Arduino/ESP32/MQTT/REST.")

    with st.form("soil_input_form"):
        location = st.text_input("Farm Location (optional)", value=st.session_state.get("location") or "")
        c1, c2, c3 = st.columns(3)
        with c1:
            n_val = st.number_input("Nitrogen — N", min_value=0.0, max_value=300.0, value=80.0, step=1.0)
            p_val = st.number_input("Phosphorus — P", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
        with c2:
            k_val = st.number_input("Potassium — K", min_value=0.0, max_value=300.0, value=40.0, step=1.0)
            ph_val = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with c3:
            temp_val = st.number_input("Temperature (°C)", min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
            moisture_val = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

        submitted = st.form_submit_button(t("analyze_button", lang), width="stretch")

    with st.expander("📷 Upload Soil Image (Computer Vision)"):
        st.file_uploader("Drag & drop a soil image (JPG/PNG)", type=["jpg", "jpeg", "png"])
        st.caption("Image-based soil classification is planned as a future enhancement because the "
                   "current trained dataset does not contain sufficient image-labelled data.")

    if submitted:
        values = {"N": n_val, "P": p_val, "K": k_val, "temperature": temp_val, "moisture": moisture_val, "ph": ph_val}
        errors = validate_soil_input(values)
        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Analyzing soil... Running ML prediction... Checking nutrient levels... "
                             "Generating fertilizer recommendation... Preparing results..."):
                soil_result = soil_health_score(values)
                soil_result.update(analyze_soil(values))

                predictor = get_predictor()
                crop_result = predictor.predict(values)

                from src.fertilizer import rule_based_recommendation
                fert_result = rule_based_recommendation(crop_result.get("primary"), values) if "primary" in crop_result else {}

                ai_summary = build_ai_summary(soil_result, crop_result, fert_result)

                st.session_state["soil_values"] = values
                st.session_state["soil_result"] = soil_result
                st.session_state["crop_result"] = crop_result
                st.session_state["fertilizer_result"] = fert_result
                st.session_state["ai_summary"] = ai_summary
                st.session_state["location"] = location

                save_analysis({
                    **values,
                    "location": location,
                    "soil_health_score": soil_result["score"],
                    "soil_health_label": soil_result["label"],
                    "recommended_crop": crop_result.get("primary"),
                    "confidence": crop_result.get("confidence"),
                    "fertilizer_status": fert_result.get("status"),
                    "recommended_fertilizer": fert_result.get("recommended_fertilizer"),
                })
            st.success("Analysis complete — see results below, or check the Dashboard / Crop Recommendation / "
                       "Fertilizer Optimization pages.")

    if st.session_state["soil_result"] is not None:
        soil_result = st.session_state["soil_result"]
        crop_result = st.session_state["crop_result"]
        fert_result = st.session_state["fertilizer_result"]

        st.markdown("---")
        st.markdown("### 🌱 Soil Analysis Result")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Soil Health Score", f"{soil_result['score']} / 100 — {soil_result['label']}")
        with c2:
            st.markdown("**Nutrient Analysis**")
            sv = st.session_state["soil_values"]
            nutrient_bar("Nitrogen (N)", sv["N"], soil_result["status"].get("N", "Optimal"), max_value=150)
            nutrient_bar("Phosphorus (P)", sv["P"], soil_result["status"].get("P", "Optimal"), max_value=150)
            nutrient_bar("Potassium (K)", sv["K"], soil_result["status"].get("K", "Optimal"), max_value=150)
            nutrient_bar("pH", sv["ph"], soil_result["status"].get("ph", "Optimal"), max_value=14)

        fig = go.Figure(go.Bar(
            x=["Nitrogen", "Phosphorus", "Potassium"],
            y=[st.session_state["soil_values"]["N"], st.session_state["soil_values"]["P"], st.session_state["soil_values"]["K"]],
            marker_color=["#2D6A4F", "#40916C", "#74C69D"],
        ))
        fig.update_layout(title="NPK Levels", height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width="stretch")

        st.markdown("### 🌾 AI Crop Recommendation")
        st.markdown(f"**Primary Crop: {crop_result['primary'].upper()}**")
        if crop_result.get("confidence") is not None:
            st.progress(min(crop_result["confidence"] / 100, 1.0))
            st.caption(f"Confidence: {crop_result['confidence']}% "
                       f"({'High Confidence' if not crop_result['low_confidence'] else 'Low Confidence'})")
            if crop_result["low_confidence"]:
                warning_box("Low confidence prediction. Consider laboratory soil testing or additional field measurements.")
        for alt in crop_result.get("alternatives", []):
            badge(f"{alt['crop']} — {alt['match']}%")

        st.markdown("### 💧 Precision Fertilizer")
        st.markdown(f"**Status:** {fert_result.get('status', '-')}")
        st.markdown(f"**Recommended Fertilizer:** {fert_result.get('recommended_fertilizer', '-')}")
        st.markdown(f"**Application Stage:** {fert_result.get('application_stage', '-')}")
        st.caption(fert_result.get("guidance", ""))

        st.markdown("### 🧠 AI Decision Summary")
        st.write(st.session_state["ai_summary"])

# ===========================================================================
# PAGE: CROP RECOMMENDATION
# ===========================================================================
elif page == "Crop Recommendation":
    if st.session_state["crop_result"] is None:
        st.info("Run an analysis on the **Soil Analysis** page first.")
    else:
        crop_result = st.session_state["crop_result"]
        st.markdown(f"## Primary Crop: {crop_result['primary'].upper()}")
        if crop_result.get("confidence") is not None:
            st.plotly_chart(confidence_gauge(crop_result["confidence"], "Model Confidence"), width="stretch")

        crop_name = crop_result["primary"].capitalize()
        calendar = config.CROP_SOWING_CALENDAR.get(crop_name)
        if calendar:
            st.markdown(f"### {t('sowing_calendar', lang)}")
            cc1, cc2, cc3 = st.columns(3)
            metric_card(t("season", lang), calendar["season"], cc1)
            metric_card(t("sowing_window", lang), calendar["sowing_window"], cc2)
            metric_card(t("harvest_window", lang), calendar["harvest_window"], cc3)
            st.caption("Standard agronomic reference — not derived from the trained ML model.")

        st.markdown("### Alternative Crops")
        if crop_result.get("alternatives"):
            df = pd.DataFrame(crop_result["alternatives"]).rename(columns={"crop": "Crop", "match": "Match %"})
            fig = go.Figure(go.Bar(x=df["Match %"], y=df["Crop"], orientation="h", marker_color="#40916C"))
            fig.update_layout(height=250, margin=dict(t=10, b=20))
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("This model does not expose class probabilities for alternatives.")

        st.markdown("### Input Parameters Summary")
        st.table(pd.DataFrame([st.session_state["soil_values"]]))

        with st.expander("Model details"):
            st.json({
                "model": metadata.get("model_name"),
                "baseline_model": metadata.get("baseline_model"),
                "test_accuracy": metadata.get("accuracy"),
                "f1_score": metadata.get("f1_score"),
                "trained_on": metadata.get("dataset"),
                "training_date": metadata.get("training_date"),
            })

# ===========================================================================
# PAGE: FERTILIZER OPTIMIZATION
# ===========================================================================
elif page == "Fertilizer Optimization":
    st.markdown("### Precision Fertilizer Prescription")

    if st.session_state["fertilizer_result"] is None:
        st.info("Run an analysis on the **Soil Analysis** page first to get the rule-based recommendation below.")
    else:
        fert_result = st.session_state["fertilizer_result"]
        c1, c2, c3 = st.columns(3)
        metric_card("Status", fert_result.get("status", "-"), c1)
        metric_card("Recommended Fertilizer", fert_result.get("recommended_fertilizer", "-"), c2)
        metric_card("Application Stage", fert_result.get("application_stage", "-"), c3)

        st.markdown("#### Nutrient Status vs. Data-Derived Crop Requirement")
        ns = fert_result.get("nutrient_status", {})
        for nutrient, status in ns.items():
            st.markdown(f"- {nutrient}: **{status}**")

        st.caption(fert_result.get("guidance", ""))
        warning_box("Fertilizer recommendations are decision-support estimates. Final dosage should consider "
                    "soil laboratory reports, crop variety, soil type, local agricultural guidance, and field conditions.")

    st.markdown("---")
    st.markdown("### 🧪 Optional: ML Fertilizer Cross-Check (Experimental)")

    if not fertilizer_model_ready():
        st.warning("Fertilizer ML model not found. Run `python training/train_fertilizer_model.py`.")
    else:
        ml_fert = get_ml_fertilizer()
        acc = ml_fert.accuracy
        st.markdown(
            f'<div class="warning-box">This model was trained on data/raw/fertilizer_data.csv and achieves '
            f'only <b>{acc*100:.1f}% test accuracy</b> across {len(ml_fert.metadata.get("classes", []))} fertilizer '
            f'classes (random chance ≈ {100/len(ml_fert.metadata.get("classes", [1])):.1f}%). The numeric features in '
            f'this dataset do not meaningfully correlate with the fertilizer label, so treat this output as '
            f'experimental — the rule-based recommendation above is the primary guidance.</div>',
            unsafe_allow_html=True,
        )
        with st.form("ml_fert_form"):
            fc1, fc2 = st.columns(2)
            with fc1:
                soil_type = st.selectbox("Soil Type", ml_fert.soil_types)
                crop_type = st.selectbox("Crop Type", ml_fert.crop_types)
                humidity = st.number_input("Humidity (%)", 0.0, 100.0, 55.0)
            with fc2:
                moisture2 = st.number_input("Moisture (%) ", 0.0, 100.0, 40.0)
                temp2 = st.number_input("Temperature (°C) ", -10.0, 60.0, 28.0)
            n2 = st.number_input("Nitrogen", 0.0, 300.0, 20.0)
            k2 = st.number_input("Potassium", 0.0, 300.0, 5.0)
            p2 = st.number_input("Phosphorous", 0.0, 300.0, 15.0)
            run_ml = st.form_submit_button("Run ML Cross-Check")

        if run_ml:
            result = ml_fert.predict(temp2, humidity, moisture2, n2, k2, p2, soil_type, crop_type)
            if "error" in result:
                st.error(result["error"])
            else:
                st.markdown(f"**Predicted Fertilizer:** {result['fertilizer']}  "
                             f"({result['confidence']}% model confidence, {result['model_test_accuracy']*100:.1f}% overall test accuracy)")

# ===========================================================================
# PAGE: WEATHER
# ===========================================================================
elif page == "Weather":
    st.markdown(f"### {t('weather_heading', lang)}")
    st.caption("Fetched in real time from OpenWeatherMap. This is live external data, "
               "not an AI prediction from the crop/fertilizer models.")

    if not weather_configured():
        st.warning("WEATHER_API_KEY is not set. Add a free API key from openweathermap.org "
                   "to your `.env` file as `WEATHER_API_KEY=your_key_here`, then restart the app.")
    else:
        default_city = st.session_state.get("location") or config.WEATHER_DEFAULT_CITY
        city = st.text_input(t("city_location", lang), value=default_city)
        if st.button(t("get_forecast", lang)):
            with st.spinner("Fetching live forecast..."):
                forecast = get_forecast(city)
            if "error" in forecast:
                st.error(forecast["error"])
            else:
                st.markdown(f"#### 5-Day Forecast — {forecast['city']}")
                cols = st.columns(len(forecast["days"]))
                for col, day in zip(cols, forecast["days"]):
                    with col:
                        metric_card(day["date"], f"{day['temp']}°C")
                        st.caption(f"{day['condition']} · {day['humidity']}% humidity")

# ===========================================================================
# PAGE: REPORTS
# ===========================================================================
elif page == "Reports":
    st.markdown("### Generate Soil Analysis Report")
    if st.session_state["soil_result"] is None:
        st.info("Run an analysis on the **Soil Analysis** page first.")
    else:
        if st.button("📄 Generate PDF Report"):
            try:
                pdf_bytes = build_report(
                    soil_values=st.session_state["soil_values"],
                    soil_analysis={**st.session_state["soil_result"]},
                    crop_result=st.session_state["crop_result"],
                    fertilizer_result=st.session_state["fertilizer_result"],
                    ai_summary=st.session_state["ai_summary"],
                    location=st.session_state.get("location") or "",
                )
                st.download_button(
                    "⬇ Download Report (PDF)", data=pdf_bytes,
                    file_name="agrisense_soil_report.pdf", mime="application/pdf",
                )
            except Exception as e:
                st.error("Unable to generate the report.")
                st.exception(e)

# ===========================================================================
# PAGE: HISTORY  (Farm Analytics stats strip + multi-panel trend grid)
# ===========================================================================
elif page == "History":
    st.markdown(f"### {t('analysis_history', lang)}")
    records = get_history()
    if not records:
        st.info("No previous analyses yet.")
    else:
        df = pd.DataFrame(records)

        st.markdown("### 📊 Farm Analytics")
        m1, m2, m3, m4 = st.columns(4)
        icon_metric_card("📋", "Total Analyses", str(len(df)), "teal", m1)
        avg_health = round(df["soil_health_score"].mean()) if "soil_health_score" in df.columns else "-"
        icon_metric_card("🌱", "Avg Soil Health", f"{avg_health}/100", "green", m2)
        top_crop = (df["recommended_crop"].mode()[0]
                    if "recommended_crop" in df.columns and not df["recommended_crop"].mode().empty else "-")
        icon_metric_card("🌾", "Most Recommended", str(top_crop).upper(), "amber", m3)
        deficient_pct = (round((df["fertilizer_status"] == "Nutrient Deficient").mean() * 100)
                         if "fertilizer_status" in df.columns else 0)
        icon_metric_card("🧪", "Deficient Samples", f"{deficient_pct}%", "coral" if deficient_pct else "green", m4)

        st.dataframe(df, width="stretch")

        st.markdown("### 📈 Trends & Breakdown")
        st.caption("All charts below are built directly from your saved analysis history — no external or fabricated data.")

        x_seq = list(range(1, len(df) + 1))

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if "recommended_crop" in df.columns:
                crop_counts = df["recommended_crop"].value_counts()
                fig = go.Figure(go.Bar(x=crop_counts.index, y=crop_counts.values, marker_color="#40916C"))
                fig.update_layout(title="Crop Recommendation Frequency", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch")
        with r1c2:
            if "soil_health_score" in df.columns:
                fig = go.Figure(go.Scatter(x=x_seq, y=df["soil_health_score"], mode="lines+markers",
                                            line=dict(color="#2D6A4F"), fill="tozeroy",
                                            fillcolor="rgba(45,106,79,0.08)"))
                fig.update_layout(title="Soil Health Score Trend", xaxis_title="Analysis #",
                                   yaxis_title="Score", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch")

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if all(c in df.columns for c in ["n", "p", "k"]):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x_seq, y=df["n"], mode="lines+markers", name="N", line=dict(color="#2D6A4F")))
                fig.add_trace(go.Scatter(x=x_seq, y=df["p"], mode="lines+markers", name="P", line=dict(color="#40916C")))
                fig.add_trace(go.Scatter(x=x_seq, y=df["k"], mode="lines+markers", name="K", line=dict(color="#74C69D")))
                fig.update_layout(title="N / P / K Trend", xaxis_title="Analysis #", height=300,
                                   margin=dict(t=40, b=20), legend=dict(orientation="h", y=-0.25))
                st.plotly_chart(fig, width="stretch")
            else:
                st.caption("N/P/K columns not found in history.")
        with r2c2:
            if "moisture" in df.columns:
                fig = go.Figure(go.Scatter(x=x_seq, y=df["moisture"], mode="lines+markers",
                                            line=dict(color="#3182CE")))
                fig.update_layout(title="Moisture Trend (%)", xaxis_title="Analysis #",
                                   yaxis_title="Moisture %", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch")

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            if "temperature" in df.columns:
                fig = go.Figure(go.Scatter(x=x_seq, y=df["temperature"], mode="lines+markers",
                                            line=dict(color="#DD6B20")))
                fig.update_layout(title="Temperature Trend (°C)", xaxis_title="Analysis #",
                                   yaxis_title="°C", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch")
        with r3c2:
            if "recommended_fertilizer" in df.columns:
                fert_counts = df["recommended_fertilizer"].value_counts()
                fig = go.Figure(go.Bar(x=fert_counts.index, y=fert_counts.values, marker_color="#B45309"))
                fig.update_layout(title="Fertilizer Recommendation Frequency", height=300, margin=dict(t=40, b=20))
                st.plotly_chart(fig, width="stretch")

        st.markdown("### 🌾 Yield Prediction")
        st.info("Yield prediction requires a historical harvest/yield dataset — crop, soil conditions, "
                "and actual outcomes over past seasons. That dataset isn't part of this project yet, so "
                "rather than show an invented number, this module is a placeholder until real data is available.")

        if st.button(t("clear_history", lang)):
            clear_history()
            st.rerun()

# ===========================================================================
# PAGE: ABOUT PROJECT
# ===========================================================================
elif page == "About Project":
    st.markdown("## About Project")
    st.markdown(f"""
**Project:** AI-Based Soil Testing and Crop Recommendation System with Precision Fertilizer Optimization
(originally proposed as *Soil Testing and Crop Recommendation Robot*)

**Purpose:** AI-assisted precision agriculture and farm decision support.

**Core Technologies:** Python, scikit-learn, Streamlit, IoT-ready architecture (MQTT/REST-ready)

**ML Model:** {metadata.get('model_name', 'Random Forest')} baseline
(compared against Decision Tree, Gradient Boosting and Gaussian Naive Bayes)

**Major Modules:**
- Soil Data Acquisition & Validation
- AI-Based Crop Recommendation
- Precision Fertilizer Optimization (rule-based + experimental ML cross-check)
- Result & Decision Support
- Dashboard & Report Generation

**Dataset-supported parameters:** {", ".join(config.CROP_FEATURES)}

**Documented future IoT-sensor parameters (in the PPT, not in the current dataset):**
{", ".join(config.FUTURE_SENSOR_PARAMETERS)}

**Not yet implemented (documented, not fabricated):** live IoT hardware connectivity, cloud storage,
mobile application, live weather API, image-based soil classification, laboratory-certified dosage.
""")