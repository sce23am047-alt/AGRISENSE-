"""
dashboard/components.py
Small reusable Streamlit rendering helpers used by app.py.

Function signatures are unchanged from the previous version, so this
is a drop-in replacement — app.py does not need any changes.
"""

import streamlit as st


def metric_card(label: str, value: str, col=None):
    target = col if col is not None else st
    target.markdown(
        f"""<div class="metric-card">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def status_class(label: str) -> str:
    mapping = {
        "GOOD": "status-good",
        "FAIR": "status-fair",
        "NEEDS ATTENTION": "status-attention",
        "Optimal": "status-good",
        "Low": "status-fair",
        "High": "status-fair",
    }
    return mapping.get(label, "")


def badge(text: str):
    st.markdown(f'<span class="badge">{text}</span>', unsafe_allow_html=True)


def warning_box(text: str):
    st.markdown(f'<div class="warning-box">⚠ {text}</div>', unsafe_allow_html=True)


def icon_metric_card(icon: str, label: str, value: str, variant: str = "teal", col=None):
    """
    Colored, icon-based metric card (SaaS-dashboard style).
    variant: one of 'teal', 'amber', 'green', 'coral', 'slate'.
    """
    target = col if col is not None else st
    target.markdown(
        f"""<div class="icon-card icon-card-{variant}">
                <div class="icon-badge">{icon}</div>
                <div>
                    <div class="icon-label">{label}</div>
                    <div class="icon-value">{value}</div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def nutrient_bar(name: str, value: float, status: str = "Optimal", max_value: float = 150.0, col=None):
    """
    A labeled progress bar for a single nutrient/parameter reading.
    'status' controls the fill color: Optimal/Good -> green,
    Low/High/Fair -> amber, anything else -> red.
    """
    target = col if col is not None else st
    pct = max(0.0, min(100.0, (value / max_value) * 100 if max_value else 0))
    fill_class = "nutrient-fill-good"
    if status in ("Low", "High", "Fair"):
        fill_class = "nutrient-fill-fair"
    elif status in ("Critical", "Needs Attention"):
        fill_class = "nutrient-fill-bad"
    target.markdown(
        f"""<div class="nutrient-row">
                <div class="nutrient-top">
                    <span class="nutrient-name">{name}</span>
                    <span class="nutrient-val">{value} &middot; {status}</span>
                </div>
                <div class="nutrient-track">
                    <div class="nutrient-fill {fill_class}" style="width:{pct:.0f}%"></div>
                </div>
            </div>""",
        unsafe_allow_html=True,
    )


def confidence_gauge(value: float, title: str = "Confidence"):
    """
    A donut-style gauge for a 0-100 confidence/score value, built with
    Plotly (already a project dependency) — shows the REAL value passed
    in, not a fabricated animation.
    """
    import plotly.graph_objects as go
    color = "#40916C" if value >= 70 else ("#D97706" if value >= 50 else "#DC2626")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 34, "family": "JetBrains Mono"}},
        title={"text": title, "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "#F0F4F1",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(height=220, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def system_status_sidebar(model_ok: bool, fert_ok: bool, db_ok: bool):
    """
    Real status indicators only — no fabricated 'live'/'syncing' state.
    Each dot reflects an actual check (model file loaded, DB reachable),
    not a simulated real-time connection.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown('<div class="status-panel-title">System Status</div>', unsafe_allow_html=True)

    def _row(ok: bool, label: str):
        dot_class = "status-dot-ok" if ok else "status-dot-bad"
        st.sidebar.markdown(
            f'<div class="status-row"><span class="{dot_class}"></span>{label}</div>',
            unsafe_allow_html=True,
        )

    _row(model_ok, "Crop ML Model")
    _row(fert_ok, "Fertilizer Engine")
    _row(db_ok, "History Database")