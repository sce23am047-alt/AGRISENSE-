"""
dashboard/styles.py
Professional dashboard theme for AgriSense AI.
Palette: deep forest green + warm neutral surface + amber/emerald accents.

Note: this is a visual redesign only. It does not add any fake
"live"/"syncing" indicators — this app computes on-demand (button
click → result), not via continuous streaming, so the UI reflects
that honestly rather than imitating always-on dashboards.
"""

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

:root {
    --forest: #14312A;
    --forest-2: #1B4332;
    --emerald: #2D6A4F;
    --emerald-light: #40916C;
    --lime: #74C69D;
    --amber: #B45309;
    --amber-soft: #FEF3E2;
    --danger: #B91C1C;
    --bg-soft: #F4F7F5;
    --surface: #FFFFFF;
    --border: #E4EAE5;
    --text-muted: #5C6B60;
    --radius: 14px;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, sans-serif;
}

.stApp {
    background-color: var(--bg-soft);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--forest) 0%, var(--forest-2) 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] * {
    color: #EAF3EC !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
section[data-testid="stSidebar"] .stRadio label {
    padding: 0.35rem 0.6rem;
    border-radius: 8px;
    transition: background 0.15s ease;
}
section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(255,255,255,0.06);
}

/* ---------- Headings ---------- */
h1, h2, h3 {
    color: var(--forest);
    font-weight: 700;
    letter-spacing: -0.01em;
}
h1 { font-size: 2rem; }

/* ---------- Metric cards ---------- */
.metric-card {
    background: var(--surface);
    border-radius: var(--radius);
    padding: 1.15rem 1.35rem;
    box-shadow: 0 1px 2px rgba(20,49,42,0.04), 0 4px 14px rgba(20,49,42,0.06);
    border: 1px solid var(--border);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(20,49,42,0.06), 0 10px 24px rgba(20,49,42,0.10);
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.65rem;
    font-weight: 800;
    color: var(--forest);
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: -0.02em;
}

/* ---------- Status colors ---------- */
.status-good { color: var(--emerald-light); font-weight: 700; }
.status-fair { color: var(--amber); font-weight: 700; }
.status-attention { color: var(--danger); font-weight: 700; }

/* ---------- Badges / pills ---------- */
.badge {
    display: inline-block;
    padding: 0.22rem 0.75rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    background: #E7F3EC;
    color: var(--emerald);
    margin-right: 0.35rem;
    margin-bottom: 0.3rem;
    border: 1px solid rgba(45,106,79,0.15);
}

/* ---------- Warning box ---------- */
.warning-box {
    background: var(--amber-soft);
    border-left: 4px solid var(--amber);
    padding: 0.8rem 1.1rem;
    border-radius: 8px;
    color: #7C2D12;
    font-size: 0.9rem;
    line-height: 1.5;
}

/* ---------- Buttons ---------- */
.stButton>button {
    background: linear-gradient(180deg, var(--emerald-light) 0%, var(--emerald) 100%);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.65rem 1.5rem;
    font-weight: 600;
    box-shadow: 0 2px 6px rgba(45,106,79,0.25);
    transition: transform 0.1s ease, box-shadow 0.15s ease;
}
.stButton>button:hover {
    background: linear-gradient(180deg, var(--emerald) 0%, var(--forest-2) 100%);
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 10px rgba(45,106,79,0.32);
}

/* ---------- System status panel (sidebar) ---------- */
.status-panel-title {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #B7CFC0 !important;
    margin: 0.4rem 0 0.5rem 0;
}
.status-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.3rem 0;
    font-size: 0.88rem;
}
.status-dot-ok, .status-dot-bad {
    width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0;
}
.status-dot-ok { background: #52D68C; box-shadow: 0 0 6px rgba(82,214,140,0.6); }
.status-dot-bad { background: #E5484D; box-shadow: 0 0 6px rgba(229,72,77,0.6); }

/* ---------- Dataframe / tables ---------- */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
}

/* ---------- Icon metric cards (colored variants) ---------- */
.icon-card {
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    color: white;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    display: flex;
    align-items: center;
    gap: 0.9rem;
}
.icon-card .icon-badge {
    font-size: 1.6rem;
    width: 46px; height: 46px;
    border-radius: 12px;
    background: rgba(255,255,255,0.22);
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.icon-card .icon-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.9;
}
.icon-card .icon-value {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}
.icon-card-teal   { background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%); }
.icon-card-amber  { background: linear-gradient(135deg, #D97706 0%, #92400E 100%); }
.icon-card-green  { background: linear-gradient(135deg, #40916C 0%, #2D6A4F 100%); }
.icon-card-coral  { background: linear-gradient(135deg, #DC6B4F 0%, #B4442E 100%); }
.icon-card-slate  { background: linear-gradient(135deg, #475569 0%, #1E293B 100%); }

/* ---------- Nutrient progress bars ---------- */
.nutrient-row {
    margin-bottom: 0.85rem;
}
.nutrient-row .nutrient-top {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    margin-bottom: 0.3rem;
}
.nutrient-row .nutrient-name {
    font-weight: 600;
    color: var(--forest);
}
.nutrient-row .nutrient-val {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: var(--text-muted);
}
.nutrient-track {
    width: 100%;
    height: 9px;
    border-radius: 999px;
    background: #E4EAE5;
    overflow: hidden;
}
.nutrient-fill {
    height: 100%;
    border-radius: 999px;
}
.nutrient-fill-good { background: linear-gradient(90deg, var(--emerald-light), var(--emerald)); }
.nutrient-fill-fair { background: linear-gradient(90deg, #F59E0B, var(--amber)); }
.nutrient-fill-bad  { background: linear-gradient(90deg, #F87171, var(--danger)); }

</style>
"""