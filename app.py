import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyTrimmer(BaseEstimator, TransformerMixin):
    def __init__(self, max_categories=20):
        self.max_categories = max_categories
        self.top_categories_ = {}
    def fit(self, X, y=None): return self
    def transform(self, X): return X

st.set_page_config(
    page_title="Bladder Cancer Recurrence Risk Predictor",
    page_icon="⊕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* ── Global ── */
  [data-testid="stAppViewContainer"] { background: #f0f4f8; }
  [data-testid="stHeader"]           { background: transparent; }
  [data-testid="stSidebar"]          { display: none; }

  /* ── Strip all column top-padding so cards align perfectly ── */
  [data-testid="stColumn"] > div:first-child {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
  [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:first-child {
    margin-top: 0 !important;
  }

  /* ── Header banner ── */
  .app-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2b6cb0 100%);
    color: white;
    padding: 2rem 2.5rem 1.8rem;
    border-radius: 14px;
    margin-bottom: 1.6rem;
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
  }
  .app-header-icon {
    flex-shrink: 0;
    margin-top: .15rem;
    opacity: .92;
  }
  .app-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0 0 .45rem;
    letter-spacing: -.02em;
    line-height: 1.15;
  }
  .app-header p  { font-size: 1.1rem; margin: 0; opacity: .82; line-height: 1.5; }

  /* ── Cards ── */
  .card {
    background: white;
    border-radius: 14px;
    padding: 1.6rem 1.8rem 1.8rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
  }
  /* left column: gap between title card and the inputs below */
  [data-testid="stColumn"]:first-child .card {
    margin-bottom: 1.4rem;
  }
  .card-title {
    display: flex;
    align-items: center;
    gap: .5rem;
    font-size: .92rem;
    font-weight: 700;
    color: #1a3a5c;
    margin-bottom: 1.2rem;
    padding-bottom: .65rem;
    border-bottom: 2px solid #e2e8f0;
    text-transform: uppercase;
    letter-spacing: .07em;
  }

  /* ── Input labels: darker, more legible ── */
  [data-testid="stWidgetLabel"] p,
  [data-testid="stWidgetLabel"] {
    color: #2d3748 !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
  }
  [data-testid="stNumberInput"] input {
    font-size: 1.05rem !important;
  }

  /* ── Input row: boxed with alternating stripe ── */
  [data-testid="stNumberInput"],
  [data-testid="stSelectbox"] {
    border-radius: 8px;
    padding: 0.55rem 0.75rem !important;
    margin-bottom: 0.5rem !important;
    border: 1px solid #d1dae6;
  }
  /* odd rows */
  [data-testid="stNumberInput"],
  [data-testid="stSelectbox"]:nth-of-type(odd) {
    background: #e8eef5;
  }
  /* even rows */
  [data-testid="stSelectbox"]:nth-of-type(even) {
    background: #dde5f0;
  }

  /* ── Label-to-input gap inside number input ── */
  [data-testid="stNumberInput"] [data-testid="stWidgetLabel"] {
    margin-bottom: 0.45rem !important;
  }
  [data-testid="stNumberInput"] > div:last-child {
    margin-top: 0.35rem !important;
  }

  /* ── Generate button ── */
  div.stButton > button {
    background: linear-gradient(135deg, #2b6cb0, #1a3a5c);
    color: white;
    border: none;
    border-radius: 8px;
    padding: .75rem 2.8rem;
    font-size: 1.08rem;
    font-weight: 700;
    width: 100%;
    letter-spacing: .03em;
    cursor: pointer;
    transition: opacity .2s, transform .1s;
    margin-top: .6rem;
  }
  div.stButton > button:hover  { opacity: .88; }
  div.stButton > button:active { transform: scale(.99); }

  /* ── Probability gauge ── */
  .gauge-wrap { text-align: center; padding: 1.2rem 0 .6rem; }
  .gauge-value {
    font-size: 4.4rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: .35rem;
    letter-spacing: -.03em;
  }
  .gauge-label { font-size: 1rem; color: #718096; letter-spacing: .03em; text-transform: uppercase; }

  /* ── Risk badge ── */
  .risk-badge {
    display: inline-block;
    padding: .45rem 1.4rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.05rem;
    margin: .6rem 0;
    letter-spacing: .02em;
  }
  .risk-low      { background:#c6f6d5; color:#22543d; }
  .risk-low-mid  { background:#bee3f8; color:#2a4365; }
  .risk-mid      { background:#fefcbf; color:#744210; }
  .risk-mid-high { background:#fed7aa; color:#7b341e; }
  .risk-high     { background:#fed7d7; color:#742a2a; }

  /* ── Progress bar ── */
  .prob-bar-bg {
    background: #e2e8f0;
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
    margin: .7rem 0 1.1rem;
  }
  .prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width .7s cubic-bezier(.4,0,.2,1);
  }

  /* ── Advice box (dynamic per risk level) ── */
  .advice-box {
    border-left: 4px solid;
    border-radius: 0 8px 8px 0;
    padding: .9rem 1.15rem;
    font-size: 1rem;
    margin-top: .8rem;
    line-height: 1.6;
  }
  .advice-low      { background:#f0fff4; border-color:#48bb78; color:#22543d; }
  .advice-low-mid  { background:#ebf8ff; border-color:#4299e1; color:#2c5282; }
  .advice-mid      { background:#fffff0; border-color:#d69e2e; color:#744210; }
  .advice-mid-high { background:#fffaf0; border-color:#ed8936; color:#7b341e; }
  .advice-high     { background:#fff5f5; border-color:#e53e3e; color:#742a2a; }

  /* ── Feature row ── */
  .feat-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: .42rem 0;
    font-size: .97rem;
    border-bottom: 1px solid #f0f0f0;
  }
  .feat-row:last-child { border-bottom: none; }
  .feat-key   { color: #718096; }
  .feat-value { font-weight: 600; color: #2d3748; text-align: right; max-width: 60%; }

  /* ── Placeholder (empty state) ── */
  .empty-state {
    background: #f7fafc;
    border: 1.5px dashed #cbd5e0;
    border-radius: 10px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    margin: 1.4rem 0 1rem;
  }
  .empty-state-icon {
    display: flex;
    justify-content: center;
    margin-bottom: 1rem;
    opacity: .45;
  }
  .empty-state p {
    color: #4a5568 !important;
    font-size: 1.05rem !important;
    line-height: 1.65 !important;
    margin: 0 !important;
  }
  .empty-state strong { color: #2d3748; }

  /* ── Disclaimer ── */
  .disclaimer {
    background: #fffbeb;
    border: 1px solid #f6e05e;
    border-radius: 10px;
    padding: .85rem 1.4rem;
    font-size: 0.85rem;
    color: #744210;
    margin-top: 1.2rem;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)

# ── SVG icon helpers ─────────────────────────────────────────────────────────
def svg_header():
    return """<svg class="app-header-icon" xmlns="http://www.w3.org/2000/svg"
      width="40" height="40" viewBox="0 0 24 24" fill="none"
      stroke="white" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2v-4M9 21H5a2 2 0 0 1-2-2v-4m0 0h18"/>
    </svg>"""

def svg_patient():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14"
      viewBox="0 0 24 24" fill="none" stroke="#1a3a5c"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
      <circle cx="12" cy="7" r="4"/>
    </svg>"""

def svg_chart():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14"
      viewBox="0 0 24 24" fill="none" stroke="#1a3a5c"
      stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="18" y1="20" x2="18" y2="10"/>
      <line x1="12" y1="20" x2="12" y2="4"/>
      <line x1="6"  y1="20" x2="6"  y2="14"/>
    </svg>"""

def svg_empty():
    return """<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48"
      viewBox="0 0 24 24" fill="none" stroke="#a0aec0"
      stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
      <path d="M9 5H7a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2h-2"/>
      <rect x="9" y="3" width="6" height="4" rx="2"/>
      <line x1="9" y1="12" x2="15" y2="12"/>
      <line x1="9" y1="16" x2="13" y2="16"/>
    </svg>"""

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-header">
  {svg_header()}
  <div>
    <h1>Bladder Cancer 2-Year Recurrence Risk Predictor</h1>
    <p>Enter the patient's clinical indicators to estimate individualized post-surgical recurrence probability.</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("bladder_model.joblib")

model = load_model()

# ── Two-column layout ─────────────────────────────────────────────────────────
left, right = st.columns([2, 3], gap="large")

with left:
    st.markdown(f'<div class="card"><div class="card-title">{svg_patient()} Patient Clinical Indicators</div>', unsafe_allow_html=True)

    rbc = st.number_input(
        "Urinary RBC Count (RBC#)  ×10⁶/L",
        min_value=0.0, max_value=100.0, value=5.2, step=0.1,
        help="Normal reference range: 0-8 ×10⁶/L",
    )

    malig_options = {
        "High-grade": 0,
        "Low-grade": 1,
        "Other (Benign / PUNLMP / Atypical / Dysplasia)": 2,
    }
    malig_choice = st.selectbox("Malignancy Grade", list(malig_options))
    malignancy = malig_options[malig_choice]

    infil_options = {"Non-invasive": 0, "Invasive": 1}
    infil_choice = st.selectbox("Infiltration Status", list(infil_options))
    infiltration = infil_options[infil_choice]

    shape_options = {"Papillary": 0, "Non-papillary (Cauliflower / Nodular / etc.)": 1}
    shape_choice = st.selectbox("Tumor Shape", list(shape_options))
    shape = shape_options[shape_choice]

    hydro_options = {"Absent": 0, "Present (Unilateral or Bilateral)": 1}
    hydro_choice = st.selectbox("Hydronephrosis Status", list(hydro_options))
    hydronephrosis = hydro_options[hydro_choice]

    necro_options = {"Absent": 0, "Present": 1}
    necro_choice = st.selectbox("Cystic Necrosis", list(necro_options))
    necrosis = necro_options[necro_choice]

    st.markdown("</div>", unsafe_allow_html=True)

    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_clicked = st.button("Generate Prediction")

with right:
    st.markdown(f'<div class="card"><div class="card-title">{svg_chart()} Prediction Result</div>', unsafe_allow_html=True)

    if predict_clicked:
        input_df = pd.DataFrame({
            '红细胞计数(RBC#)-尿液': [rbc],
            '恶性值': [malignancy],
            '浸润值': [infiltration],
            '形状值': [shape],
            '肾盂积水值': [hydronephrosis],
            '坏死值': [necrosis],
        })
        prob = model.predict_proba(input_df)[0, 1]
        pct = prob * 100

        if prob < 0.2:
            level, badge_cls, bar_color, advice_cls = "Low Risk", "risk-low", "#48bb78", "advice-low"
            advice = "Routine follow-up is recommended. Standard post-operative surveillance protocol is appropriate."
        elif prob < 0.4:
            level, badge_cls, bar_color, advice_cls = "Low-Medium Risk", "risk-low-mid", "#4299e1", "advice-low-mid"
            advice = "Close follow-up is recommended. Consider shortening the surveillance intervals."
        elif prob < 0.6:
            level, badge_cls, bar_color, advice_cls = "Medium Risk", "risk-mid", "#d69e2e", "advice-mid"
            advice = "Enhanced monitoring is recommended. Review adjuvant therapy options with the clinical team."
        elif prob < 0.8:
            level, badge_cls, bar_color, advice_cls = "Medium-High Risk", "risk-mid-high", "#ed8936", "advice-mid-high"
            advice = "Active intervention issuggested.A Multidisciplinary team consultation is advised."
        else:
            level, badge_cls, bar_color, advice_cls = "High Risk", "risk-high", "#e53e3e", "advice-high"
            advice = "Immediate intervention  ishighly recommended. An Urgent specialist referral is warranted."

        st.markdown(f"""
        <div class="gauge-wrap">
          <div class="gauge-value" style="color:{bar_color};">{pct:.2f}%</div>
          <div class="gauge-label">2-Year Recurrence Probability</div>
        </div>
        <div style="text-align:center;">
          <span class="risk-badge {badge_cls}">{level}</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width:{pct:.2f}%;background:{bar_color};"></div>
        </div>
        <div class="advice-box {advice_cls}">{advice}</div>
        <br>
        <div class="card-title" style="margin-top:.4rem;">{svg_patient()} Input Summary</div>
        """, unsafe_allow_html=True)

        rows = [
            ("Urinary RBC Count", f"{rbc:.1f} ×10⁶/L"),
            ("Malignancy Grade", malig_choice),
            ("Infiltration Status", infil_choice),
            ("Tumor Shape", shape_choice),
            ("Hydronephrosis", hydro_choice),
            ("Cystic Necrosis", necro_choice),
        ]
        for key, val in rows:
            st.markdown(
                f'<div class="feat-row"><span class="feat-key">{key}</span>'
                f'<span class="feat-value">{val}</span></div>',
                unsafe_allow_html=True,
            )

    else:
        st.markdown(f"""
        <div class="empty-state">
          <div class="empty-state-icon">{svg_empty()}</div>
          <p>Enter the patient indicators on the left<br>
          and click <strong>Generate Prediction</strong> to view results.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Disclaimer ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  ⚠️ <strong>Disclaimer:</strong> This tool is intended for research and educational purposes only.
  It does not constitute medical advice or a formal clinical diagnosis.
  All clinical decisions must be made by qualified healthcare professionals.
</div>
""", unsafe_allow_html=True)
