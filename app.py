import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Aviation Parts Repairability Classifier",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .block-container { padding: 2rem 3rem; }

    .hero {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .hero h1 { font-size: 1.8rem; font-weight: 700; margin: 0 0 0.4rem 0; color: white; }
    .hero p { font-size: 0.95rem; color: #bfdbfe; margin: 0 0 1rem 0; line-height: 1.6; }
    .disclaimer-box {
        background: rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.82rem;
        color: #dbeafe;
    }

    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 0.72rem;
        font-weight: 700;
        color: #1d4ed8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 1rem;
    }

    .result-green {
        background: #f0fdf4;
        border: 2px solid #16a34a;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .result-red {
        background: #fff1f2;
        border: 2px solid #dc2626;
        border-radius: 14px;
        padding: 2rem;
        text-align: center;
    }
    .result-title-green { font-size: 1.6rem; font-weight: 800; color: #15803d; margin: 0.5rem 0; }
    .result-title-red { font-size: 1.6rem; font-weight: 800; color: #dc2626; margin: 0.5rem 0; }
    .result-desc { font-size: 0.88rem; color: #64748b; margin-top: 0.5rem; line-height: 1.6; }

    .metric-row { display: flex; gap: 0.75rem; margin-bottom: 0.5rem; }
    .metric-card {
        flex: 1;
        background: #f1f5f9;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-val { font-size: 1.3rem; font-weight: 700; color: #1d4ed8; display: block; }
    .metric-lbl { font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }

    .tag-pos {
        display: inline-block;
        background: #f0fdf4;
        border: 1px solid #86efac;
        color: #166534;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        margin: 0.25rem;
        font-weight: 500;
    }
    .tag-neg {
        display: inline-block;
        background: #fff1f2;
        border: 1px solid #fca5a5;
        color: #991b1b;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        margin: 0.25rem;
        font-weight: 500;
    }
    .tag-neu {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        color: #475569;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        margin: 0.25rem;
        font-weight: 500;
    }

    .bar-wrap {
        background: #e2e8f0;
        border-radius: 8px;
        height: 12px;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    .bar-green { background: linear-gradient(90deg, #16a34a, #4ade80); height: 100%; border-radius: 8px; }
    .bar-red   { background: linear-gradient(90deg, #dc2626, #f87171); height: 100%; border-radius: 8px; }
    .bar-blue  { background: linear-gradient(90deg, #1d4ed8, #60a5fa); height: 100%; border-radius: 8px; }

    .sidebar-metric {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.83rem;
    }
    .sidebar-metric:last-child { border-bottom: none; }
    .s-key { color: #94a3b8; }
    .s-val { color: #1e293b; font-weight: 600; }

    .stButton > button {
        background: #1d4ed8 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        padding: 0.65rem !important;
    }
    .stButton > button:hover { background: #1e40af !important; }

    [data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0; }

    .footer {
        margin-top: 2rem;
        padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
        text-align: center;
        color: #94a3b8;
        font-size: 0.78rem;
        line-height: 1.8;
    }
    .footer a { color: #1d4ed8; text-decoration: none; }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Train model ───────────────────────────────────────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 500
    data = {
        'material_type': np.random.choice(['SP','T','C','M','N'], n, p=[0.35,0.15,0.25,0.15,0.10]),
        'has_pma': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'is_tool': np.random.choice([0, 1], n, p=[0.85, 0.15]),
        'is_fixed_asset': np.random.choice([0, 1], n, p=[0.8, 0.2]),
        'ata_chapter': np.random.choice(['00-00','32-00','49-00','72-00','97-97','99-01'], n),
        'sales_price_sar': np.round(np.random.exponential(scale=50, size=n), 2),
        'weight_kg': np.round(np.random.exponential(scale=4, size=n), 2),
    }
    df = pd.DataFrame(data)
    def classify(row):
        score = 0
        if row['sales_price_sar'] > 40: score += 1
        if row['has_pma'] == 1: score += 1
        if row['material_type'] in ['C','SP']: score += 1
        if row['is_tool'] == 1: score -= 2
        return 1 if score >= 2 else 0
    df['is_repairable'] = df.apply(classify, axis=1)
    np.random.seed(7)
    noise = np.random.random(n) < 0.08
    df.loc[noise, 'is_repairable'] = 1 - df.loc[noise, 'is_repairable']
    missing = np.random.random(n) < 0.05
    df.loc[missing, 'has_pma'] = np.nan
    df['has_pma'] = df['has_pma'].fillna(df['has_pma'].mode()[0])
    X = df.drop('is_repairable', axis=1)
    y = df['is_repairable']
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['material_type','ata_chapter']),
        ('num', 'passthrough', ['has_pma','is_tool','is_fixed_asset','sales_price_sar','weight_kg'])
    ])
    pipe = Pipeline([('preprocessor', preprocessor),
                     ('classifier', RandomForestClassifier(n_estimators=100,
                                                           class_weight='balanced',
                                                           random_state=42))])
    pipe.fit(X_train, y_train)
    return pipe

model = train_model()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Performance")
    for key, val in [
        ("Algorithm", "Random Forest"),
        ("Test Accuracy", "88%"),
        ("CV F1 Score", "0.891 ± 0.023"),
        ("Training Parts", "400"),
        ("Validation", "5-Fold Stratified"),
        ("Data", "Synthetic"),
    ]:
        st.markdown(f"""
        <div class="sidebar-metric">
            <span class="s-key">{key}</span>
            <span class="s-val">{val}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Class Recall")

    st.markdown("**Unrepairable** — 95%")
    st.markdown("""<div class="bar-wrap"><div class="bar-blue" style="width:95%"></div></div>""",
                unsafe_allow_html=True)
    st.markdown("**Repairable** — 76%")
    st.markdown("""<div class="bar-wrap"><div class="bar-blue" style="width:76%"></div></div>""",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub Repository](https://github.com/raghaddbae/aviation-parts-repairability-classifier)")
    st.markdown("[👤 LinkedIn](https://www.linkedin.com/in/raghadbaeshen)")
    st.markdown("[🚀 Live App](https://aviation-parts-classifier.streamlit.app/)")

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>✈️ Aviation Parts Repairability Classifier</h1>
    <p>A machine learning proof-of-concept that predicts whether an aircraft part should be 
    classified as <strong>repairable</strong> or <strong>unrepairable</strong>, 
    based on its operational attributes. Trained on synthetic data modeled on real MRO logic.</p>
    <div class="disclaimer-box">
        ⚠️ <strong>Demonstration only.</strong> All data is synthetic and randomly generated. 
        This does not represent any real company, system, or dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Presets ───────────────────────────────────────────────
st.markdown("**🧪 Try a quick example:**")
pc1, pc2, pc3, pc4 = st.columns(4)
preset = None
with pc1:
    if st.button("💰 Expensive Component"): preset = {'material_type':'C','has_pma':1,'is_tool':0,'is_fixed_asset':0,'ata_chapter':'72-00','sales_price_sar':120.0,'weight_kg':3.5}
with pc2:
    if st.button("🔧 Cheap Tool"): preset = {'material_type':'T','has_pma':0,'is_tool':1,'is_fixed_asset':0,'ata_chapter':'00-00','sales_price_sar':15.0,'weight_kg':1.2}
with pc3:
    if st.button("📦 Standard Part"): preset = {'material_type':'SP','has_pma':1,'is_tool':0,'is_fixed_asset':1,'ata_chapter':'32-00','sales_price_sar':85.0,'weight_kg':6.0}
with pc4:
    if st.button("🗑️ Raw Consumable"): preset = {'material_type':'M','has_pma':0,'is_tool':0,'is_fixed_asset':0,'ata_chapter':'97-97','sales_price_sar':8.0,'weight_kg':0.3}

st.markdown("---")

# ── Inputs ────────────────────────────────────────────────
defaults = preset if preset else {
    'material_type':'SP','has_pma':1,'is_tool':0,
    'is_fixed_asset':0,'ata_chapter':'49-00',
    'sales_price_sar':50.0,'weight_kg':2.0
}

st.markdown("**📋 Enter Part Attributes**")
c1, c2, c3 = st.columns(3)

with c1:
    material_type = st.selectbox("Material Type",
        ['SP','C','T','M','N'],
        index=['SP','C','T','M','N'].index(defaults['material_type']),
        help="SP=Standard Part · C=Component · T=Tool · M=Raw Material · N=Non-aircraft")
    has_pma = st.selectbox("Manufacturer Approval (PMA)",
        [1,0], index=[1,0].index(defaults['has_pma']),
        format_func=lambda x: "✅ Has PMA" if x==1 else "❌ No PMA")

with c2:
    ata_chapter = st.selectbox("ATA Chapter",
        ['00-00','32-00','49-00','72-00','97-97','99-01'],
        index=['00-00','32-00','49-00','72-00','97-97','99-01'].index(defaults['ata_chapter']),
        help="Aircraft system classification code")
    is_tool = st.selectbox("Is this a tool?",
        [0,1], index=[0,1].index(defaults['is_tool']),
        format_func=lambda x: "🔧 Yes — it is a tool" if x==1 else "No — it is a part")

with c3:
    sales_price_sar = st.number_input("Sales Price (SAR)",
        min_value=0.0, max_value=10000.0,
        value=float(defaults['sales_price_sar']), step=1.0)
    weight_kg = st.number_input("Weight (kg)",
        min_value=0.0, max_value=500.0,
        value=float(defaults['weight_kg']), step=0.1)
    is_fixed_asset = st.selectbox("Fixed Asset?",
        [0,1], index=[0,1].index(defaults['is_fixed_asset']),
        format_func=lambda x: "Yes" if x==1 else "No")

st.markdown("&nbsp;", unsafe_allow_html=True)
predict_clicked = st.button("🔍 Predict Repairability", use_container_width=True)

# ── Result ────────────────────────────────────────────────
if predict_clicked or preset:
    input_df = pd.DataFrame([{
        'material_type': material_type, 'has_pma': has_pma,
        'is_tool': is_tool, 'is_fixed_asset': is_fixed_asset,
        'ata_chapter': ata_chapter, 'sales_price_sar': sales_price_sar,
        'weight_kg': weight_kg
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    conf_rep = probability[1] * 100
    conf_unr = probability[0] * 100

    st.markdown("---")
    st.markdown("**🎯 Prediction Result**")

    r1, r2 = st.columns(2)

    with r1:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-green">
                <div style="font-size:2.5rem">✅</div>
                <div class="result-title-green">REPAIRABLE</div>
                <div class="result-desc">This part is likely worth repairing rather than discarding as a consumable.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-red">
                <div style="font-size:2.5rem">❌</div>
                <div class="result-title-red">UNREPAIRABLE</div>
                <div class="result-desc">This part is likely treated as a one-time-use consumable — not worth repairing.</div>
            </div>""", unsafe_allow_html=True)

    with r2:
        st.markdown("**📊 Confidence Breakdown**")
        st.markdown(f"Repairable — **{conf_rep:.1f}%**")
        st.markdown(f"""<div class="bar-wrap"><div class="bar-green" style="width:{conf_rep:.0f}%"></div></div>""",
                    unsafe_allow_html=True)
        st.markdown(f"Unrepairable — **{conf_unr:.1f}%**")
        st.markdown(f"""<div class="bar-wrap"><div class="bar-red" style="width:{conf_unr:.0f}%"></div></div>""",
                    unsafe_allow_html=True)
        st.caption("The model assigns a probability to each class based on how closely this part's attributes match patterns learned during training.")

    # ── Why this prediction ───────────────────────────────
    st.markdown("**🧠 Why This Prediction?**")

    tags = []
    if sales_price_sar > 40:
        tags.append(('pos', f'💰 Price {sales_price_sar:.0f} SAR > 40 SAR threshold → pushes toward Repairable'))
    else:
        tags.append(('neg', f'💰 Price {sales_price_sar:.0f} SAR ≤ 40 SAR → pushes toward Unrepairable'))

    if has_pma == 1:
        tags.append(('pos', '✅ Has manufacturer approval (PMA) → pushes toward Repairable'))
    else:
        tags.append(('neg', '❌ No manufacturer approval → pushes toward Unrepairable'))

    if material_type in ['C','SP']:
        tags.append(('pos', f'📦 Material type {material_type} → pushes toward Repairable'))
    else:
        tags.append(('neu', f'📦 Material type {material_type} → neutral or slight push toward Unrepairable'))

    if is_tool == 1:
        tags.append(('neg', '🔧 Classified as a tool → strong override toward Unrepairable'))
    else:
        tags.append(('neu', '✓ Not a tool → no penalty'))

    tag_html = "".join([
        f'<span class="tag-pos">{t}</span>' if k=='pos'
        else f'<span class="tag-neg">{t}</span>' if k=='neg'
        else f'<span class="tag-neu">{t}</span>'
        for k, t in tags
    ])

    st.markdown(tag_html, unsafe_allow_html=True)
    st.caption("Note: ATA Chapter, weight, and fixed asset status also contribute to the model's internal decisions, though their influence is smaller than the four factors above.")

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <a href="https://www.linkedin.com/in/raghadbaeshen">Raghad Baeshen</a> — 
    IT Specialist in aviation MRO, exploring AI for enterprise operations.<br>
    All data is synthetic · Random Forest · Accuracy: 88% · CV F1: 0.891 ± 0.023<br>
    <a href="https://github.com/raghaddbae/aviation-parts-repairability-classifier">View full project on GitHub →</a>
</div>
""", unsafe_allow_html=True)
