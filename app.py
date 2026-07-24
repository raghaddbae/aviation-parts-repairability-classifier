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

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Parts Repairability Classifier",
    page_icon="✈️",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean background */
    .stApp { background-color: #0f1117; }
    
    /* Main content area */
    .block-container { padding: 2rem 3rem; }

    /* Hero header */
    .hero {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border: 1px solid #2563eb30;
        border-radius: 16px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin: 0 0 1.5rem 0;
        line-height: 1.6;
    }
    .disclaimer {
        background: #1e2030;
        border-left: 3px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 0.75rem 1rem;
        font-size: 0.82rem;
        color: #94a3b8;
    }

    /* Metric cards */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2563eb20;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        flex: 1;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #2563eb;
        display: block;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 1rem;
        display: block;
    }

    /* Input card */
    .input-card {
        background: #1a1f2e;
        border: 1px solid #2563eb20;
        border-radius: 12px;
        padding: 1.5rem;
    }

    /* Result cards */
    .result-repairable {
        background: linear-gradient(135deg, #052e16 0%, #0f2d1a 100%);
        border: 1px solid #16a34a;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-unrepairable {
        background: linear-gradient(135deg, #2d1515 0%, #1f0f0f 100%);
        border: 1px solid #dc2626;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0.5rem 0;
    }
    .result-label-green { color: #4ade80; }
    .result-label-red { color: #f87171; }
    .result-desc {
        font-size: 0.88rem;
        color: #94a3b8;
        line-height: 1.6;
        margin-top: 0.75rem;
    }

    /* Confidence bar */
    .conf-bar-container {
        background: #0f1117;
        border-radius: 8px;
        height: 10px;
        margin: 0.4rem 0 0.2rem 0;
        overflow: hidden;
    }
    .conf-bar-fill-green {
        background: linear-gradient(90deg, #16a34a, #4ade80);
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }
    .conf-bar-fill-blue {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        height: 100%;
        border-radius: 8px;
    }
    .conf-bar-fill-red {
        background: linear-gradient(90deg, #dc2626, #f87171);
        height: 100%;
        border-radius: 8px;
    }
    .conf-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Feature influence tags */
    .tag-positive {
        display: inline-block;
        background: #052e16;
        border: 1px solid #16a34a40;
        color: #4ade80;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        margin: 0.2rem;
    }
    .tag-negative {
        display: inline-block;
        background: #2d1515;
        border: 1px solid #dc262640;
        color: #f87171;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        margin: 0.2rem;
    }
    .tag-neutral {
        display: inline-block;
        background: #1a1f2e;
        border: 1px solid #2563eb30;
        color: #94a3b8;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.78rem;
        margin: 0.2rem;
    }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100% !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1e40af, #1d4ed8) !important;
        transform: translateY(-1px) !important;
    }

    /* Preset buttons */
    .stButton > button[kind="secondary"] {
        background: #1a1f2e !important;
        border: 1px solid #2563eb30 !important;
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        padding: 0.4rem 0.8rem !important;
    }

    /* Sidebar */
    .css-1d391kg, [data-testid="stSidebar"] {
        background: #0d1117 !important;
    }
    .sidebar-card {
        background: #1a1f2e;
        border: 1px solid #2563eb20;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 0.7rem;
        font-weight: 600;
        color: #2563eb;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    .sidebar-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid #2563eb10;
        font-size: 0.82rem;
    }
    .sidebar-row:last-child { border-bottom: none; }
    .sidebar-key { color: #64748b; }
    .sidebar-val { color: #f1f5f9; font-weight: 600; }

    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2563eb15;
        text-align: center;
        color: #475569;
        font-size: 0.78rem;
        line-height: 1.8;
    }
    .footer a { color: #2563eb; text-decoration: none; }

    /* Hide streamlit branding */
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

# ── Sidebar — Model Info ──────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">⚙️ Model Performance</div>
        <div class="sidebar-row"><span class="sidebar-key">Algorithm</span><span class="sidebar-val">Random Forest</span></div>
        <div class="sidebar-row"><span class="sidebar-key">Test Accuracy</span><span class="sidebar-val">88%</span></div>
        <div class="sidebar-row"><span class="sidebar-key">CV F1 Score</span><span class="sidebar-val">0.891 ± 0.023</span></div>
        <div class="sidebar-row"><span class="sidebar-key">Training Parts</span><span class="sidebar-val">400</span></div>
        <div class="sidebar-row"><span class="sidebar-key">Validation</span><span class="sidebar-val">5-Fold Stratified</span></div>
        <div class="sidebar-row"><span class="sidebar-key">Data Type</span><span class="sidebar-val">Synthetic</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">📊 Class Performance</div>
        <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:0.5rem;">Unrepairable</div>
        <div class="conf-bar-container"><div class="conf-bar-fill-blue" style="width:95%"></div></div>
        <div class="conf-label"><span>Recall</span><span style="color:#f1f5f9">95%</span></div>
        <div style="font-size:0.78rem; color:#94a3b8; margin:0.75rem 0 0.5rem 0;">Repairable</div>
        <div class="conf-bar-container"><div class="conf-bar-fill-blue" style="width:76%"></div></div>
        <div class="conf-label"><span>Recall</span><span style="color:#f1f5f9">76%</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-title">🔗 Links</div>
        <div style="font-size:0.82rem; line-height:2;">
            <a href="https://github.com/raghaddbae/aviation-parts-repairability-classifier" 
               style="color:#2563eb; text-decoration:none;">📁 GitHub Repository</a><br>
            <a href="https://www.linkedin.com/in/raghadbaeshen" 
               style="color:#2563eb; text-decoration:none;">👤 LinkedIn</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-title">✈️ Aviation Parts Repairability Classifier</p>
    <p class="hero-subtitle">
        A machine learning proof-of-concept that predicts whether an aircraft part 
        should be classified as <strong style="color:#f1f5f9">repairable</strong> or 
        <strong style="color:#f1f5f9">unrepairable</strong>, based on its operational attributes.
        Trained on synthetic data modeled on real MRO classification logic.
    </p>
    <div class="disclaimer">
        ⚠️ <strong>Demonstration only.</strong> All data is synthetic and randomly generated. 
        This does not represent any real company, system, or dataset.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Quick Presets ─────────────────────────────────────────
st.markdown('<span class="section-label">🧪 Quick Test — Try an Example</span>', unsafe_allow_html=True)

preset_cols = st.columns(4)
preset = None
with preset_cols[0]:
    if st.button("💰 Expensive Component"):
        preset = {'material_type':'C','has_pma':1,'is_tool':0,'is_fixed_asset':0,
                  'ata_chapter':'72-00','sales_price_sar':120.0,'weight_kg':3.5}
with preset_cols[1]:
    if st.button("🔧 Cheap Tool"):
        preset = {'material_type':'T','has_pma':0,'is_tool':1,'is_fixed_asset':0,
                  'ata_chapter':'00-00','sales_price_sar':15.0,'weight_kg':1.2}
with preset_cols[2]:
    if st.button("📦 Standard Part"):
        preset = {'material_type':'SP','has_pma':1,'is_tool':0,'is_fixed_asset':1,
                  'ata_chapter':'32-00','sales_price_sar':85.0,'weight_kg':6.0}
with preset_cols[3]:
    if st.button("🗑️ Consumable"):
        preset = {'material_type':'M','has_pma':0,'is_tool':0,'is_fixed_asset':0,
                  'ata_chapter':'97-97','sales_price_sar':8.0,'weight_kg':0.3}

st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

# ── Input Form ────────────────────────────────────────────
st.markdown('<span class="section-label">📋 Part Attributes</span>', unsafe_allow_html=True)

defaults = preset if preset else {
    'material_type': 'SP', 'has_pma': 1, 'is_tool': 0,
    'is_fixed_asset': 0, 'ata_chapter': '49-00',
    'sales_price_sar': 50.0, 'weight_kg': 2.0
}

col1, col2, col3 = st.columns(3)

with col1:
    material_type = st.selectbox("Material Type",
        options=['SP', 'C', 'T', 'M', 'N'],
        index=['SP','C','T','M','N'].index(defaults['material_type']),
        help="SP=Standard Part · C=Component · T=Tool · M=Raw Material · N=Non-aircraft"
    )
    has_pma = st.selectbox("Manufacturer Approval (PMA)",
        options=[1, 0],
        index=[1,0].index(defaults['has_pma']),
        format_func=lambda x: "✅ Has PMA" if x == 1 else "❌ No PMA"
    )

with col2:
    ata_chapter = st.selectbox("ATA Chapter",
        options=['00-00', '32-00', '49-00', '72-00', '97-97', '99-01'],
        index=['00-00','32-00','49-00','72-00','97-97','99-01'].index(defaults['ata_chapter']),
        help="Aircraft system classification code"
    )
    is_tool = st.selectbox("Is this a tool?",
        options=[0, 1],
        index=[0,1].index(defaults['is_tool']),
        format_func=lambda x: "🔧 Yes — it is a tool" if x == 1 else "No — it is a part"
    )

with col3:
    sales_price_sar = st.number_input("Sales Price (SAR)",
        min_value=0.0, max_value=10000.0,
        value=float(defaults['sales_price_sar']), step=1.0
    )
    weight_kg = st.number_input("Weight (kg)",
        min_value=0.0, max_value=500.0,
        value=float(defaults['weight_kg']), step=0.1
    )
    is_fixed_asset = st.selectbox("Fixed Asset?",
        options=[0, 1],
        index=[0,1].index(defaults['is_fixed_asset']),
        format_func=lambda x: "Yes — tracked as fixed asset" if x == 1 else "No"
    )

st.markdown("<div style='margin:1.5rem 0'></div>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Repairability", use_container_width=True)

# ── Result ────────────────────────────────────────────────
if predict_clicked or preset:
    input_df = pd.DataFrame([{
        'material_type': material_type,
        'has_pma': has_pma,
        'is_tool': is_tool,
        'is_fixed_asset': is_fixed_asset,
        'ata_chapter': ata_chapter,
        'sales_price_sar': sales_price_sar,
        'weight_kg': weight_kg
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    conf_repairable = probability[1] * 100
    conf_unrepairable = probability[0] * 100
    confidence = probability[prediction] * 100

    st.markdown("<div style='margin:2rem 0 1rem 0'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">🎯 Prediction Result</span>', unsafe_allow_html=True)

    res_col, conf_col = st.columns([1, 1])

    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-repairable">
                <div style="font-size:2.5rem">✅</div>
                <div class="result-label result-label-green">REPAIRABLE</div>
                <div class="result-desc">
                    Based on the part's attributes, the model predicts this part 
                    is worth repairing rather than discarding as a consumable.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unrepairable">
                <div style="font-size:2.5rem">❌</div>
                <div class="result-label result-label-red">UNREPAIRABLE</div>
                <div class="result-desc">
                    Based on the part's attributes, the model predicts this part 
                    is treated as a one-time-use consumable — not worth repairing.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with conf_col:
        st.markdown(f"""
        <div style="background:#1a1f2e; border:1px solid #2563eb20; border-radius:12px; padding:1.5rem;">
            <div class="sidebar-title">📊 Confidence Breakdown</div>

            <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:0.3rem;">
                Repairable
            </div>
            <div class="conf-bar-container">
                <div class="conf-bar-fill-green" style="width:{conf_repairable:.0f}%"></div>
            </div>
            <div class="conf-label">
                <span>Probability</span>
                <span style="color:#4ade80; font-weight:700">{conf_repairable:.1f}%</span>
            </div>

            <div style="margin-top:1rem; font-size:0.82rem; color:#94a3b8; margin-bottom:0.3rem;">
                Unrepairable
            </div>
            <div class="conf-bar-container">
                <div class="conf-bar-fill-red" style="width:{conf_unrepairable:.0f}%"></div>
            </div>
            <div class="conf-label">
                <span>Probability</span>
                <span style="color:#f87171; font-weight:700">{conf_unrepairable:.1f}%</span>
            </div>

            <div style="margin-top:1.25rem; padding-top:1rem; border-top:1px solid #2563eb15;
                        font-size:0.78rem; color:#64748b; line-height:1.6;">
                The model assigns a probability to each class based on how closely 
                this part's attributes match patterns learned during training.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Why this prediction ───────────────────────────────
    st.markdown("<div style='margin:1.5rem 0 0.5rem 0'></div>", unsafe_allow_html=True)
    st.markdown('<span class="section-label">🧠 Why This Prediction?</span>', unsafe_allow_html=True)

    # Build influence tags based on actual input values
    tags = []
    if sales_price_sar > 40:
        tags.append(('positive', f'💰 Price {sales_price_sar:.0f} SAR > 40 SAR threshold → pushes toward Repairable'))
    else:
        tags.append(('negative', f'💰 Price {sales_price_sar:.0f} SAR ≤ 40 SAR → pushes toward Unrepairable'))

    if has_pma == 1:
        tags.append(('positive', '✅ Has manufacturer approval (PMA) → pushes toward Repairable'))
    else:
        tags.append(('negative', '❌ No manufacturer approval → pushes toward Unrepairable'))

    if material_type in ['C', 'SP']:
        tags.append(('positive', f'📦 Material type {material_type} (Component/Standard Part) → pushes toward Repairable'))
    else:
        tags.append(('neutral', f'📦 Material type {material_type} → neutral or slight push toward Unrepairable'))

    if is_tool == 1:
        tags.append(('negative', '🔧 Classified as a tool → strong push toward Unrepairable (overrides other factors)'))
    else:
        tags.append(('neutral', '✓ Not a tool → no penalty applied'))

    tag_html = ""
    for tag_type, tag_text in tags:
        tag_html += f'<div class="tag-{tag_type}">{tag_text}</div>'

    st.markdown(f"""
    <div style="background:#1a1f2e; border:1px solid #2563eb20; border-radius:12px; padding:1.5rem;">
        <div style="font-size:0.82rem; color:#94a3b8; margin-bottom:1rem; line-height:1.6;">
            The model uses a scoring system based on four key attributes. 
            Here is how your part's inputs influenced the prediction:
        </div>
        {tag_html}
        <div style="margin-top:1rem; font-size:0.75rem; color:#475569; 
                    padding-top:0.75rem; border-top:1px solid #2563eb10;">
            Note: ATA Chapter, weight, and fixed asset status also contribute to the model's 
            internal tree decisions, though their influence is smaller than the four factors above.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <a href="https://www.linkedin.com/in/raghadbaeshen">Raghad Baeshen</a> 
    — IT Specialist in aviation MRO, exploring AI for enterprise operations.<br>
    All data is synthetic. Model: Random Forest · Accuracy: 88% · CV F1: 0.891 ± 0.023<br>
    <a href="https://github.com/raghaddbae/aviation-parts-repairability-classifier">View full project on GitHub →</a>
</div>
""", unsafe_allow_html=True)
