import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
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
    .block-container { padding: 2rem 3rem 3rem 3rem; }

    .hero {
        background: linear-gradient(135deg, #1e40af 0%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
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

    div[data-testid="column"] .stButton > button {
        background: white !important;
        color: #1d4ed8 !important;
        border: 2px solid #1d4ed8 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        width: 100% !important;
        padding: 0.6rem 0.5rem !important;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #eff6ff !important;
    }

    .stButton > button {
        background: #1d4ed8 !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        width: 100% !important;
        padding: 0.75rem !important;
    }
    .stButton > button:hover { background: #1e40af !important; }

    .result-green {
        background: #f0fdf4; border: 2px solid #16a34a;
        border-radius: 14px; padding: 1.5rem; text-align: center;
    }
    .result-red {
        background: #fff1f2; border: 2px solid #dc2626;
        border-radius: 14px; padding: 1.5rem; text-align: center;
    }
    .result-title-green { font-size: 1.5rem; font-weight: 800; color: #15803d; margin: 0.4rem 0; }
    .result-title-red   { font-size: 1.5rem; font-weight: 800; color: #dc2626; margin: 0.4rem 0; }
    .result-desc { font-size: 0.85rem; color: #64748b; margin-top: 0.4rem; line-height: 1.6; }

    .tag-pos {
        display: inline-block; background: #f0fdf4; border: 1px solid #86efac;
        color: #166534; border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.8rem; margin: 0.2rem; font-weight: 500;
    }
    .tag-neg {
        display: inline-block; background: #fff1f2; border: 1px solid #fca5a5;
        color: #991b1b; border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.8rem; margin: 0.2rem; font-weight: 500;
    }
    .tag-neu {
        display: inline-block; background: #f1f5f9; border: 1px solid #cbd5e1;
        color: #475569; border-radius: 20px; padding: 0.3rem 0.9rem;
        font-size: 0.8rem; margin: 0.2rem; font-weight: 500;
    }

    /* Prediction history cards */
    .hist-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.8rem;
    }
    .hist-card-rep {
        border-left: 4px solid #16a34a;
    }
    .hist-card-unr {
        border-left: 4px solid #dc2626;
    }
    .hist-label-rep { color: #15803d; font-weight: 700; font-size: 0.85rem; }
    .hist-label-unr { color: #dc2626; font-weight: 700; font-size: 0.85rem; }
    .hist-attr { color: #94a3b8; font-size: 0.75rem; margin-top: 0.3rem; line-height: 1.6; }
    .hist-conf { color: #64748b; font-size: 0.75rem; margin-top: 0.2rem; }

    [data-testid="stSidebar"] { background: white !important; border-right: 1px solid #e2e8f0; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .footer {
        margin-top: 2rem; padding-top: 1.5rem;
        border-top: 1px solid #e2e8f0;
        text-align: center; color: #94a3b8;
        font-size: 0.78rem; line-height: 1.8;
    }
    .footer a { color: #1d4ed8; text-decoration: none; }
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

# ── Session state for prediction history ─────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar — Prediction History ─────────────────────────
with st.sidebar:
    st.markdown("### 📋 Prediction History")

    if len(st.session_state.history) == 0:
        st.markdown("""
        <div style="color:#94a3b8; font-size:0.82rem; 
                    text-align:center; padding:2rem 0; line-height:1.8;">
            No predictions yet.<br>
            Enter part attributes and<br>
            click <strong>Predict</strong> to begin.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary counts
        total = len(st.session_state.history)
        rep_count = sum(1 for h in st.session_state.history if h['prediction'] == 1)
        unr_count = total - rep_count

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total", total)
        col_b.metric("✅ Rep.", rep_count)
        col_c.metric("❌ Unr.", unr_count)

        st.markdown("---")

        # Clear button
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

        st.markdown("---")

        # History cards — most recent first
        for i, entry in enumerate(reversed(st.session_state.history)):
            card_class = "hist-card hist-card-rep" if entry['prediction'] == 1 else "hist-card hist-card-unr"
            label_class = "hist-label-rep" if entry['prediction'] == 1 else "hist-label-unr"
            label_text = "✅ REPAIRABLE" if entry['prediction'] == 1 else "❌ UNREPAIRABLE"
            pma_text = "Has PMA" if entry['has_pma'] == 1 else "No PMA"
            tool_text = "Tool" if entry['is_tool'] == 1 else "Not a tool"

            st.markdown(f"""
            <div class="{card_class}">
                <div class="{label_class}">{label_text}</div>
                <div class="hist-attr">
                    {entry['material_type']} · {entry['ata_chapter']}<br>
                    {entry['sales_price_sar']:.0f} SAR · {entry['weight_kg']} kg<br>
                    {pma_text} · {tool_text}
                </div>
                <div class="hist-conf">
                    Confidence: {entry['confidence']:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔗 Links")
    st.markdown("[📁 GitHub](https://github.com/raghaddbae/aviation-parts-repairability-classifier)")
    st.markdown("[👤 LinkedIn](https://www.linkedin.com/in/raghadbaeshen)")

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
    if st.button("💰 Expensive Component", use_container_width=True):
        preset = {'material_type':'C','has_pma':1,'is_tool':0,'is_fixed_asset':0,'ata_chapter':'72-00','sales_price_sar':120.0,'weight_kg':3.5}
with pc2:
    if st.button("🔧 Cheap Tool", use_container_width=True):
        preset = {'material_type':'T','has_pma':0,'is_tool':1,'is_fixed_asset':0,'ata_chapter':'00-00','sales_price_sar':15.0,'weight_kg':1.2}
with pc3:
    if st.button("📦 Standard Part", use_container_width=True):
        preset = {'material_type':'SP','has_pma':1,'is_tool':0,'is_fixed_asset':1,'ata_chapter':'32-00','sales_price_sar':85.0,'weight_kg':6.0}
with pc4:
    if st.button("🗑️ Raw Consumable", use_container_width=True):
        preset = {'material_type':'M','has_pma':0,'is_tool':0,'is_fixed_asset':0,'ata_chapter':'97-97','sales_price_sar':8.0,'weight_kg':0.3}

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
    has_pma = st.selectbox("Manufacturer Approval (PMA)", [1,0],
        index=[1,0].index(defaults['has_pma']),
        format_func=lambda x: "✅ Has PMA" if x==1 else "❌ No PMA")

with c2:
    ata_chapter = st.selectbox("ATA Chapter",
        ['00-00','32-00','49-00','72-00','97-97','99-01'],
        index=['00-00','32-00','49-00','72-00','97-97','99-01'].index(defaults['ata_chapter']),
        help="Aircraft system classification code")
    is_tool = st.selectbox("Is this a tool?", [0,1],
        index=[0,1].index(defaults['is_tool']),
        format_func=lambda x: "🔧 Yes — it is a tool" if x==1 else "No — it is a part")

with c3:
    sales_price_sar = st.number_input("Sales Price (SAR)",
        min_value=0.0, max_value=10000.0,
        value=float(defaults['sales_price_sar']), step=1.0)
    weight_kg = st.number_input("Weight (kg)",
        min_value=0.0, max_value=500.0,
        value=float(defaults['weight_kg']), step=0.1)
    is_fixed_asset = st.selectbox("Fixed Asset?", [0,1],
        index=[0,1].index(defaults['is_fixed_asset']),
        format_func=lambda x: "Yes" if x==1 else "No")

st.markdown(" ")
predict_clicked = st.button("🔍 Predict Repairability", use_container_width=True)

# ── Result ────────────────────────────────────────────────
if predict_clicked or preset:
    input_df = pd.DataFrame([{
        'material_type': material_type, 'has_pma': has_pma,
        'is_tool': is_tool, 'is_fixed_asset': is_fixed_asset,
        'ata_chapter': ata_chapter,
        'sales_price_sar': sales_price_sar, 'weight_kg': weight_kg
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]
    conf_rep = probability[1] * 100
    conf_unr = probability[0] * 100
    confidence = conf_rep if prediction == 1 else conf_unr

    # Save to history
    st.session_state.history.append({
        'prediction': prediction,
        'confidence': confidence,
        'material_type': material_type,
        'has_pma': has_pma,
        'is_tool': is_tool,
        'is_fixed_asset': is_fixed_asset,
        'ata_chapter': ata_chapter,
        'sales_price_sar': sales_price_sar,
        'weight_kg': weight_kg
    })

    st.markdown("---")
    st.markdown("**🎯 Prediction Result**")

    res_col, gauge_col, chart_col = st.columns([1.2, 1, 1])

    # Result card
    with res_col:
        if prediction == 1:
            st.markdown(f"""
            <div class="result-green">
                <div style="font-size:2.5rem">✅</div>
                <div class="result-title-green">REPAIRABLE</div>
                <div class="result-desc">
                    This part is likely worth repairing rather than discarding as a consumable.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-red">
                <div style="font-size:2.5rem">❌</div>
                <div class="result-title-red">UNREPAIRABLE</div>
                <div class="result-desc">
                    This part is likely treated as a one-time-use consumable — not worth repairing.
                </div>
            </div>""", unsafe_allow_html=True)

    # Gauge chart
    with gauge_col:
        gauge_color = "#16a34a" if prediction == 1 else "#dc2626"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence,
            number={'suffix': "%", 'font': {'size': 28, 'color': gauge_color}},
            title={'text': "Model Confidence", 'font': {'size': 13, 'color': '#64748b'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1,
                         'tickcolor': '#e2e8f0', 'tickfont': {'size': 10}},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50],  'color': '#f8fafc'},
                    {'range': [50, 75], 'color': '#f1f5f9'},
                    {'range': [75, 100],'color': '#e2e8f0'}
                ],
            }
        ))
        fig_gauge.update_layout(
            height=220, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor='white'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # Attribute influence chart
    with chart_col:
        attr_names = ['Price > 40 SAR', 'Has PMA', 'Material Type', 'Is Tool']
        attr_scores = [
            1   if sales_price_sar > 40        else -1,
            1   if has_pma == 1               else -1,
            1   if material_type in ['C','SP'] else -0.5,
            -2  if is_tool == 1               else 0
        ]
        attr_colors = [
            '#16a34a' if s > 0 else '#dc2626' if s < 0 else '#94a3b8'
            for s in attr_scores
        ]
        fig_bar = go.Figure(go.Bar(
            x=attr_scores, y=attr_names, orientation='h',
            marker_color=attr_colors,
            text=[f"+{s}" if s > 0 else str(s) for s in attr_scores],
            textposition='outside',
            textfont=dict(size=11)
        ))
        fig_bar.update_layout(
            title=dict(text="Attribute Influence Score",
                       font=dict(size=13, color='#64748b')),
            height=220, margin=dict(l=10, r=40, t=40, b=10),
            paper_bgcolor='white', plot_bgcolor='white',
            xaxis=dict(range=[-2.5, 1.5], showgrid=True,
                       gridcolor='#f1f5f9', zeroline=True,
                       zerolinecolor='#e2e8f0', zerolinewidth=2,
                       tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=10)),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Why this prediction tags
    st.markdown("**🧠 Why This Prediction?**")
    tags_html = ""
    tags_html += f'<span class="tag-pos">💰 Price {sales_price_sar:.0f} SAR > threshold → Repairable</span>' \
        if sales_price_sar > 40 else \
        f'<span class="tag-neg">💰 Price {sales_price_sar:.0f} SAR ≤ threshold → Unrepairable</span>'
    tags_html += '<span class="tag-pos">✅ Has PMA → Repairable</span>' \
        if has_pma == 1 else '<span class="tag-neg">❌ No PMA → Unrepairable</span>'
    tags_html += f'<span class="tag-pos">📦 {material_type} → Repairable</span>' \
        if material_type in ['C','SP'] else \
        f'<span class="tag-neu">📦 {material_type} → Neutral</span>'
    tags_html += '<span class="tag-neg">🔧 Tool → Overrides all other factors</span>' \
        if is_tool == 1 else '<span class="tag-neu">✓ Not a tool → No penalty</span>'

    st.markdown(tags_html, unsafe_allow_html=True)
    st.caption("ATA Chapter, weight, and fixed asset status also contribute to the model's decisions, though their influence is smaller than the four attributes above.")

# ── Footer ────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built by <a href="https://www.linkedin.com/in/raghadbaeshen">Raghad Baeshen</a> —
    IT Specialist in aviation MRO, exploring AI for enterprise operations.<br>
    All data is synthetic · Random Forest · Accuracy: 88% · CV F1: 0.891 ± 0.023<br>
    <a href="https://github.com/raghaddbae/aviation-parts-repairability-classifier">
    View full project on GitHub →</a>
</div>
""", unsafe_allow_html=True)
