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

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Aviation Parts Repairability Classifier",
    page_icon="✈️",
    layout="centered"
)

# ── Title ─────────────────────────────────────────────────
st.title("✈️ Aviation Parts Repairability Classifier")
st.markdown("""
A machine learning proof-of-concept that predicts whether an aircraft part 
should be classified as **repairable** or **unrepairable**, based on its attributes.

> ⚠️ **Disclaimer:** This application uses a model trained entirely on **synthetic, 
randomly generated data**. It does not represent any real company, system, or dataset. 
It is intended solely as a technical demonstration.
""")

st.divider()

# ── Train model on synthetic data ─────────────────────────
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

# ── Input form ────────────────────────────────────────────
st.subheader("Enter Part Attributes")

col1, col2 = st.columns(2)

with col1:
    material_type = st.selectbox(
        "Material Type",
        options=['SP', 'C', 'T', 'M', 'N'],
        help="SP=Standard Part, C=Component, T=Tool, M=Raw Material, N=Non-aircraft"
    )
    has_pma = st.radio(
        "Has PMA (Manufacturer Approval)?",
        options=[1, 0],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    is_tool = st.radio(
        "Is this a tool?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    ata_chapter = st.selectbox(
        "ATA Chapter",
        options=['00-00', '32-00', '49-00', '72-00', '97-97', '99-01'],
        help="Aircraft system classification code"
    )
    sales_price_sar = st.number_input(
        "Sales Price (SAR)",
        min_value=0.0,
        max_value=10000.0,
        value=50.0,
        step=1.0
    )
    weight_kg = st.number_input(
        "Weight (kg)",
        min_value=0.0,
        max_value=500.0,
        value=2.0,
        step=0.1
    )
    is_fixed_asset = st.radio(
        "Is Fixed Asset?",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

st.divider()

# ── Prediction ────────────────────────────────────────────
if st.button("🔍 Predict Repairability", use_container_width=True):

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
    confidence = probability[prediction] * 100

    if prediction == 1:
        st.success(f"### ✅ REPAIRABLE")
        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.markdown("""
        **What this means:** Based on the part attributes provided, the model predicts 
        this part is likely worth repairing rather than discarding.
        """)
    else:
        st.error(f"### ❌ UNREPAIRABLE")
        st.metric("Model Confidence", f"{confidence:.1f}%")
        st.markdown("""
        **What this means:** Based on the part attributes provided, the model predicts 
        this part is likely treated as a one-time-use consumable.
        """)

    st.divider()

    st.subheader("Confidence Breakdown")
    conf_df = pd.DataFrame({
        'Classification': ['Unrepairable', 'Repairable'],
        'Confidence': [f"{probability[0]*100:.1f}%", f"{probability[1]*100:.1f}%"]
    })
    st.table(conf_df)

    st.info("""
    ℹ️ **How to interpret this:** The model assigns a probability to each class. 
    The higher the confidence, the more clearly the part's attributes align with 
    that classification pattern in the training data.
    """)

# ── Footer ────────────────────────────────────────────────
st.divider()
st.markdown("""
**About this project**  
Built by [Raghad Baeshen](https://www.linkedin.com/in/raghadbaeshen) as a proof-of-concept 
exploring AI automation of manual classification tasks in aviation MRO operations.  
All data is synthetic. Model accuracy on test set: **88%** (Random Forest, 5-fold CV F1: 0.891 ± 0.023).  
[View full project on GitHub](https://github.com/raghaddbae/aviation-parts-repairability-classifier)
""")
