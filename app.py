import streamlit as st
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="📚 Rajhans Book Store AI",
    layout="wide"
)

st.title("📚 Rajhans Pustak Peeth - AI Book App")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("data.csv")   # CSV convert करून upload करा
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# =========================
# SIDEBAR FILTER
# =========================
st.sidebar.header("🔍 Filter Books")

authors = st.sidebar.multiselect(
    "लेखक निवडा",
    options=df['लेखक'].unique()
)

price_range = st.sidebar.slider(
    "किंमत Range",
    int(df['किंमत'].min()),
    int(df['किंमत'].max()),
    (int(df['किंमत'].min()), int(df['किंमत'].max()))
)

filtered_df = df.copy()

if authors:
    filtered_df = filtered_df[filtered_df['लेखक'].isin(authors)]

filtered_df = filtered_df[
    (filtered_df['किंमत'] >= price_range[0]) &
    (filtered_df['किंमत'] <= price_range[1])
]

# =========================
# SEARCH
# =========================
search = st.text_input("🔎 पुस्तक शोधा")

if search:
    filtered_df = filtered_df[
        filtered_df['पुस्तकाचे नाव'].str.contains(search, case=False)
    ]

# =========================
# DISPLAY DATA
# =========================
st.subheader("📚 Book List")

st.dataframe(filtered_df, use_container_width=True)

# =========================
# DOWNLOAD EXCEL
# =========================
st.download_button(
    "📥 Download Excel",
    filtered_df.to_csv(index=False),
    file_name="books.csv"
)

# =========================
# ML PREDICTION
# =========================
st.subheader("🤖 AI Prediction")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input("किंमत टाका", value=100)

with col2:
    discount = st.number_input("सवलत टाका", value=10)

if st.button("Predict"):
    import numpy as np
    
    input_data = np.array([[price, discount]])
    
    try:
        pred = model.predict(input_data)
        st.success(f"📊 Prediction Result: {pred[0]}")
    except:
        st.error("⚠️ Model input mismatch आहे")
