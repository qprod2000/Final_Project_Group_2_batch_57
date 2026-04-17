import streamlit as st
import pandas as pd
import joblib

from config import MODEL_PATH, TOP_N
from utils import format_duration, format_inr
from engine import find_best_flights

# =========================
# LOAD
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")

    df = df.drop(columns=[col for col in ["index"] if col in df.columns])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Production Ready)")

df = load_data()
model = load_model()

input_data = {}

col1, col2 = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

if source_col:
    input_data[source_col] = col1.selectbox("Kota Asal", sorted(df[source_col].unique()))

if dest_col:
    input_data[dest_col] = col2.selectbox("Kota Tujuan", sorted(df[dest_col].unique()))

input_data["days_left"] = st.slider("Sisa Hari", 0, 30, 10, step=1)

# =========================
# RUN
# =========================
if st.button("🚀 Cari Rekomendasi Terbaik"):

    eco, biz = find_best_flights(df, model, input_data, TOP_N)

    st.subheader("💰 Top Ekonomi")
    for _, r in eco.iterrows():
        st.write(
            f"✈️ {r['airline']} ({r['flight']}) | "
            f"{r['stops_clean']} | "
            f"{format_duration(r['duration'])} | "
            f"{format_inr(r['price'])}"
        )

    st.subheader("💺 Top Bisnis")
    for _, r in biz.iterrows():
        st.write(
            f"✈️ {r['airline']} ({r['flight']}) | "
            f"{r['stops_clean']} | "
            f"{format_duration(r['duration'])} | "
            f"{format_inr(r['price'])}"
        )