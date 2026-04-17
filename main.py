import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib   # type: ignore
import os

from engine import find_best_flights, explain
from utils import format_duration, format_inr

# =========================
# LOAD
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("airlines_flights_data.csv")


@st.cache_resource
def load_model():
    path = os.path.join(os.getcwd(), "model_tiket.pkl")
    return joblib.load(path)


df = load_data()
model = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Final Production)")

# =========================
# INPUT
# =========================
col1, col2 = st.columns(2)

source_col = next((c for c in ["source_city", "source"] if c in df.columns), None)
dest_col = next((c for c in ["destination_city", "destination"] if c in df.columns), None)

input_data = {}

if source_col:
    input_data[source_col] = col1.selectbox("Kota Asal", sorted(df[source_col].unique()))

if dest_col:
    input_data[dest_col] = col2.selectbox("Kota Tujuan", sorted(df[dest_col].unique()))

# days
input_data["days_left"] = st.slider("Sisa Hari", 0, 30, 10, step=1)

# preference
preference = st.slider("Prioritas: Hemat 💰 vs Cepat ⚡", 0, 100, 50)

# =========================
# RUN
# =========================
if st.button("🚀 Cari Rekomendasi Terbaik"):

    eco, biz, base_price, base_duration = find_best_flights(
        df, model, input_data, preference
    )

    # ===== ECONOMY =====
    st.subheader("💰 Top 3 Ekonomi")

    cols = st.columns(3)
    for i, (_, r) in enumerate(eco.iterrows()):
        with cols[i]:
            st.markdown(f"""
            ✈️ **{r['airline']}**
            - {r['stops_clean']}
            - ⏱ {format_duration(r['duration'])}
            - 💰 {format_inr(r['price'])}
            """)
            st.caption(explain(r, base_price, base_duration))

    # ===== BUSINESS =====
    st.subheader("💺 Top 3 Bisnis")

    cols = st.columns(3)
    for i, (_, r) in enumerate(biz.iterrows()):
        with cols[i]:
            st.markdown(f"""
            ✈️ **{r['airline']}**
            - {r['stops_clean']}
            - ⏱ {format_duration(r['duration'])}
            - 💰 {format_inr(r['price'])}
            """)
            st.caption(explain(r, base_price, base_duration))