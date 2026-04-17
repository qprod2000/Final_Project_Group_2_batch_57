import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import joblib   # type: ignore
from engine import find_best_flights, explain

st.set_page_config(page_title="✈️ Flight Price Predictor", layout="wide")

# =========================
# LOAD DATA & MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

model = load_model()
df = load_data()

# =========================
# TITLE
# =========================
st.title("✈️ Prediksi & Rekomendasi Tiket Pesawat")
st.caption("AI-powered recommendation (price vs duration tradeoff)")

# =========================
# USER INPUT
# =========================
col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox("Airline", df["airline"].unique())

with col2:
    source = st.selectbox("Source City", df["source_city"].unique())

with col3:
    destination = st.selectbox("Destination City", df["destination_city"].unique())

col4, col5 = st.columns(2)

with col4:
    stops = st.selectbox("Stops", df["stops"].unique())

with col5:
    travel_class = st.selectbox("Class", df["class"].unique())

duration = st.slider("Max Duration (hours)", 1.0, 20.0, 10.0)

# Preferensi user
preference = st.slider("Prioritas: Hemat 💰 vs Cepat ⚡", 0, 100, 50)

# =========================
# PREP INPUT DATA
# =========================
input_data = {
    "duration": duration,
    # Tambahkan encoding sesuai model kamu
}

# =========================
# BUTTON
# =========================
if st.button("🔍 Cari Rekomendasi"):

    eco, biz, base_price, base_duration = find_best_flights(
        df, model, input_data, preference
    )

    # =========================
    # ECONOMY
    # =========================
    st.subheader("💰 Rekomendasi Economy")

    cols = st.columns(3)

    for i, (_, row) in enumerate(eco.iterrows()):
        with cols[i]:
            st.markdown(f"""
            ### ✈️ {row['airline']}
            **Stops:** {row['stops']}  
            **Harga:** ₹ {int(row['price'])}  
            **Durasi:** {row['duration']} jam  
            ⭐ **Score:** {round(row['value_score'], 2)}
            """)
            st.caption(explain(row, base_price, base_duration))

    # =========================
    # BUSINESS
    # =========================
    st.subheader("💼 Rekomendasi Business")

    cols = st.columns(3)

    for i, (_, row) in enumerate(biz.iterrows()):
        with cols[i]:
            st.markdown(f"""
            ### ✈️ {row['airline']}
            **Stops:** {row['stops']}  
            **Harga:** ₹ {int(row['price'])}  
            **Durasi:** {row['duration']} jam  
            ⭐ **Score:** {round(row['value_score'], 2)}
            """)
            st.caption(explain(row, base_price, base_duration))