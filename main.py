import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore

MODEL_PATH = "model_tiket.pkl"
META_PATH = "model_meta.pkl"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")

    # WAKTU INDONESIA
    time_map = {
        "Early_Morning": "dini hari",
        "Morning": "pagi",
        "Afternoon": "siang",
        "Evening": "malam",
        "Night": "tengah malam"
    }

    for col in ["departure_time", "arrival_time"]:
        if col in df.columns:
            df[col] = df[col].replace(time_map)

    # KELAS
    if "class" in df.columns:
        df["class"] = df["class"].replace({
            "Economy": "ekonomi",
            "Business": "bisnis"
        })

    return df


# =========================
# LOAD MODEL (CEPAT)
# =========================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# =========================
# AI ADVISOR
# =========================
def advisor(input_data):
    recs = []

    if input_data.get("days_left", 0) < 5:
        recs.append("⚠️ Harga tinggi karena booking terlalu dekat")
    elif input_data.get("days_left", 0) > 20:
        recs.append("💰 Lebih murah karena booking jauh hari")

    if input_data.get("stops", 0) > 0:
        recs.append("🔄 Transit bisa lebih murah tapi lebih lama")
    else:
        recs.append("✈️ Direct flight lebih cepat tapi mahal")

    if input_data.get("class") == "bisnis":
        recs.append("💺 Kelas bisnis meningkatkan harga signifikan")

    return recs


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Price Advisor", layout="wide")
st.title("✈️ AI Flight Price Advisor (Indonesia - Fast)")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

# INPUT
feature_cols = df.drop(columns=["price"]).columns

col1, col2 = st.columns(2)
input_data = {}

for i, col in enumerate(feature_cols):
    container = col1 if i % 2 == 0 else col2

    if df[col].dtype == "object":
        input_data[col] = container.selectbox(col, df[col].unique())
    else:
        input_data[col] = container.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

# PREDIKSI
if st.button("🔍 Prediksi & Rekomendasi"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    c1, c2 = st.columns(2)

    c1.metric("Estimasi Harga", f"Rp {int(pred):,}")

    with c2:
        st.subheader("Rekomendasi AI")
        for r in advisor(input_data):
            st.write("-", r)