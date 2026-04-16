import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
import pandas.api.types as ptypes   # type: ignore

MODEL_PATH = "model_tiket.pkl"
META_PATH = "model_meta.pkl"

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")


# Hapus kolom yang tidak dipakai
drop_cols = ["index", "flight"]

for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# Pastikan semua kategori string (fix error)
for col in df.select_dtypes(include="object").columns:
     df[col] = df[col].astype(str)

# Mapping waktu Indonesia
time_map = {
    "Early_Morning": "Dini Hari",
    "Morning": "Pagi",
    "Afternoon": "Siang",
    "Evening": "Sore",
    "Night": "Malam"
    }

for col in ["departure_time", "arrival_time"]:
        if col in df.columns:
            df[col] = df[col].replace(time_map)

# Mapping kelas
if "class" in df.columns:
        df["class"] = df["class"].replace({
            "Economy": "ekonomi",
            "Business": "bisnis"
        })

# =========================
# LOAD MODEL (FAST)
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
st.title("✈️ AI Flight Price Advisor (Indonesia - Final)")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

# =========================
# INPUT
# =========================
feature_cols = df.drop(columns=["price"]).columns

col1, col2 = st.columns(2)
input_data = {}

for i, col in enumerate(feature_cols):
    container = col1 if i % 2 == 0 else col2

    # =========================
    # 🔥 SPECIAL: DURATION
    # =========================
    if col.lower() == "duration":
        val = container.slider(
            "Durasi (jam)",
            min_value=0.0,
            max_value=24.0,
            value=1.0,
            step=0.25  # 15 menit
        )

        hours = int(val)
        minutes = int((val - hours) * 60)

        container.caption(f"⏱ {hours} jam {minutes} menit")

        input_data[col] = val

    # =========================
    # NUMERIC
    # =========================
    elif ptypes.is_numeric_dtype(df[col]):
        input_data[col] = container.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )

    # =========================
    # CATEGORICAL
    # =========================
    else:
        input_data[col] = container.selectbox(
            col,
            sorted(df[col].dropna().astype(str).unique())
        )

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi & Rekomendasi"):
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    c1, c2 = st.columns(2)

    c1.metric("Estimasi Harga", f"Rp {int(pred):,}")

    with c2:
        st.subheader("Rekomendasi AI")
        for r in advisor(input_data):
            st.write("-", r)