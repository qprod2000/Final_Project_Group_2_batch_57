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

    # Hapus kolom tidak perlu
    df = df.drop(columns=[col for col in ["index", "flight"] if col in df.columns])

    # Bersihkan data
    df = df.replace(["None", "nan", ""], pd.NA)

    # Paksa string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # Mapping waktu (TETAP PUNYA KAMU)
    time_map = {
        "Early_Morning": "Dini hari",
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
            "Economy": "Ekonomi",
            "Business": "Bisnis"
        })

    return df


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# =========================
# LABEL UI
# =========================
label_map = {
    "airline": "Maskapai",
    "departure_time": "Waktu Keberangkatan",
    "arrival_time": "Waktu Kedatangan",
    "stops": "Jumlah Transit",
    "duration": "Durasi Penerbangan",
    "days_left": "Sisa Hari Pemesanan",
    "class": "Kelas Penerbangan"
}


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

    if input_data.get("class") == "Bisnis":
        recs.append("💺 Kelas bisnis meningkatkan harga signifikan")

    return recs


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Price Advisor", layout="wide")
st.title("✈️ AI Flight Price Advisor (Final Version)")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

input_data = {}

# =========================
# ✈️ ROUTE SECTION (FINAL FIX)
# =========================
st.subheader("✈️ Rute Penerbangan")

route_cols = st.columns(2)

# Deteksi kolom asal
source_col = None
for col in ["source", "source_city", "from", "origin"]:
    if col in df.columns:
        source_col = col
        break

# Deteksi kolom tujuan
dest_col = None
for col in ["destination", "destination_city", "to", "dest"]:
    if col in df.columns:
        dest_col = col
        break

if source_col:
    input_data[source_col] = route_cols[0].selectbox(
        "Kota Asal",
        sorted(df[source_col].dropna().unique())
    )

if dest_col:
    input_data[dest_col] = route_cols[1].selectbox(
        "Kota Tujuan",
        sorted(df[dest_col].dropna().unique())
    )

# =========================
# INPUT LAINNYA
# =========================
feature_cols = df.drop(columns=["price"]).columns

col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):

    # skip route biar tidak double
    if col in ["source", "destination", "source_city", "destination_city"]:
        continue

    container = col1 if i % 2 == 0 else col2
    label = label_map.get(col, col)

    # =========================
    # DURATION
    # =========================
    if col.lower() == "duration":
        val = container.slider(label, 0.0, 24.0, 1.0, step=0.25)

        hours = int(val)
        minutes = int((val - hours) * 60)

        container.caption(f"⏱ {hours} jam {minutes} menit")

        input_data[col] = val

    # =========================
    # DAYS LEFT
    # =========================
    elif col.lower() == "days_left":
        val = container.slider(label, 0.0, 30.0, 10.0, step=0.5)

        days = int(val)
        hours = int((val - days) * 24)

        container.caption(f"📅 {days} hari {hours} jam sebelum keberangkatan")

        input_data[col] = val

    # =========================
    # NUMERIC SAFE
    # =========================
    elif ptypes.is_numeric_dtype(df[col]):
        try:
            num_series = pd.to_numeric(df[col], errors="coerce")

            min_val = float(num_series.min())
            max_val = float(num_series.max())
            mean_val = float(num_series.mean())

            input_data[col] = container.slider(
                label,
                min_val,
                max_val,
                mean_val
            )
        except:
            input_data[col] = container.selectbox(
                label,
                sorted(df[col].dropna().astype(str).unique())
            )

    # =========================
    # CATEGORICAL
    # =========================
    else:
        input_data[col] = container.selectbox(
            label,
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