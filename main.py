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
    df = df.drop(columns=[col for col in ["index"] if col in df.columns])

    # Bersihkan data
    df = df.replace(["None", "nan", ""], pd.NA)

    # Paksa string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # Mapping waktu (TETAP)
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

    # 🔥 Mapping stops ke Indonesia
    if "stops" in df.columns:
        df["stops"] = df["stops"].replace({
            "0 stops": "Langsung",
            "1 stop": "1 Transit",
            "2 stops": "2 Transit",
            "3 stops": "3 Transit"
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
# LABEL MAP
# =========================
label_map = {
    "source": "Kota Asal",
    "source_city": "Kota Asal",
    "destination": "Kota Tujuan",
    "destination_city": "Kota Tujuan",
    "departure_time": "Waktu Keberangkatan",
    "arrival_time": "Waktu Kedatangan",
    "stops": "Jumlah Transit",
    "duration": "Durasi Penerbangan",
    "days_left": "Sisa Hari Pemesanan",
    "class": "Kelas Penerbangan"
}


# =========================
# AI FLIGHT ENGINE
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    flight_options = df[["airline", "flight", "stops"]].drop_duplicates()

    # Limit biar cepat
    if len(flight_options) > 1000:
        flight_options = flight_options.sample(500, random_state=42)

    results = []

    reverse_map = {
        "Langsung": "0 stops",
        "1 Transit": "1 stop",
        "2 Transit": "2 stops",
        "3 Transit": "3 stops"
    }

    for _, row in flight_options.iterrows():
        temp = input_data.copy()

        temp["airline"] = row["airline"]
        temp["flight"] = row["flight"]

        # 🔥 convert balik ke format model
        temp["stops"] = reverse_map.get(row["stops"], row["stops"])

        try:
            pred = model.predict(pd.DataFrame([temp]))[0]

            results.append({
                "airline": row["airline"],
                "flight": row["flight"],
                "stops": row["stops"],  # tampilkan versi Indonesia
                "price": pred
            })
        except:
            continue

    results = sorted(results, key=lambda x: x["price"])

    return results[:top_n]


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Advisor", layout="wide")
st.title("✈️ AI Flight Price Advisor")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

input_data = {}

# =========================
# ROUTE
# =========================
st.subheader("✈️ Rute Penerbangan")

route_cols = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

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
# INPUT LAIN
# =========================
feature_cols = df.drop(columns=["price"]).columns

col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):

    if col in ["airline", "flight", "source", "destination", "source_city", "destination_city"]:
        continue

    container = col1 if i % 2 == 0 else col2
    label = label_map.get(col, col)

    # DURATION
    if col.lower() == "duration":
        val = container.slider(label, 0.0, 24.0, 1.0, step=0.25)

        hours = int(val)
        minutes = int((val - hours) * 60)

        container.caption(f"{hours} jam {minutes} menit")
        input_data[col] = val

    # DAYS LEFT
    elif col.lower() == "days_left":
        val = container.slider(label, 0.0, 30.0, 10.0, step=0.5)

        days = int(val)
        hours = int((val - days) * 24)

        container.caption(f"{days} hari {hours} jam")
        input_data[col] = val

    # NUMERIC
    elif ptypes.is_numeric_dtype(df[col]):
        try:
            num = pd.to_numeric(df[col], errors="coerce")

            input_data[col] = container.slider(
                label,
                float(num.min()),
                float(num.max()),
                float(num.mean())
            )
        except:
            input_data[col] = container.selectbox(
                label,
                sorted(df[col].dropna().astype(str).unique())
            )

    # CATEGORICAL
    else:
        input_data[col] = container.selectbox(
            label,
            sorted(df[col].dropna().astype(str).unique())
        )


# =========================
# PREDICT
# =========================
if st.button("🔍 Cari Penerbangan Terbaik"):

    results = find_best_flights(df, model, input_data)

    st.subheader("🏆 Rekomendasi Penerbangan Terbaik")

    for i, r in enumerate(results, 1):
        st.write(
            f"{i}. ✈️ {r['airline']} ({r['flight']}) | "
            f"{r['stops']} | 💰 Rp {int(r['price']):,}"
        )