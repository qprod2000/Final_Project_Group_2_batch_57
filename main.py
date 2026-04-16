import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
import pandas.api.types as ptypes   # type: ignore

MODEL_PATH = "model_tiket.pkl"
META_PATH = "model_meta.pkl"

# =========================
# DURATION BINNING (4 bins per hour)
# =========================
def create_duration_bins(duration_hours):
    """
    Convert duration in hours to 4 time bins per hour.
    Each hour is divided into 4 bins: 0-15min, 15-30min, 30-45min, 45-60min
    Maximum 24 hours (96 bins total)
    """
    if pd.isna(duration_hours) or duration_hours <= 0:
        return "Bin_0"
    
    # Cap at 24 hours
    duration_hours = min(float(duration_hours), 24)
    
    # Convert to minutes and calculate bin
    total_minutes = duration_hours * 60
    bin_number = int(total_minutes // 15)  # Each bin is 15 minutes
    bin_number = min(bin_number, 95)  # Max 96 bins (24 hours * 4)
    
    return f"Bin_{bin_number}"


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")

    # FIX tipe data
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # Create 4 time bins per hour for duration (max 24 hours)
    if "duration" in df.columns:
        df["duration_bin"] = df["duration"].apply(create_duration_bins)

    # Mapping waktu Indonesia
    time_map = {
        "Early_Morning": "Dini Hari",
        "Morning": "Pagi",
        "Afternoon": "Siang",
        "Evening": "Malam",
        "Night": "Tengah Malam"
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

def _safe_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def recommend_airline(input_data, df):
    if "airline" not in df.columns:
        return None

    route_cols = ["source_city", "destination_city", "class", "stops", "departure_time"]
    fallback_sets = [
        route_cols,
        ["source_city", "destination_city", "class", "stops"],
        ["source_city", "destination_city", "class"],
        ["source_city", "destination_city"],
    ]

    for cols in fallback_sets:
        if not all(col in df.columns for col in cols):
            continue
        mask = pd.Series(True, index=df.index)
        for col in cols:
            mask &= df[col] == input_data.get(col)
        subset = df[mask]
        if not subset.empty:
            return subset.groupby("airline")["price"].mean().sort_values().index[0]

    return df.groupby("airline")["price"].mean().sort_values().index[0]


def recommend_flights(input_data, df, top_n=2):
    if "flight" not in df.columns or "airline" not in df.columns:
        return []

    route_cols = ["source_city", "destination_city", "class", "stops", "departure_time"]
    fallback_sets = [
        route_cols,
        ["source_city", "destination_city", "class", "stops"],
        ["source_city", "destination_city", "class"],
        ["source_city", "destination_city"],
    ]

    for cols in fallback_sets:
        if not all(col in df.columns for col in cols):
            continue
        mask = pd.Series(True, index=df.index)
        for col in cols:
            mask &= df[col] == input_data.get(col)
        subset = df[mask]
        if not subset.empty:
            grouped = subset.groupby(["airline", "flight"])["price"].mean()
            best_pairs = grouped.sort_values().head(top_n).reset_index()
            return best_pairs.to_dict(orient="records")

    grouped = df.groupby(["airline", "flight"])["price"].mean()
    best_pairs = grouped.sort_values().head(top_n).reset_index()
    return best_pairs.to_dict(orient="records")


def advisor(input_data, df):
    recs = []
    days_left = _safe_number(input_data.get("days_left", 0))

    if days_left < 5:
        recs.append("⚠️ Harga tinggi karena booking terlalu dekat")
    elif days_left > 20:
        recs.append("💰 Lebih murah karena booking jauh hari")

    stops = input_data.get("stops")
    if stops == "two_or_more":
        recs.append("🔄 Rute dengan 2+ stop biasanya menawarkan beberapa pilihan airline dan nomor penerbangan")
    elif _safe_number(stops) > 0:
        recs.append("🔄 Transit bisa lebih murah tapi lebih lama")
    else:
        recs.append("✈️ Direct flight lebih cepat tapi mahal")

    if input_data.get("class") == "Bisnis":
        recs.append("💺 Kelas bisnis meningkatkan harga signifikan")

    best_airline = recommend_airline(input_data, df)
    if best_airline:
        recs.append(f"✈️ Rekomendasi maskapai: {best_airline}")

    best_pairs = recommend_flights(input_data, df, top_n=3 if stops == "two_or_more" else 1)
    if best_pairs:
        if stops == "two_or_more" and len(best_pairs) > 1:
            recs.append("🛫 Rekomendasi penerbangan terbaik untuk 2+ stop:")
            for pair in best_pairs:
                recs.append(f"   - {pair['airline']} / {pair['flight']}")
        else:
            pair = best_pairs[0]
            recs.append(f"🛫 Rekomendasi nomor penerbangan: {pair['flight']} ({pair['airline']})")

    return recs


# =========================
# UI
# =========================
st.set_page_config(page_title="Aplikasi Prediksi Harga Tiket Pesawat", layout="wide")
st.title("✈️ Aplikasi Prediksi Harga Tiket Pesawat")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

# =========================
# INPUT (FIXED VERSION)
# =========================
feature_cols = df.drop(columns=["price", "index", "flight", "airline"]).columns

col1, col2 = st.columns(2)
input_data = {}

for i, col in enumerate(feature_cols):
    container = col1 if i % 2 == 0 else col2

    if ptypes.is_numeric_dtype(df[col]):
        input_data[col] = container.slider(
            col,
            float(df[col].min()),
            float(df[col].max()),
            float(df[col].mean())
        )
    else:
        input_data[col] = container.selectbox(
            col,
            sorted(df[col].dropna().astype(str).unique())
        )

# =========================
# PREDIKSI
# =========================
if st.button("🔍 Prediksi & Rekomendasi"):
    input_data.setdefault("index", 0)
    input_data["airline"] = recommend_airline(input_data, df)
    flight_options = recommend_flights(input_data, df, top_n=3 if input_data.get("stops") == "two_or_more" else 1)
    if flight_options:
        input_data["flight"] = flight_options[0]["flight"]
    input_df = pd.DataFrame([input_data])
    pred = model.predict(input_df)[0]

    c1, c2 = st.columns(2)

    c1.metric("Estimasi Harga", f"Rp {int(pred):,}")

    with c2:
        st.subheader("Rekomendasi AI")
        for r in advisor(input_data, df):
            st.write("-", r)