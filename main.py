import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore

MODEL_PATH = "model_tiket.pkl"

# =========================
# TRANSIT NORMALIZATION
# =========================
def normalize_stops(x):
    x = str(x).lower()

    if x in ["0 stops", "zero", "non-stop"]:
        return "Langsung"
    elif x in ["1 stop", "one"]:
        return "1 Transit"
    else:
        return "2 Transit"


# =========================
# HELPER
# =========================
def format_duration(x):
    h = int(x)
    m = int((x - h) * 60)
    return f"{h} jam {m} menit"

def format_inr(x):
    return f"₹ {int(x):,}"


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")

    df = df.drop(columns=[col for col in ["index"] if col in df.columns])

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

    # 🔥 pastikan selalu ada
    df["stops_clean"] = df["stops"].apply(normalize_stops)

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================
# AI ENGINE (FINAL FIX)
# =========================
def find_best_flights(df, model, input_data, top_n=3):

    # 🔥 SAFETY FIX
    if "stops_clean" not in df.columns:
        df["stops_clean"] = df["stops"].apply(normalize_stops)

    # =========================
    # STRATIFIED SAMPLING
    # =========================
    df_work = (
        df.groupby("stops_clean", group_keys=False)
        .apply(lambda x: x.sample(min(60, len(x)), random_state=42))
    )

    # 🔥 FIX KRITIS (WAJIB)
    df_work["stops_clean"] = df_work["stops"].apply(normalize_stops)

    # =========================
    # PREPARE PREDICTION
    # =========================
    df_pred = df_work.copy()

    for k, v in input_data.items():
        df_pred[k] = v

    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred = df_pred[model.feature_names_in_]

    try:
        preds = model.predict(df_pred)
    except:
        return pd.DataFrame(), pd.DataFrame()

    df_work["price"] = preds

# =========================
# TRUE BALANCED SCORING (FIXED)
# =========================

df_work["price_norm"] = (
    (df_work["price"] - df_work["price"].min()) /
    (df_work["price"].max() - df_work["price"].min() + 1e-6)
)

df_work["duration_norm"] = (
    (df_work["duration"] - df_work["duration"].min()) /
    (df_work["duration"].max() - df_work["duration"].min() + 1e-6)
)

# 🔥 clean stops
df_work["stops_clean"] = df_work["stops"].apply(normalize_stops)
df_work["stops_clean"] = df_work["stops_clean"].astype(str).str.strip()

stops_weight = {
    "Langsung": 0.0,
    "1 Transit": 0.1,
    "2 Transit": 0.2
}

# 🔥 SAFE mapping
df_work["stops_penalty"] = (
    df_work["stops_clean"]
    .map(stops_weight)
    .fillna(0.15)
)

df_work["score"] = (
    df_work["price_norm"] * 0.55 +
    df_work["duration_norm"] * 0.30 +
    df_work["stops_penalty"] * 0.15
)

    # =========================
    # 💺 BUSINESS VS ECONOMY
    # =========================
    if price_diff > 0:

        percent_price = (price_diff / eco_best["price"]) * 100

        if time_diff > 0:
            insights.append(
                f"💺 Bisnis lebih cepat {int(time_diff*60)} menit dengan tambahan {format_inr(price_diff)} (+{percent_price:.1f}%)"
            )

            if percent_price < 20:
                insights.append("🔥 Upgrade ke Bisnis tergolong worth it")
            else:
                insights.append("⚖️ Upgrade ke Bisnis kurang optimal dari sisi harga")

        else:
            insights.append("⚠️ Bisnis tidak memberikan keuntungan waktu signifikan")

    # =========================
    # ✈️ TRANSIT ANALYSIS
    # =========================
    if eco_best["stops_clean"] != "Langsung":

        insights.append(
            f"✈️ Rekomendasi terbaik menggunakan {eco_best['stops_clean']} karena harga lebih kompetitif"
        )

        if eco_best["stops_clean"] == "2 Transit":
            insights.append("⚠️ Perjalanan cukup panjang, pertimbangkan kenyamanan")

    else:
        insights.append("⚡ Penerbangan langsung masih menjadi pilihan paling efisien")

    # =========================
    # 🎯 FINAL RECOMMENDATION
    # =========================
    if price_diff < 0:
        insights.append("🔥 Kelas Bisnis justru lebih murah pada kondisi ini (anomali harga)")
    else:
        insights.append("🎯 Ekonomi tetap menjadi pilihan paling hemat secara keseluruhan")

    return insights


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Final Stable)")

df = load_data()
model = load_model()

input_data = {}

# ROUTE
col1, col2 = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

if source_col:
    input_data[source_col] = col1.selectbox("Kota Asal", sorted(df[source_col].unique()))
else:
    st.error("Kolom asal tidak ditemukan")

if dest_col:
    input_data[dest_col] = col2.selectbox("Kota Tujuan", sorted(df[dest_col].unique()))
else:
    st.error("Kolom tujuan tidak ditemukan")

# DAYS
input_data["days_left"] = st.slider("Sisa Hari", 0, 30, 10, step=1)

# RUN
if st.button("🚀 Cari Rekomendasi Terbaik"):

    eco, biz = find_best_flights(df, model, input_data)

    if eco.empty and biz.empty:
        st.warning("Tidak ada hasil ditemukan")
    else:
        st.subheader("💰 Top 3 Ekonomi")

        for _, r in eco.iterrows():
            st.write(
                f"✈️ {r['airline']} ({r['flight']}) | "
                f"{r['stops_clean']} | ⏱ {format_duration(r['duration'])} | "
                f"💰 {format_inr(r['price'])}"
            )

        st.subheader("💺 Top 3 Bisnis")

        for _, r in biz.iterrows():
            st.write(
                f"✈️ {r['airline']} ({r['flight']}) | "
                f"{r['stops_clean']} | ⏱ {format_duration(r['duration'])} | "
                f"💰 {format_inr(r['price'])}"
            )

        st.subheader("🧠 Insight AI")

        for i in generate_insight(eco, biz):
            st.write(i)