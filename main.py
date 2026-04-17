import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore

MODEL_PATH = "model_tiket.pkl"

# =========================
# NORMALIZE TRANSIT
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

    df["stops_clean"] = df["stops"].apply(normalize_stops)

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================
# AI ENGINE (FINAL)
# =========================
def find_best_flights(df, model, input_data, top_n=3):

    df = df.copy()

    # ensure column exists
    df["stops_clean"] = df["stops"].apply(normalize_stops)

    # =========================
    # STRATIFIED SAMPLING
    # =========================
    df_work = (
        df.groupby("stops_clean", group_keys=False)
        .apply(lambda x: x.sample(min(60, len(x)), random_state=42))
    )

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
# VALUE-BASED SCORING 🔥 FINAL
# =========================

# normalisasi
df_work["price_norm"] = (
    (df_work["price"] - df_work["price"].min()) /
    (df_work["price"].max() - df_work["price"].min() + 1e-6)
)

df_work["duration_norm"] = (
    (df_work["duration"] - df_work["duration"].min()) /
    (df_work["duration"].max() - df_work["duration"].min() + 1e-6)
)

# =========================
# 🔥 VALUE METRIC (KUNCI UTAMA)
# =========================
# semakin kecil = semakin worth it

df_work["value_score"] = (
    df_work["price_norm"] +
    (df_work["duration_norm"] * 0.6)
)

# =========================
# 🔥 TRANSIT BONUS (INI YANG MEMBUAT TRANSIT BISA MENANG)
# =========================

def transit_adjustment(row):
    if row["stops_clean"] == "Langsung":
        return 0
    elif row["stops_clean"] == "1 Transit":
        return -0.05  # bonus kecil
    else:
        return -0.08  # bonus lebih besar

df_work["transit_bonus"] = df_work.apply(transit_adjustment, axis=1)

# =========================
# FINAL SCORE
# =========================
df_work["score"] = df_work["value_score"] + df_work["transit_bonus"]


def generate_insight(eco, biz):

    insights = []

    if eco.empty or biz.empty:
        return ["Tidak cukup data"]

    eco_best = eco.iloc[0]
    biz_best = biz.iloc[0]

    # =========================
    # TRANSIT INSIGHT 🔥
    # =========================
    if eco_best["stops_clean"] != "Langsung":

        insights.append(
            f"✈️ Transit ({eco_best['stops_clean']}) dipilih karena memberikan value terbaik (lebih hemat dibanding direct)"
        )

        if eco_best["stops_clean"] == "2 Transit":
            insights.append("⚠️ Perjalanan lebih panjang, cocok jika prioritas harga")

    else:
        insights.append("⚡ Direct flight unggul karena efisiensi waktu")

    # =========================
    # BUSINESS ANALYSIS
    # =========================
    price_diff = biz_best["price"] - eco_best["price"]
    time_diff = eco_best["duration"] - biz_best["duration"]

    if price_diff > 0 and time_diff > 0:
        percent = (price_diff / eco_best["price"]) * 100

        insights.append(
            f"💺 Bisnis lebih cepat {int(time_diff*60)} menit dengan tambahan {format_inr(price_diff)} (+{percent:.1f}%)"
        )

        if percent < 20:
            insights.append("🔥 Upgrade ke Bisnis cukup worth it")
        else:
            insights.append("⚖️ Upgrade ke Bisnis kurang optimal")

    elif price_diff < 0:
        insights.append("🔥 Bisnis lebih murah dari ekonomi (rare case)")

    # =========================
    # FINAL DECISION
    # =========================
    insights.append("🎯 AI memilih berdasarkan keseimbangan harga vs waktu (value-based decision)")

    return insights


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Final AI Version)")

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