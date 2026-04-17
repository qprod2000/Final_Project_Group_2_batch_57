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

def format_class(c):
    return "Bisnis" if str(c).lower() == "business" else "Ekonomi"


# =========================
# LOAD
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
# AI ENGINE (BALANCED)
# =========================
def find_best_flights(df, model, input_data, top_n=3):

    # 🔥 STRATIFIED SAMPLING (FIX UTAMA)
    df_work = (
        df.groupby("stops_clean", group_keys=False)
        .apply(lambda x: x.sample(min(60, len(x)), random_state=42))
    )

    # =========================
    # PREPARE
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
    # BALANCED SCORING ⚖️
    # =========================
    df_work["price_norm"] = (df_work["price"] - df_work["price"].min()) / (df_work["price"].max() - df_work["price"].min() + 1e-6)
    df_work["duration_norm"] = (df_work["duration"] - df_work["duration"].min()) / (df_work["duration"].max() - df_work["duration"].min() + 1e-6)

    stops_weight = {
        "Langsung": 0.0,
        "1 Transit": 0.2,
        "2 Transit": 0.4
    }

    df_work["stops_penalty"] = df_work["stops_clean"].map(stops_weight)

    df_work["score"] = (
        df_work["price_norm"] * 0.6 +
        df_work["duration_norm"] * 0.3 +
        df_work["stops_penalty"] * 0.1
    )

    df_work = df_work.sort_values("score")

    eco = df_work[df_work["class"].str.lower() == "economy"].head(top_n)
    biz = df_work[df_work["class"].str.lower() == "business"].head(top_n)

    return eco, biz, df_work


# =========================
# INSIGHT ENGINE
# =========================
def generate_insight(eco, biz):

    insights = []

    if not eco.empty and not biz.empty:

        eco_best = eco.iloc[0]
        biz_best = biz.iloc[0]

        price_diff = biz_best["price"] - eco_best["price"]
        time_diff = eco_best["duration"] - biz_best["duration"]

        if price_diff > 0 and time_diff > 0:
            insights.append(
                f"💡 Upgrade ke Bisnis: tambah {format_inr(price_diff)} untuk hemat {int(time_diff*60)} menit"
            )

        if eco_best["stops_clean"] != "Langsung":
            insights.append("💡 Transit memberikan opsi harga lebih hemat")

        insights.append("🎯 AI memilih berdasarkan keseimbangan harga, waktu, dan transit")

    return insights


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Balanced AI)")

df = load_data()
model = load_model()

input_data = {}

# ROUTE
col1, col2 = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

if source_col:
    input_data[source_col] = col1.selectbox("Kota Asal", sorted(df[source_col].unique()))

if dest_col:
    input_data[dest_col] = col2.selectbox("Kota Tujuan", sorted(df[dest_col].unique()))

# DAYS
input_data["days_left"] = st.slider("Sisa Hari", 0, 30, 10, step=1)

# RUN
if st.button("🚀 Cari Rekomendasi Terbaik"):

    eco, biz, all_data = find_best_flights(df, model, input_data)

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