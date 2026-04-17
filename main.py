import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore

MODEL_PATH = "model_tiket.pkl"

# =========================
# MAP
# =========================
display_map = {
    "0 stops": "Langsung",
    "1 stop": "1 Transit",
    "2 stops": "2 Transit",
    "2+ stops": "2+ Transit",
    "zero": "Langsung",
    "one": "1 Transit",
    "two": "2 Transit",
    "two or more": "2+ Transit",
}

input_stops_map = {
    "zero": "Langsung",
    "one": "1 Transit",
    "two": "2 Transit",
    "two or more": "2+ Transit",
    "0 stops": "Langsung",
    "1 stop": "1 Transit",
    "2 stops": "2 Transit",
    "2+ stops": "2+ Transit",
}

reverse_input_map = {
    "Langsung": "zero",
    "1 Transit": "one",
    "2 Transit": "two",
    "2+ Transit": "two or more",
}

# =========================
# HELPER
# =========================
def format_duration(x):
    h = int(x)
    m = int((x - h) * 60)
    return f"{h} jam {m} menit"

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

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# =========================
# AI ENGINE (FAST VERSION)
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    df_work = df.copy()

    # =========================
    # FILTER TRANSIT (SMART)
    # =========================
    user_stops = input_data.get("stops")

    if user_stops:
        mapped = reverse_input_map.get(user_stops, user_stops)
        filtered = df_work[df_work["stops"] == mapped]

        if len(filtered) > 10:
            df_work = filtered

    # =========================
    # SAMPLING (CEPAT)
    # =========================
    df_work = df_work.sample(min(150, len(df_work)), random_state=42)

    # =========================
    # PREPARE INPUT SEKALI
    # =========================
    df_pred = df_work.copy()

    for k, v in input_data.items():
        df_pred[k] = v

    # align feature ke model
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred = df_pred[model.feature_names_in_]

    # =========================
    # 🔥 PREDICT SEKALI
    # =========================
    try:
        preds = model.predict(df_pred)
    except:
        return pd.DataFrame(), pd.DataFrame()

    df_work["price"] = preds

    # =========================
    # SCORING AI
    # =========================
    df_work["price_norm"] = (df_work["price"] - df_work["price"].min()) / (df_work["price"].max() - df_work["price"].min() + 1e-6)
    df_work["duration_norm"] = (df_work["duration"] - df_work["duration"].min()) / (df_work["duration"].max() - df_work["duration"].min() + 1e-6)

    df_work["score"] = df_work["price_norm"] * 0.7 + df_work["duration_norm"] * 0.3

    df_work = df_work.sort_values("score")

    return df_work.head(top_n), df_work


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Optimizer", layout="wide")
st.title("✈️ AI Flight Optimizer (Fast Version)")

df = load_data()
model = load_model()

input_data = {}

# =========================
# ROUTE (AUTO DETECT)
# =========================
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

# =========================
# STOPS
# =========================
raw_stops = sorted(df["stops"].unique())
display_stops = [input_stops_map.get(str(x).lower(), x) for x in raw_stops]

selected = st.selectbox("Jumlah Transit", display_stops)
input_data["stops"] = selected

# =========================
# DAYS
# =========================
input_data["days_left"] = st.slider("Sisa Hari", 0, 30, 10, step=1)

# =========================
# RUN
# =========================
if st.button("🚀 Cari Rekomendasi Terbaik"):

    top, all_data = find_best_flights(df, model, input_data)

    if top.empty:
        st.warning("Tidak ada hasil ditemukan, coba ubah parameter")
    else:
        cheapest = all_data.loc[all_data["price"].idxmin()]
        fastest = all_data.loc[all_data["duration"].idxmin()]

        st.subheader("🏆 Rekomendasi Terbaik")

        for i, r in top.iterrows():

            tag = ""

            if r["flight"] == cheapest["flight"]:
                tag += " 💰 Termurah"
            if r["flight"] == fastest["flight"]:
                tag += " ⚡ Tercepat"
            if i == top.index[0]:
                tag += " ⭐ Best Value"

            st.write(
                f"✈️ {r['airline']} ({r['flight']}) | "
                f"{display_map.get(r['stops'], r['stops'])} | "
                f"⏱ {format_duration(r['duration'])} | "
                f"💺 {format_class(r['class'])} | "
                f"💰 INR {int(r['price']):,}"
                f"{tag}"
            )