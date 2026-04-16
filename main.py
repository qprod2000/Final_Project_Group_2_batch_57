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
    "3 stops": "3 Transit",
    "zero": "Langsung",
    "one": "1 Transit",
    "two or more": "2+ Transit"
}

input_stops_map = {
    "zero": "Langsung",
    "one": "1 Transit",
    "two or more": "2+ Transit",
    "0 stops": "Langsung",
    "1 stop": "1 Transit",
    "2 stops": "2 Transit"
}

reverse_input_map = {
    "Langsung": "zero",
    "1 Transit": "one",
    "2 Transit": "two or more",
    "2+ Transit": "two or more"
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
# AI ENGINE (SMART)
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    user_stops = input_data.get("stops")

    if user_stops:
        mapped = reverse_input_map.get(user_stops, user_stops)
        df = df[df["stops"] == mapped]

    if len(df) == 0:
        df = df.sample(50)

    options = df[["airline", "flight", "stops", "duration", "class"]].drop_duplicates()

    results = []

    for _, row in options.iterrows():
        temp = input_data.copy()

        temp["airline"] = row["airline"]
        temp["flight"] = row["flight"]
        temp["duration"] = row["duration"]
        temp["stops"] = row["stops"]
        temp["class"] = row["class"]

        try:
            pred = model.predict(pd.DataFrame([temp]))[0]

            results.append({
                "airline": row["airline"],
                "flight": row["flight"],
                "stops": row["stops"],
                "duration": row["duration"],
                "class": row["class"],
                "price": pred
            })
        except:
            continue

    if len(results) == 0:
        return []

    df_res = pd.DataFrame(results)

    # =========================
    # SCORING SYSTEM
    # =========================
    df_res["price_norm"] = (df_res["price"] - df_res["price"].min()) / (df_res["price"].max() - df_res["price"].min() + 1e-6)
    df_res["duration_norm"] = (df_res["duration"] - df_res["duration"].min()) / (df_res["duration"].max() - df_res["duration"].min() + 1e-6)

    # 🔥 Best value (bisa di-tune)
    df_res["score"] = df_res["price_norm"] * 0.7 + df_res["duration_norm"] * 0.3

    df_res = df_res.sort_values("score")

    return df_res.head(top_n), df_res


# =========================
# UI
# =========================
st.title("✈️ AI Flight Optimizer")

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

# STOPS
raw_stops = sorted(df["stops"].unique())
display_stops = [input_stops_map.get(str(x).lower(), x) for x in raw_stops]

selected = st.selectbox("Jumlah Transit", display_stops)
input_data["stops"] = selected

# DAYS
input_data["days_left"] = st.slider("Sisa Hari", 0.0, 30.0, 10.0, step=0.5)

# =========================
# RUN
# =========================
if st.button("🚀 Cari Rekomendasi Terbaik"):

    top, all_data = find_best_flights(df, model, input_data)

    if len(top) == 0:
        st.warning("Tidak ada hasil")
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
                f"💰 Rp {int(r['price']):,}"
                f"{tag}"
            )