import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore

MODEL_PATH = "model_tiket.pkl"
META_PATH = "model_meta.pkl"

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
    "two": "2 Transit",
    "more than two": "2+ Transit",
    "two or more": "2+ Transit",
    "0 stops": "Langsung",
    "1 stop": "1 Transit",
    "2 stops": "2 Transit"
}

reverse_input_map = {
    "Langsung": "zero",
    "1 Transit": "one",
    "2 Transit": "two",
    "2+ Transit": "two or more"
}

# =========================
# HELPER
# =========================
def format_duration(x):
    h = int(x)
    m = int((x - h) * 60)
    return f"{h} jam {m} menit"


# =========================
# LOAD DATA
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
# AI ENGINE
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    user_stops = input_data.get("stops")

    if user_stops:
        mapped = reverse_input_map.get(user_stops, user_stops)
        filtered_df = df[df["stops"] == mapped]
    else:
        filtered_df = df

    if len(filtered_df) == 0:
        filtered_df = df.sample(50)

    options = filtered_df[["airline", "flight", "stops", "duration"]].drop_duplicates()

    results = []

    for _, row in options.iterrows():
        temp = input_data.copy()

        temp["airline"] = row["airline"]
        temp["flight"] = row["flight"]
        temp["duration"] = row["duration"]
        temp["stops"] = row["stops"]

        try:
            pred = model.predict(pd.DataFrame([temp]))[0]

            results.append({
                "airline": row["airline"],
                "flight": row["flight"],
                "stops": input_stops_map.get(str(row["stops"]).lower(), row["stops"]),
                "duration": row["duration"],
                "price": pred
            })

        except:
            continue

    if len(results) == 0:
        return df.sample(5).to_dict("records")

    return sorted(results, key=lambda x: x["price"])[:top_n]


# =========================
# UI
# =========================
st.title("✈️ AI Flight Price Advisor")

df = load_data()
model = load_model()

input_data = {}

# =========================
# ROUTE (FIX FINAL)
# =========================
col1, col2 = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

if source_col:
    input_data[source_col] = col1.selectbox("Kota Asal", sorted(df[source_col].unique()))
else:
    st.error("Kolom kota asal tidak ditemukan")

if dest_col:
    input_data[dest_col] = col2.selectbox("Kota Tujuan", sorted(df[dest_col].unique()))
else:
    st.error("Kolom kota tujuan tidak ditemukan")

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
input_data["days_left"] = st.slider("Sisa Hari", 0.0, 30.0, 10.0, step=0.5)

# =========================
# PREDICT
# =========================
if st.button("🔍 Cari Penerbangan Terbaik"):

    results = find_best_flights(df, model, input_data)

    st.subheader("🏆 Rekomendasi")

    for i, r in enumerate(results, 1):

        stops_label = display_map.get(r["stops"], r["stops"])

        st.write(
            f"{i}. ✈️ {r['airline']} ({r['flight']}) | "
            f"{stops_label} | ⏱ {format_duration(r['duration'])} | "
            f"💰 Rp {int(r['price']):,}"
        )