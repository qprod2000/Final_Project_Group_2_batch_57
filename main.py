import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
import pandas.api.types as ptypes   # type: ignore

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
    "2 stops": "2 Transit",
    "2+ stops": "2+ Transit"
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
def format_duration(hours_float):
    h = int(hours_float)
    m = int((hours_float - h) * 60)
    return f"{h} jam {m} menit"


# =========================
# LOAD
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("airlines_flights_data.csv")

    df = df.drop(columns=[col for col in ["index"] if col in df.columns])
    df = df.replace(["None", "nan", ""], pd.NA)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str)

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

    if "class" in df.columns:
        df["class"] = df["class"].replace({
            "Economy": "Ekonomi",
            "Business": "Bisnis"
        })

    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = joblib.load(META_PATH)
    return model, meta


# =========================
# LABEL
# =========================
label_map = {
    "source": "Kota Asal",
    "destination": "Kota Tujuan",
    "departure_time": "Waktu Keberangkatan",
    "arrival_time": "Waktu Kedatangan",
    "stops": "Jumlah Transit",
    "days_left": "Sisa Hari",
    "class": "Kelas"
}


# =========================
# AI ENGINE (FIX FINAL)
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    user_stops = input_data.get("stops", None)

    # 🔥 FILTER SESUAI PILIHAN USER
    if user_stops:
        mapped = reverse_input_map.get(user_stops, user_stops)
        filtered_df = df[df["stops"] == mapped]
    else:
        filtered_df = df

    flight_options = filtered_df[["airline", "flight", "stops", "duration"]].drop_duplicates()

    # fallback jika kosong
    if len(flight_options) == 0:
        flight_options = df.sample(50)

    results = []

    for _, row in flight_options.iterrows():
        temp = input_data.copy()

        temp["airline"] = row["airline"]
        temp["flight"] = row["flight"]
        temp["duration"] = row["duration"]
        temp["stops"] = row["stops"]  # 🔥 TIDAK DI-OVERRIDE

        try:
            temp_df = pd.DataFrame([temp])

            if hasattr(model, "feature_names_in_"):
                for col in model.feature_names_in_:
                    if col not in temp_df.columns:
                        temp_df[col] = 0
                temp_df = temp_df[model.feature_names_in_]

            pred = model.predict(temp_df)[0]

            results.append({
                "airline": row["airline"],
                "flight": row["flight"],
                "stops": input_stops_map.get(str(row["stops"]).lower(), row["stops"]),
                "duration": row["duration"],
                "price": pred
            })

        except:
            continue

    return sorted(results, key=lambda x: x["price"])[:top_n]


# =========================
# UI
# =========================
st.set_page_config(page_title="AI Flight Advisor", layout="wide")
st.title("✈️ AI Flight Price Advisor")

df = load_data()
model, meta = load_model()

st.success(f"Model: {meta['model']} | MAE: {meta['mae']:.2f}")

input_data = {}

# ROUTE
col1, col2 = st.columns(2)
input_data["source"] = col1.selectbox("Kota Asal", sorted(df["source"].unique()))
input_data["destination"] = col2.selectbox("Kota Tujuan", sorted(df["destination"].unique()))

# INPUT LAIN
col3, col4 = st.columns(2)

# stops
raw_stops = sorted(df["stops"].unique())
display_stops = [input_stops_map.get(str(x).lower(), x) for x in raw_stops]
selected = col3.selectbox("Jumlah Transit", display_stops)
input_data["stops"] = selected

# days
days = col4.slider("Sisa Hari", 0.0, 30.0, 10.0, step=0.5)
input_data["days_left"] = days

# =========================
# PREDICT
# =========================
if st.button("🔍 Cari Penerbangan Terbaik"):

    results = find_best_flights(df, model, input_data)

    st.subheader("🏆 Rekomendasi")

    for i, r in enumerate(results, 1):
        st.write(
            f"{i}. ✈️ {r['airline']} ({r['flight']}) | "
            f"{r['stops']} | ⏱ {format_duration(r['duration'])} | "
            f"💰 Rp {int(r['price']):,}"
        )