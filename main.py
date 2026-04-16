import joblib   # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
import pandas.api.types as ptypes   # type: ignore

MODEL_PATH = "model_tiket.pkl"
META_PATH = "model_meta.pkl"

# =========================
# GLOBAL MAP
# =========================
display_map = {
    "0 stops": "Langsung",
    "1 stop": "1 Transit",
    "2 stops": "2 Transit",
    "3 stops": "3 Transit",
    "zero": "Langsung",
    "one": "1 Transit",
    "two or more": "2+ Transit",
    "Langsung": "Langsung",
    "1 Transit": "1 Transit",
    "2 Transit": "2 Transit",
    "3 Transit": "3 Transit",
    "2+ Transit": "2+ Transit"
}

reverse_map = {
    "Langsung": "0 stops",
    "1 Transit": "1 stop",
    "2 Transit": "2 stops",
    "3 Transit": "3 stops",
    "2+ Transit": "2 stops"
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
# LOAD DATA
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
    "days_left": "Sisa Hari Pemesanan",
    "class": "Kelas Penerbangan"
}


# =========================
# AI ENGINE (FINAL)
# =========================
def find_best_flights(df, model, input_data, top_n=5):

    flight_options = df[["airline", "flight", "stops", "duration"]].drop_duplicates()

    if len(flight_options) > 1000:
        flight_options = flight_options.sample(500, random_state=42)

    results = []

    for _, row in flight_options.iterrows():
        temp = input_data.copy()

        temp["airline"] = row["airline"]
        temp["flight"] = row["flight"]
        temp["duration"] = row["duration"]

        raw = str(row["stops"]).lower().strip()
        normalized = input_stops_map.get(raw, row["stops"])

        temp["stops"] = reverse_map.get(normalized, normalized)

        try:
            temp_df = pd.DataFrame([temp])

            # 🔥 AUTO ALIGN FEATURE
            if hasattr(model, "feature_names_in_"):
                for col in model.feature_names_in_:
                    if col not in temp_df.columns:
                        temp_df[col] = 0

                temp_df = temp_df[model.feature_names_in_]

            pred = model.predict(temp_df)[0]

            results.append({
                "airline": row["airline"],
                "flight": row["flight"],
                "stops": normalized,
                "duration": row["duration"],
                "price": pred
            })

        except:
            continue

    # 🔥 FALLBACK (anti kosong)
    if len(results) == 0:
        fallback = df.sample(min(5, len(df)))

        results = []
        for _, row in fallback.iterrows():
            results.append({
                "airline": row.get("airline", "-"),
                "flight": row.get("flight", "-"),
                "stops": input_stops_map.get(str(row.get("stops", "")).lower(), "-"),
                "duration": row.get("duration", 1),
                "price": row.get("price", 0)
            })

        return results

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

# =========================
# ROUTE
# =========================
st.subheader("✈️ Rute Penerbangan")

col_r1, col_r2 = st.columns(2)

source_col = next((c for c in ["source", "source_city"] if c in df.columns), None)
dest_col = next((c for c in ["destination", "destination_city"] if c in df.columns), None)

if source_col:
    input_data[source_col] = col_r1.selectbox(
        "Kota Asal",
        sorted(df[source_col].dropna().unique())
    )

if dest_col:
    input_data[dest_col] = col_r2.selectbox(
        "Kota Tujuan",
        sorted(df[dest_col].dropna().unique())
    )


# =========================
# INPUT
# =========================
feature_cols = df.drop(columns=["price"]).columns

col1, col2 = st.columns(2)

for i, col in enumerate(feature_cols):

    if col in ["airline", "flight", "duration", "source", "destination", "source_city", "destination_city"]:
        continue

    container = col1 if i % 2 == 0 else col2
    label = label_map.get(col, col)

    # 🔥 STOPS
    if col.lower() == "stops":

        raw_options = sorted(df[col].dropna().astype(str).unique())

        display_options = [
            input_stops_map.get(opt.lower(), opt)
            for opt in raw_options
        ]

        selected = container.selectbox(label, display_options)
        input_data[col] = reverse_input_map.get(selected, selected)

    # 🔥 DAYS LEFT
    elif col.lower() == "days_left":

        val = container.slider(label, 0.0, 30.0, 10.0, step=0.5)

        d = int(val)
        h = int((val - d) * 24)

        container.caption(f"{d} hari {h} jam")
        input_data[col] = val

    # 🔥 NUMERIC
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

    # 🔥 CATEGORICAL
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

        stops_label = display_map.get(r["stops"], r["stops"])
        duration_label = format_duration(r["duration"])

        st.write(
            f"{i}. ✈️ {r['airline']} ({r['flight']}) | "
            f"{stops_label} | ⏱ {duration_label} | 💰 Rp {int(r['price']):,}"
        )