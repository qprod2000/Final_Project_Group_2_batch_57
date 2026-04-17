# ============================================================
# config.py — Konfigurasi global AI Flight Optimizer
# ============================================================

# Path ke file model yang sudah di-train (joblib)
# Ganti dengan path aktual file .pkl kamu
MODEL_PATH = "model/random_forest.pkl"

# Path ke dataset CSV
DATA_PATH = "airlines_flights_data.csv"

# Jumlah rekomendasi yang ditampilkan per kelas
TOP_N = 3

# Mapping stops: teks → angka (sesuai preprocessing di notebook)
STOPS_MAPPING = {
    "zero":         0,
    "one":          1,
    "two_or_more":  2,
}

# Mapping waktu keberangkatan/kedatangan → angka
TIME_MAPPING = {
    "Early_Morning": 0,
    "Morning":       1,
    "Afternoon":     2,
    "Evening":       3,
    "Night":         4,
    "Late_Night":    5,
}

# Mapping kelas penerbangan → angka
CLASS_MAPPING = {
    "Economy":  0,
    "Business": 1,
}

# Label tampilan untuk stops
STOPS_LABEL = {
    "zero":        "Non-stop",
    "one":         "1 Henti",
    "two_or_more": "2+ Henti",
}

# Label tampilan untuk waktu
TIME_LABEL = {
    "Early_Morning": "Subuh (03–06)",
    "Morning":       "Pagi (06–12)",
    "Afternoon":     "Siang (12–18)",
    "Evening":       "Sore (18–21)",
    "Night":         "Malam (21–00)",
    "Late_Night":    "Larut (00–03)",
}

# Semua maskapai yang ada di dataset
AIRLINES = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]

# Semua kota yang ada di dataset
CITIES = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]