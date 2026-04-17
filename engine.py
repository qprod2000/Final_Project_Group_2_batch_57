# ============================================================
# engine.py — Mesin rekomendasi penerbangan
#
# Alur:
#   1. Filter data berdasarkan input user (rute + hari)
#   2. Siapkan fitur sesuai preprocessing notebook (FP_GROUP_2)
#   3. Prediksi harga via model .pkl (joblib)  ← SLOT MODEL
#   4. Fallback ke harga aktual jika model belum tersedia
#   5. Ranking & kembalikan Top-N per kelas
# ============================================================

import pandas as pd # type: ignore
import numpy as np  # type: ignore

from config import (
    TOP_N,
    STOPS_MAPPING,
    TIME_MAPPING,
    CLASS_MAPPING,
    AIRLINES,
    CITIES,
)


# ── Preprocessing ──────────────────────────────────────────
def _preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Terapkan transformasi yang sama persis dengan notebook:
    - Map class, stops, departure_time, arrival_time ke angka
    - One-hot encode airline, source_city, destination_city
    - Drop kolom yang tidak dipakai model
    """
    df = df_input.copy()

    # Drop kolom yang tidak masuk ke model
    drop_cols = [c for c in ["index", "flight", "price"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Encode ordinal
    df["class"]          = df["class"].map(CLASS_MAPPING).fillna(0).astype(int)
    df["stops"]          = df["stops"].map(STOPS_MAPPING).fillna(0).astype(int)
    df["departure_time"] = df["departure_time"].map(TIME_MAPPING).fillna(0).astype(int)
    df["arrival_time"]   = df["arrival_time"].map(TIME_MAPPING).fillna(0).astype(int)

    # One-hot encode sesuai urutan fit saat training
    df = pd.get_dummies(df, columns=["airline", "source_city", "destination_city"])

    # Pastikan semua kolom one-hot yang ada di training juga ada di sini
    # (kolom yang hilang diisi 0, kolom ekstra di-drop)
    expected_ohe = (
        [f"airline_{a}"          for a in sorted(AIRLINES)] +
        [f"source_city_{c}"      for c in sorted(CITIES)]   +
        [f"destination_city_{c}" for c in sorted(CITIES)]
    )
    for col in expected_ohe:
        if col not in df.columns:
            df[col] = 0

    # Buang kolom OHE yang tidak ada di training
    extra = [c for c in df.columns
             if (c.startswith("airline_") or c.startswith("source_city_") or
                 c.startswith("destination_city_"))
             and c not in expected_ohe]
    df = df.drop(columns=extra, errors="ignore")

    return df


# ── Scoring fallback (tanpa model) ─────────────────────────
def _score_no_model(df_filtered: pd.DataFrame, days_left: int) -> pd.Series:
    """
    Rule-based score jika model .pkl belum tersedia.
    Score lebih rendah = lebih direkomendasikan.
    Bobot: harga 60% + durasi 25% + urgency 15%
    """
    prices = df_filtered["price"].values.astype(float)
    durs   = df_filtered["duration"].values.astype(float)

    p_min, p_range = prices.min(), prices.max() - prices.min() or 1.0
    d_min, d_range = durs.min(),   durs.max()   - durs.min()   or 1.0

    norm_p = (prices - p_min) / p_range
    norm_d = (durs   - d_min) / d_range

    urgency = 0.3 if days_left <= 3 else 0.1 if days_left <= 7 else 0.0
    score   = norm_p * 0.60 + norm_d * 0.25 + norm_p * urgency * 0.15

    return pd.Series(score, index=df_filtered.index)


# ── Main function ───────────────────────────────────────────
def find_best_flights(
    df: pd.DataFrame,
    model,          # joblib model atau None
    input_data: dict,
    top_n: int = TOP_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parameter
    ---------
    df         : DataFrame asli dari CSV
    model      : model hasil joblib.load(), atau None jika belum tersedia
    input_data : dict dengan key 'source_city', 'destination_city', 'days_left'
    top_n      : jumlah rekomendasi per kelas

    Return
    ------
    (eco_top, biz_top) — dua DataFrame, masing-masing top_n baris
    """
    src   = input_data.get("source_city")
    dst   = input_data.get("destination_city")
    days  = int(input_data.get("days_left", 10))

    # 1. Filter rute
    mask = (df["source_city"] == src) & (df["destination_city"] == dst)
    df_route = df[mask].copy()

    if df_route.empty:
        empty = pd.DataFrame()
        return empty, empty

    # 2. Tambahkan days_left dari slider ke setiap baris
    df_route["days_left"] = days

    # 3. Prediksi harga atau fallback
    if model is not None:
        try:
            X = _preprocess(df_route)

            # Selaraskan kolom dengan fitur yang dipakai model saat training
            if hasattr(model, "feature_names_in_"):
                train_cols = list(model.feature_names_in_)
                for c in train_cols:
                    if c not in X.columns:
                        X[c] = 0
                X = X[train_cols]

            df_route["predicted_price"] = model.predict(X)
            sort_col = "predicted_price"

        except Exception as e:
            # Jika prediksi gagal, gunakan harga aktual
            df_route["predicted_price"] = df_route["price"]
            sort_col = "price"
    else:
        # ── SLOT MODEL BELUM DI-LOAD ──
        # Gunakan rule-based scoring dari harga aktual
        df_route["score"] = _score_no_model(df_route, days)
        sort_col = "score"

    # 4. Bersihkan kolom tampilan
    df_route["stops_clean"] = df_route["stops"].map(
        lambda x: {"zero": "Non-stop", "one": "1 Henti", "two_or_more": "2+ Henti"}.get(x, x)
    )

    # 5. Split per kelas & ranking
    def _top(class_name: str) -> pd.DataFrame:
        sub = df_route[df_route["class"] == class_name].copy()
        if sub.empty:
            return sub
        sub = sub.sort_values(sort_col, ascending=True)
        # Deduplikasi: ambil 1 baris terbaik per maskapai
        sub = sub.drop_duplicates(subset=["airline"], keep="first")
        return sub.head(top_n).reset_index(drop=True)

    eco = _top("Economy")
    biz = _top("Business")

    return eco, biz