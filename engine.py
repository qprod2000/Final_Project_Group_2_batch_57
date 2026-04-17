import pandas as pd # type: ignore
import joblib   # type: ignore
import streamlit as st  # type: ignore
import numpy as np  # type: ignore
from utils import normalize_stops


def find_best_flights(df, model, input_data, preference=50, top_n=3):
    df = df.copy()

    # normalize stops
    df["stops_clean"] = df["stops"].apply(normalize_stops)

    # sampling biar cepat
    df = df.sample(min(300, len(df)), random_state=42)

    # ====== PREPARE FEATURE ======
    df_pred = df.copy()
    for k, v in input_data.items():
        df_pred[k] = v

    # align dengan model
    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df_pred.columns:
                df_pred[col] = 0
        df_pred = df_pred[model.feature_names_in_]

    # ====== PREDICT ======
    df["price"] = model.predict(df_pred)

    # ====== BASELINE (DIRECT) ======
    direct = df[df["stops_clean"] == "Langsung"]
    base_price = direct["price"].min() if not direct.empty else df["price"].median()
    base_duration = direct["duration"].min() if not direct.empty else df["duration"].median()

    # ====== VALUE SCORE ======
    # preferensi: 0 = hemat, 100 = cepat
    penalty = 30 + (100 - preference)

    df["price_saving"] = base_price - df["price"]
    df["extra_time"] = df["duration"] - base_duration

    df["value_score"] = df["price_saving"] - (df["extra_time"] * penalty)

    # ====== SORT ======
    df = df.sort_values("value_score", ascending=False)

    eco = df[df["class"].str.lower() == "economy"].head(top_n)
    biz = df[df["class"].str.lower() == "business"].head(top_n)

    return eco, biz, base_price, base_duration


def explain(row, base_price, base_duration):
    saving = base_price - row["price"]
    extra_time = row["duration"] - base_duration

    if saving > 0:
        return f"Hemat ₹{int(saving)} walaupun +{extra_time:.1f} jam"
    else:
        return f"Lebih cepat {abs(extra_time):.1f} jam walaupun lebih mahal"