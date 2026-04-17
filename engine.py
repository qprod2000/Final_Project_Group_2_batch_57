import pandas as pd # type: ignore
import numpy as np  # type: ignore

def find_best_flights(df, model, input_data, preference=50):
    df = df.copy()

    # Sampling biar cepat
    df = df.sample(min(300, len(df)), random_state=42)

    # Predict price
    df["predicted_price"] = model.predict(df[input_data.keys()])

    # Baseline (untuk perbandingan)
    base_price = df["price"].mean()
    base_duration = df["duration"].mean()

    # Hitung value score
    penalty = 20 + (100 - preference)

    df["price_saving"] = base_price - df["price"]
    df["extra_time"] = df["duration"] - base_duration

    df["value_score"] = df["price_saving"] - (df["extra_time"] * penalty)

    # Pisahkan kelas
    eco = df[df["class"] == "Economy"].sort_values("value_score", ascending=False).head(3)
    biz = df[df["class"] == "Business"].sort_values("value_score", ascending=False).head(3)

    return eco, biz, base_price, base_duration


def explain(row, base_price, base_duration):
    saving = base_price - row["price"]
    extra_time = row["duration"] - base_duration

    if saving > 0:
        return f"Hemat ₹{int(saving)} (+{extra_time:.1f} jam)"
    else:
        return f"Lebih cepat ({abs(extra_time):.1f} jam lebih singkat)"