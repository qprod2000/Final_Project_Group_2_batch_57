import pandas as pd # type: ignore
import joblib   # type: ignore  

def find_best_flights(df, model, input_data, preference=50):
    df = df.copy()

    df = df.sample(min(300, len(df)), random_state=42)

    df["price"] = model.predict(df[model.feature_names_in_])

    base_price = df["price"].mean()
    base_duration = df["duration"].mean()

    penalty = 20 + (100 - preference)

    df["value_score"] = (
        (base_price - df["price"]) -
        ((df["duration"] - base_duration) * penalty)
    )

    eco = df[df["class"] == "Economy"].sort_values("value_score", ascending=False).head(3)
    biz = df[df["class"] == "Business"].sort_values("value_score", ascending=False).head(3)

    return eco, biz, base_price, base_duration


def explain(row, base_price, base_duration):
    saving = base_price - row["price"]
    extra_time = row["duration"] - base_duration

    if saving > 0:
        return f"Hemat ₹{int(saving)} (+{extra_time:.1f} jam)"
    else:
        return f"Lebih cepat {abs(extra_time):.1f} jam"