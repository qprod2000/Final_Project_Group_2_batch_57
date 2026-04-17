import pandas as pd #type: ignore
from utils import normalize_stops

def find_best_flights(df, model, input_data, top_n=3):

    df = df.copy()
    df["stops_clean"] = df["stops"].apply(normalize_stops)

    # =========================
    # STRATIFIED SAMPLING
    # =========================
    df_work = (
        df.groupby("stops_clean", group_keys=False)
        .apply(lambda x: x.sample(min(60, len(x)), random_state=42))
    )

    # =========================
    # PREDICTION
    # =========================
    df_pred = df_work.copy()

    for k, v in input_data.items():
        df_pred[k] = v

    if hasattr(model, "feature_names_in_"):
        for col in model.feature_names_in_:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred = df_pred[model.feature_names_in_]

    preds = model.predict(df_pred)
    df_work["price"] = preds

    # =========================
    # VALUE-BASED SCORING 🔥
    # =========================
    df_work["price_norm"] = (
        (df_work["price"] - df_work["price"].min()) /
        (df_work["price"].max() - df_work["price"].min() + 1e-6)
    )

    df_work["duration_norm"] = (
        (df_work["duration"] - df_work["duration"].min()) /
        (df_work["duration"].max() - df_work["duration"].min() + 1e-6)
    )

    def transit_bonus(row):
        if row["stops_clean"] == "Langsung":
            return 0
        elif row["stops_clean"] == "1 Transit":
            return -0.05
        else:
            return -0.08

    df_work["score"] = (
        df_work["price_norm"] +
        (df_work["duration_norm"] * 0.6)
    ) + df_work.apply(transit_bonus, axis=1)

    df_work = df_work.sort_values("score")

    eco = df_work[df_work["class"].str.lower() == "economy"].head(top_n)
    biz = df_work[df_work["class"].str.lower() == "business"].head(top_n)

    return eco, biz