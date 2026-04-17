# ============================================================
# main.py — AI Flight Optimizer (single-file, Streamlit Cloud ready)
# Jalankan: streamlit run main.py
# ============================================================

import os
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import streamlit as st  # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "airlines_flights_data.csv"
TOP_N     = 3

STOPS_MAPPING = {"zero": 0, "one": 1, "two_or_more": 2}
TIME_MAPPING  = {
    "Early_Morning": 0, "Morning": 1, "Afternoon": 2,
    "Evening": 3, "Night": 4, "Late_Night": 5,
}
CLASS_MAPPING = {"Economy": 0, "Business": 1}

STOPS_LABEL = {
    "zero": "Non-stop", "one": "1 Henti", "two_or_more": "2+ Henti"
}
TIME_LABEL = {
    "Early_Morning": "Subuh (03–06)", "Morning":    "Pagi (06–12)",
    "Afternoon":     "Siang (12–18)", "Evening":    "Sore (18–21)",
    "Night":         "Malam (21–00)", "Late_Night": "Larut (00–03)",
}

AIRLINES = ["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"]
CITIES   = ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]


# ============================================================
# UTILS
# ============================================================

def format_duration(hours: float) -> str:
    h = int(hours)
    m = round((hours - h) * 60)
    return f"{h}j {m}m" if m > 0 else f"{h}j"

def format_inr(amount: float) -> str:
    return f"₹{int(round(amount)):,}"

def stops_color(stops_raw: str) -> str:
    return {"zero": "#1D9E75", "one": "#E8C84A", "two_or_more": "#E24B4A"}.get(stops_raw, "#9593A0")


# ============================================================
# ENGINE
# ============================================================

def _preprocess(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df = df.drop(columns=[c for c in ["index", "flight", "price"] if c in df.columns])
    df["class"]          = df["class"].map(CLASS_MAPPING).fillna(0).astype(int)
    df["stops"]          = df["stops"].map(STOPS_MAPPING).fillna(0).astype(int)
    df["departure_time"] = df["departure_time"].map(TIME_MAPPING).fillna(0).astype(int)
    df["arrival_time"]   = df["arrival_time"].map(TIME_MAPPING).fillna(0).astype(int)
    df = pd.get_dummies(df, columns=["airline", "source_city", "destination_city"])
    expected_ohe = (
        [f"airline_{a}"          for a in sorted(AIRLINES)] +
        [f"source_city_{c}"      for c in sorted(CITIES)]   +
        [f"destination_city_{c}" for c in sorted(CITIES)]
    )
    for col in expected_ohe:
        if col not in df.columns:
            df[col] = 0
    extra = [c for c in df.columns
             if (c.startswith("airline_") or c.startswith("source_city_")
                 or c.startswith("destination_city_")) and c not in expected_ohe]
    return df.drop(columns=extra, errors="ignore")


def _score_no_model(df_filtered: pd.DataFrame, days_left: int) -> pd.Series:
    prices = df_filtered["price"].values.astype(float)
    durs   = df_filtered["duration"].values.astype(float)
    stops  = df_filtered["stops"].map(STOPS_MAPPING).fillna(0).values.astype(float)
    p_min, p_range = prices.min(), (prices.max() - prices.min()) or 1.0
    d_min, d_range = durs.min(),   (durs.max()   - durs.min())   or 1.0
    s_range        = stops.max() or 1.0
    norm_p  = (prices - p_min) / p_range
    norm_d  = (durs   - d_min) / d_range
    norm_s  = stops / s_range
    urgency = 0.3 if days_left <= 3 else 0.1 if days_left <= 7 else 0.0
    score   = norm_p * 0.55 + norm_d * 0.25 + norm_s * 0.10 + norm_p * urgency * 0.10
    return pd.Series(score, index=df_filtered.index)


def find_best_flights(df, model, input_data, top_n=TOP_N):
    src  = input_data.get("source_city")
    dst  = input_data.get("destination_city")
    days = int(input_data.get("days_left", 10))
    mask     = (df["source_city"] == src) & (df["destination_city"] == dst)
    df_route = df[mask].copy()
    if df_route.empty:
        return pd.DataFrame(), pd.DataFrame()
    df_route["days_left"] = days
    if model is not None:
        try:
            X = _preprocess(df_route)
            if hasattr(model, "feature_names_in_"):
                train_cols = list(model.feature_names_in_)
                for c in train_cols:
                    if c not in X.columns:
                        X[c] = 0
                X = X[train_cols]
            df_route["predicted_price"] = model.predict(X)
            sort_col = "predicted_price"
        except Exception:
            df_route["score"] = _score_no_model(df_route, days)
            sort_col = "score"
    else:
        df_route["score"] = _score_no_model(df_route, days)
        sort_col = "score"
    df_route["stops_clean"] = df_route["stops"].map(STOPS_LABEL)

    def _top(class_name):
        sub = df_route[df_route["class"] == class_name].copy()
        if sub.empty:
            return sub
        sub = sub.sort_values(sort_col, ascending=True)
        sub = sub.drop_duplicates(subset=["airline", "stops"], keep="first")
        return sub.head(top_n).reset_index(drop=True)

    return _top("Economy"), _top("Business")


# ============================================================
# STREAMLIT — Setup
# ============================================================

st.set_page_config(page_title="AI Flight Optimizer", page_icon="✈️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
#MainMenu, header, footer { visibility: hidden; }
.stApp { background-color: #0c0c0f; }
.app-title    { font-family:'Syne',sans-serif; font-size:26px; font-weight:700; color:#f0eeea; letter-spacing:-0.5px; margin-bottom:2px; }
.app-subtitle { font-family:'DM Mono',monospace; font-size:12px; color:#5a5870; margin-bottom:1.5rem; }
.stat-row  { display:flex; gap:10px; margin-bottom:1.5rem; }
.stat-card { flex:1; background:#13131a; border:0.5px solid rgba(255,255,255,0.08); border-radius:12px; padding:14px 16px; }
.stat-label{ font-size:10px; text-transform:uppercase; letter-spacing:0.5px; color:#5a5870; margin-bottom:6px; }
.stat-value{ font-family:'DM Mono',monospace; font-size:18px; font-weight:500; }
.section-hd   { display:flex; align-items:center; gap:9px; border-bottom:0.5px solid rgba(255,255,255,0.08); padding-bottom:10px; margin-bottom:12px; margin-top:24px; }
.dot          { width:9px; height:9px; border-radius:50%; display:inline-block; }
.section-label{ font-size:11px; font-weight:500; letter-spacing:0.7px; text-transform:uppercase; color:#9593a0; }
.flight-card  { background:#13131a; border:0.5px solid rgba(255,255,255,0.08); border-radius:13px; padding:16px 18px; margin-bottom:10px; display:flex; align-items:center; gap:14px; }
.flight-card.top1{ border-color:rgba(232,200,74,0.25); background:rgba(232,200,74,0.03); }
.rank-badge   { font-family:'DM Mono',monospace; font-size:11px; color:#5a5870; min-width:22px; }
.rank-badge.gold{ color:#e8c84a; }
.card-body    { flex:1; }
.card-airline { font-size:15px; font-weight:600; color:#f0eeea; }
.card-time    { font-family:'DM Mono',monospace; font-size:12px; color:#9593a0; margin-left:8px; }
.chip         { display:inline-block; font-size:11px; padding:3px 9px; border-radius:20px; border:0.5px solid rgba(255,255,255,0.1); color:#9593a0; background:#1a1a24; margin-right:5px; margin-top:5px; }
.price-block  { text-align:right; }
.price-main   { font-family:'DM Mono',monospace; font-size:16px; font-weight:500; color:#f0eeea; }
.price-label  { font-size:10px; color:#5a5870; text-transform:uppercase; letter-spacing:0.4px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["index"] if c in df.columns])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df


# ============================================================
# TRAIN MODEL — otomatis saat app load, di-cache selamanya
# ============================================================

@st.cache_resource(show_spinner="Melatih model Random Forest...")
def train_model(data_path: str):
    df = pd.read_csv(data_path)
    df = df.drop(columns=[c for c in ["index", "flight"] if c in df.columns])
    df["class"]          = df["class"].map(CLASS_MAPPING).fillna(0).astype(int)
    df["stops"]          = df["stops"].map(STOPS_MAPPING).fillna(0).astype(int)
    df["departure_time"] = df["departure_time"].map(TIME_MAPPING).fillna(0).astype(int)
    df["arrival_time"]   = df["arrival_time"].map(TIME_MAPPING).fillna(0).astype(int)
    X = pd.get_dummies(df.drop("price", axis=1),
                       columns=["airline", "source_city", "destination_city"])
    y = df["price"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    return rf


df    = load_data()
model = train_model(DATA_PATH)


# ============================================================
# HEADER
# ============================================================

col_title, col_badge = st.columns([5, 2])
with col_title:
    st.markdown('<div class="app-title">✈️ AI Flight Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="app-subtitle">{len(df):,} penerbangan · '
        f'{df["airline"].nunique()} maskapai · {df["source_city"].nunique()} kota</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    st.markdown(
        '<br><span style="display:inline-flex;align-items:center;gap:6px;'
        'font-family:\'DM Mono\',monospace;font-size:11px;padding:4px 12px;border-radius:20px;'
        'background:rgba(29,158,117,0.12);color:#1D9E75;border:0.5px solid rgba(29,158,117,0.25);">'
        '● Random Forest aktif</span>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ============================================================
# FORM
# ============================================================

col1, col2, col3 = st.columns([2, 2, 3])
with col1:
    src = st.selectbox("🛫 Kota Asal", sorted(CITIES), index=sorted(CITIES).index("Delhi"))
with col2:
    dst_opts = [c for c in sorted(CITIES) if c != src]
    dst = st.selectbox("🛬 Kota Tujuan", dst_opts,
                       index=dst_opts.index("Mumbai") if "Mumbai" in dst_opts else 0)
with col3:
    days_left = st.slider("📅 Sisa Hari Sebelum Keberangkatan", 1, 49, 10, 1)

st.markdown("<br>", unsafe_allow_html=True)
search = st.button("🔍 Cari Rekomendasi Terbaik", width="stretch", type="primary")


# ============================================================
# RESULTS
# ============================================================

if search:
    if src == dst:
        st.error("⚠️ Kota asal dan tujuan tidak boleh sama.")
        st.stop()

    with st.spinner("Menganalisis penerbangan..."):
        eco, biz = find_best_flights(
            df, model,
            {"source_city": src, "destination_city": dst, "days_left": days_left},
            TOP_N,
        )

    if eco.empty and biz.empty:
        st.warning(f"Tidak ada data untuk rute **{src} → {dst}**.")
        st.stop()

    # Transit insight
    def _transit_insight(results):
        if results.empty or results.iloc[0]["stops"] == "zero":
            return ""
        top_row   = results.iloc[0]
        stops_lbl = STOPS_LABEL.get(top_row["stops"], top_row["stops"])
        nonstop   = results[results["stops"] == "zero"]
        if nonstop.empty:
            return (f'<span style="font-size:11px;color:#e8c84a;">'
                    f'⚡ Tidak ada non-stop — {stops_lbl} terbaik</span>')
        selisih = nonstop["price"].min() - top_row["price"]
        pct     = selisih / nonstop["price"].min() * 100
        return (f'<span style="font-size:11px;color:#e8c84a;">'
                f'⚡ {stops_lbl} lebih murah {format_inr(selisih)} ({pct:.0f}%) vs non-stop</span>')

    best_eco       = format_inr(eco.iloc[0]["price"]) if not eco.empty else "—"
    best_biz       = format_inr(biz.iloc[0]["price"]) if not biz.empty else "—"
    best_eco_stops = STOPS_LABEL.get(eco.iloc[0]["stops"], "") if not eco.empty else ""
    best_biz_stops = STOPS_LABEL.get(biz.iloc[0]["stops"], "") if not biz.empty else ""
    insight_eco    = _transit_insight(eco)
    insight_biz    = _transit_insight(biz)

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-label">Rute</div>
        <div class="stat-value" style="color:#f0eeea;font-size:15px;">{src} → {dst}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Harga Eco Terbaik</div>
        <div class="stat-value" style="color:#1D9E75;">{best_eco}</div>
        <div style="font-size:10px;color:#5a5870;margin-top:3px;">{best_eco_stops}</div>
        {insight_eco}
      </div>
      <div class="stat-card">
        <div class="stat-label">Harga Bisnis Terbaik</div>
        <div class="stat-value" style="color:#7F77DD;">{best_biz}</div>
        <div style="font-size:10px;color:#5a5870;margin-top:3px;">{best_biz_stops}</div>
        {insight_biz}
      </div>
      <div class="stat-card">
        <div class="stat-label">Sisa Hari</div>
        <div class="stat-value" style="color:#e8c84a;">{days_left} hari</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    def render_section(results, kelas):
        if results.empty:
            st.info(f"Tidak ada hasil kelas {kelas}.")
            return
        dot_color = "#1D9E75" if kelas == "Economy" else "#7F77DD"
        label     = "Ekonomi" if kelas == "Economy" else "Bisnis"
        st.markdown(f"""
        <div class="section-hd">
          <span class="dot" style="background:{dot_color};box-shadow:0 0 8px {dot_color}88;"></span>
          <span class="section-label">{label} — Top {TOP_N}</span>
        </div>""", unsafe_allow_html=True)

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            top1_cls   = "top1" if rank == 1 else ""
            rank_cls   = "gold" if rank == 1 else ""
            sc         = stops_color(row["stops"])
            sl         = STOPS_LABEL.get(row["stops"], row["stops"])
            dep        = TIME_LABEL.get(row.get("departure_time", ""), "")
            arr        = TIME_LABEL.get(row.get("arrival_time", ""), "")
            time_str   = f"{dep} → {arr}" if arr else dep
            dur_str    = format_duration(row["duration"])
            price_str  = format_inr(row["price"])
            airline_nm = row["airline"].replace("_", " ")
            pred_html  = ""
            if "predicted_price" in row.index:
                pred_html = (f'<div class="price-label" style="margin-top:4px;">'
                             f'prediksi: {format_inr(row["predicted_price"])}</div>')
            st.markdown(f"""
            <div class="flight-card {top1_cls}">
              <div class="rank-badge {rank_cls}">{str(rank).zfill(2)}</div>
              <div class="card-body">
                <div>
                  <span class="card-airline">{airline_nm}</span>
                  <span class="card-time">{time_str}</span>
                </div>
                <div style="margin-top:6px;">
                  <span class="chip" style="color:{sc};background:rgba(0,0,0,0.2);border-color:{sc}44;">{sl}</span>
                  <span class="chip">⏱ {dur_str}</span>
                  <span class="chip">✈️ {row['flight']}</span>
                </div>
              </div>
              <div class="price-block">
                <div class="price-main">{price_str}</div>
                <div class="price-label">harga aktual</div>
                {pred_html}
              </div>
            </div>""", unsafe_allow_html=True)

    col_eco, col_biz = st.columns(2, gap="large")
    with col_eco:
        render_section(eco, "Economy")
    with col_biz:
        render_section(biz, "Business")