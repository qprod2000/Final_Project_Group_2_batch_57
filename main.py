# ============================================================
# main.py — AI Flight Optimizer (single-file, Streamlit Cloud ready)
# Jalankan: streamlit run main.py
# ============================================================

import os
import io
import time
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import joblib   # type: ignore
import streamlit as st  # type: ignore
from sklearn.model_selection import train_test_split    # type: ignore
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   # type: ignore
from xgboost import XGBRegressor    # type: ignore


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "model/best_model.pkl"
DATA_PATH  = "airlines_flights_data.csv"
TOP_N      = 3

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
    drop_cols = [c for c in ["index", "flight", "price"] if c in df.columns]
    df = df.drop(columns=drop_cols)
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
    df = df.drop(columns=extra, errors="ignore")
    return df


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
            df_route["predicted_price"] = df_route["price"]
            sort_col = "price"
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
.model-badge  { display:inline-flex; align-items:center; gap:6px; font-family:'DM Mono',monospace; font-size:11px; padding:4px 12px; border-radius:20px; }
.model-ok     { background:rgba(29,158,117,0.12); color:#1D9E75; border:0.5px solid rgba(29,158,117,0.25); }
.model-miss   { background:rgba(232,200,74,0.10); color:#e8c84a; border:0.5px solid rgba(232,200,74,0.25); }
.model-err    { background:rgba(226,75,74,0.10);  color:#e24b4a; border:0.5px solid rgba(226,75,74,0.25); }
.metric-box   { background:#13131a; border:0.5px solid rgba(255,255,255,0.08); border-radius:12px; padding:16px 18px; margin-bottom:10px; }
.metric-name  { font-size:11px; text-transform:uppercase; letter-spacing:0.5px; color:#5a5870; margin-bottom:4px; }
.metric-val   { font-family:'DM Mono',monospace; font-size:22px; font-weight:500; }
.train-log    { font-family:'DM Mono',monospace; font-size:12px; color:#5a5870; background:#0c0c0f; border:0.5px solid rgba(255,255,255,0.06); border-radius:8px; padding:12px 16px; margin-top:8px; line-height:1.8; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Memuat dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["index"] if c in df.columns])
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)
    return df


@st.cache_resource(show_spinner="Memuat model ML...")
def load_model():
    # Cari model: utama → legacy RF → legacy XGB
    candidates = [
        "model/best_model.pkl",
        "model/random_forest.pkl",
        "model/xgboost.pkl",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                mdl = joblib.load(path)
                return mdl, "ok", f"Model dimuat dari `{path}`"
            except Exception as e:
                return None, "error", f"Gagal load model: {e}"
    return None, "missing", "Belum ada model — pergi ke Train Model"


df                             = load_data()
model, model_status, model_msg = load_model()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-size:18px;font-weight:700;'
        'color:#f0eeea;margin-bottom:4px;">✈️ Flight Optimizer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-family:\'DM Mono\',monospace;font-size:11px;'
        'color:#5a5870;margin-bottom:20px;">AI-powered · v1.0</div>',
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigasi",
        ["🔍 Cari Penerbangan", "🤖 Train Model"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:10px;color:#5a5870;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:8px;">Status Model</div>',
        unsafe_allow_html=True,
    )
    badge_cls = {"ok": "model-ok", "missing": "model-miss", "error": "model-err"}[model_status]
    badge_ico = {"ok": "●", "missing": "◌", "error": "✕"}[model_status]
    badge_txt = {"ok": "Model aktif", "missing": "Belum ada model", "error": "Model error"}[model_status]
    st.markdown(
        f'<span class="model-badge {badge_cls}">{badge_ico} {badge_txt}</span>',
        unsafe_allow_html=True,
    )
    if model_status == "ok":
        st.caption(f"📁 `{MODEL_PATH}`")
    else:
        st.caption("Pergi ke **Train Model** untuk membuat model baru.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"📊 {len(df):,} penerbangan")
    st.caption(f"🏙️ {df['source_city'].nunique()} kota · {df['airline'].nunique()} maskapai")


# ============================================================
# PAGE 1 — Cari Penerbangan
# ============================================================

if page == "🔍 Cari Penerbangan":

    st.markdown('<div class="app-title">Cari Penerbangan Terbaik</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Rekomendasi berdasarkan harga, durasi &amp; jumlah transit</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

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
    search = st.button("🔍 Cari Rekomendasi Terbaik", use_container_width=True, type="primary")

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

        def _transit_insight(results):
            if results.empty or results.iloc[0]["stops"] == "zero":
                return ""
            top_row   = results.iloc[0]
            stops_lbl = STOPS_LABEL.get(top_row["stops"], top_row["stops"])
            top_price = top_row["price"]
            nonstop   = results[results["stops"] == "zero"]
            if nonstop.empty:
                return (f'<span style="font-size:11px;color:#e8c84a;">'
                        f'⚡ Tidak ada non-stop — {stops_lbl} terbaik</span>')
            selisih = nonstop["price"].min() - top_price
            pct     = selisih / nonstop["price"].min() * 100
            return (f'<span style="font-size:11px;color:#e8c84a;">'
                    f'⚡ {stops_lbl} lebih murah {format_inr(selisih)} ({pct:.0f}%) vs non-stop</span>')

        insight_eco    = _transit_insight(eco)
        insight_biz    = _transit_insight(biz)
        best_eco       = format_inr(eco.iloc[0]["price"]) if not eco.empty else "—"
        best_biz       = format_inr(biz.iloc[0]["price"]) if not biz.empty else "—"
        best_eco_stops = STOPS_LABEL.get(eco.iloc[0]["stops"], "") if not eco.empty else ""
        best_biz_stops = STOPS_LABEL.get(biz.iloc[0]["stops"], "") if not biz.empty else ""

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
                if model_status == "ok" and "predicted_price" in row.index:
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

        with st.expander("📊 Lihat Data Mentah"):
            combined  = pd.concat([eco, biz]).reset_index(drop=True)
            show_cols = ["airline", "flight", "class", "departure_time",
                         "arrival_time", "stops", "duration", "price"]
            if "predicted_price" in combined.columns:
                show_cols.append("predicted_price")
            st.dataframe(combined[[c for c in show_cols if c in combined.columns]],
                         use_container_width=True)


# ============================================================
# PAGE 2 — Train Model
# ============================================================

elif page == "🤖 Train Model":

    st.markdown('<div class="app-title">Auto Train &amp; Select Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">'
        'Random Forest &amp; XGBoost dilatih otomatis — model terbaik dipilih &amp; disimpan'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Info box ─────────────────────────────────────────────
    st.markdown("""
    <div style="background:#13131a;border:0.5px solid rgba(255,255,255,0.08);
                border-radius:12px;padding:16px 20px;margin-bottom:20px;">
      <div style="font-size:13px;color:#9593a0;line-height:1.8;">
        Klik tombol di bawah — sistem akan otomatis:
        <span style="color:#f0eeea;">① Preprocessing data</span> →
        <span style="color:#1D9E75;">② Latih Random Forest</span> →
        <span style="color:#7F77DD;">③ Latih XGBoost</span> →
        <span style="color:#e8c84a;">④ Bandingkan R² Score</span> →
        <span style="color:#f0eeea;">⑤ Simpan model terbaik sebagai <code>model/best_model.pkl</code></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Parameter (collapsed di expander) ────────────────────
    with st.expander("⚙️ Parameter Training (opsional)", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            test_size    = st.slider("Test Set (%)", 10, 30, 20, 5)
        with col_b:
            n_estimators = st.slider("n_estimators", 50, 300, 100, 50)
        with col_c:
            random_state = st.number_input("Random State", value=42, step=1)

    st.markdown("<br>", unsafe_allow_html=True)
    train_btn = st.button(
        "🚀 Mulai Auto Training",
        use_container_width=True,
        type="primary",
    )

    if train_btn:
        log_lines    = []
        progress_bar = st.progress(0, text="Menyiapkan data...")
        log_box      = st.empty()

        def log(msg):
            log_lines.append(msg)
            log_box.markdown(
                '<div class="train-log">' + "<br>".join(log_lines) + "</div>",
                unsafe_allow_html=True,
            )

        # ── 1. Preprocessing ─────────────────────────────────
        log("⟳ Preprocessing data...")
        train_df = df.copy()
        train_df = train_df.drop(columns=[c for c in ["index", "flight"] if c in train_df.columns])
        train_df["class"]          = train_df["class"].map(CLASS_MAPPING).fillna(0).astype(int)
        train_df["stops"]          = train_df["stops"].map(STOPS_MAPPING).fillna(0).astype(int)
        train_df["departure_time"] = train_df["departure_time"].map(TIME_MAPPING).fillna(0).astype(int)
        train_df["arrival_time"]   = train_df["arrival_time"].map(TIME_MAPPING).fillna(0).astype(int)
        X = pd.get_dummies(
            train_df.drop("price", axis=1),
            columns=["airline", "source_city", "destination_city"],
        )
        y = train_df["price"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=int(random_state)
        )
        progress_bar.progress(10, text="Data siap — mulai training...")
        log(f"✓ {len(train_df):,} baris · {X.shape[1]} fitur · "
            f"train={len(X_train):,} test={len(X_test):,}")

        results   = {}   # nama → {model, r2, r2_train, mae, rmse, elapsed}

        # ── 2. Random Forest ─────────────────────────────────
        log("⟳ [1/2] Melatih Random Forest...")
        progress_bar.progress(20, text="Training Random Forest...")
        t0 = time.time()
        rf = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        t_rf = time.time() - t0

        y_pred_rf  = rf.predict(X_test)
        results["Random Forest"] = {
            "model":    rf,
            "r2":       r2_score(y_test, y_pred_rf),
            "r2_train": rf.score(X_train, y_train),
            "mae":      mean_absolute_error(y_test, y_pred_rf),
            "rmse":     np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "elapsed":  t_rf,
        }
        rf_r = results["Random Forest"]
        progress_bar.progress(55, text="Training XGBoost...")
        log(f"✓ Random Forest selesai ({t_rf:.1f}s) · "
            f"R²={rf_r['r2']:.4f} · MAE=₹{rf_r['mae']:,.0f}")

        # ── 3. XGBoost ───────────────────────────────────────
        log("⟳ [2/2] Melatih XGBoost...")
        t0 = time.time()
        xgb = XGBRegressor(
            n_estimators=int(n_estimators),
            learning_rate=0.1,
            max_depth=6,
            random_state=int(random_state),
            n_jobs=-1,
            verbosity=0,
        )
        xgb.fit(X_train, y_train)
        t_xgb = time.time() - t0

        y_pred_xgb = xgb.predict(X_test)
        results["XGBoost"] = {
            "model":    xgb,
            "r2":       r2_score(y_test, y_pred_xgb),
            "r2_train": xgb.score(X_train, y_train),
            "mae":      mean_absolute_error(y_test, y_pred_xgb),
            "rmse":     np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "elapsed":  t_xgb,
        }
        xgb_r = results["XGBoost"]
        progress_bar.progress(85, text="Membandingkan & menyimpan...")
        log(f"✓ XGBoost selesai ({t_xgb:.1f}s) · "
            f"R²={xgb_r['r2']:.4f} · MAE=₹{xgb_r['mae']:,.0f}")

        # ── 4. Pilih model terbaik ────────────────────────────
        best_name  = max(results, key=lambda k: results[k]["r2"])
        other_name = [k for k in results if k != best_name][0]
        best       = results[best_name]
        other      = results[other_name]
        margin     = best["r2"] - other["r2"]

        log(f"★ Model terbaik: {best_name} "
            f"(R² lebih tinggi {margin:.4f} dari {other_name})")

        # ── 5. Simpan ─────────────────────────────────────────
        os.makedirs("model", exist_ok=True)
        fpath = "model/best_model.pkl"
        joblib.dump(best["model"], fpath)

        buf = io.BytesIO()
        joblib.dump(best["model"], buf)
        buf.seek(0)
        model_bytes = buf.read()

        progress_bar.progress(100, text="Selesai!")
        log(f"✓ Disimpan → {fpath} ({len(model_bytes)/1024:.1f} KB)")

        # ── Tabel perbandingan ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Perbandingan Model")

        winner_ico = {"Random Forest": "🌲", "XGBoost": "⚡"}
        for name, res in results.items():
            is_best   = name == best_name
            border    = "rgba(232,200,74,0.35)" if is_best else "rgba(255,255,255,0.08)"
            bg        = "rgba(232,200,74,0.04)" if is_best else "#13131a"
            crown     = f'<span style="color:#e8c84a;margin-left:8px;">★ Terpilih</span>' if is_best else ""
            ico       = winner_ico.get(name, "🤖")
            name_color = "#1D9E75" if name == "Random Forest" else "#7F77DD"

            st.markdown(f"""
            <div style="background:{bg};border:0.5px solid {border};
                        border-radius:13px;padding:18px 20px;margin-bottom:10px;">
              <div style="display:flex;align-items:center;justify-content:space-between;
                          margin-bottom:14px;">
                <div style="font-size:15px;font-weight:600;color:{name_color};">
                  {ico} {name}{crown}
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:11px;color:#5a5870;">
                  {res['elapsed']:.1f}s
                </div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;">
                <div class="metric-box" style="margin:0;">
                  <div class="metric-name">R² Test</div>
                  <div class="metric-val" style="color:{'#1D9E75' if is_best else '#f0eeea'};
                       font-size:18px;">{res['r2']:.4f}</div>
                </div>
                <div class="metric-box" style="margin:0;">
                  <div class="metric-name">R² Train</div>
                  <div class="metric-val" style="color:#9593a0;font-size:18px;">
                    {res['r2_train']:.4f}</div>
                </div>
                <div class="metric-box" style="margin:0;">
                  <div class="metric-name">MAE</div>
                  <div class="metric-val" style="color:#f0eeea;font-size:18px;">
                    ₹{res['mae']:,.0f}</div>
                </div>
                <div class="metric-box" style="margin:0;">
                  <div class="metric-name">RMSE</div>
                  <div class="metric-val" style="color:#f0eeea;font-size:18px;">
                    ₹{res['rmse']:,.0f}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Feature Importance ────────────────────────────────
        if hasattr(best["model"], "feature_importances_"):
            fi = pd.DataFrame({
                "Fitur":      X_train.columns,
                "Importance": best["model"].feature_importances_,
            }).sort_values("Importance", ascending=False).head(10)
            with st.expander(f"🔍 Top 10 Feature Importance ({best_name})"):
                st.dataframe(fi.reset_index(drop=True), use_container_width=True)

        # ── Download ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.success(
            f"✅ **{best_name}** terpilih sebagai model terbaik "
            f"(R²={best['r2']:.4f}) dan disimpan ke `{fpath}`"
        )

        st.download_button(
            label=f"⬇️ Download best_model.pkl  ({best_name})",
            data=model_bytes,
            file_name="best_model.pkl",
            mime="application/octet-stream",
            use_container_width=True,
        )

        st.info(
            "Setelah download, letakkan `best_model.pkl` ke folder `model/` "
            "lalu refresh halaman — model akan otomatis aktif.",
            icon="💡",
        )

        st.cache_resource.clear()