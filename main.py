# ============================================================
# app.py — AI Flight Optimizer · Streamlit UI
# ============================================================
# Jalankan dengan:
#   streamlit run app.py
# ============================================================

import os
import streamlit as st  # type: ignore
import pandas as pd # type: ignore
import joblib   # type: ignore

from config import MODEL_PATH, DATA_PATH, TOP_N, CITIES, STOPS_LABEL, TIME_LABEL
from utils  import format_duration, format_inr, stops_color, class_color
from engine import find_best_flights


# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

/* Sembunyikan header bawaan Streamlit */
#MainMenu, header, footer { visibility: hidden; }

/* Background */
.stApp { background-color: #0c0c0f; }

/* Judul utama */
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 700;
    color: #f0eeea;
    letter-spacing: -0.5px;
    margin-bottom: 2px;
}
.app-subtitle {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #5a5870;
    margin-bottom: 1.5rem;
}

/* Panel form */
.form-panel {
    background: #13131a;
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

/* Kartu hasil */
.flight-card {
    background: #13131a;
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 13px;
    padding: 16px 18px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.2s;
}
.flight-card:hover { border-color: rgba(255,255,255,0.14); }
.flight-card.top1 {
    border-color: rgba(232,200,74,0.25);
    background: rgba(232,200,74,0.03);
}

.rank-badge {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #5a5870;
    min-width: 22px;
}
.rank-badge.gold { color: #e8c84a; }

.card-body { flex: 1; }
.card-airline { font-size: 15px; font-weight: 600; color: #f0eeea; }
.card-time    { font-family: 'DM Mono', monospace; font-size: 12px; color: #9593a0; margin-left: 8px; }

.chip {
    display: inline-block;
    font-size: 11px;
    padding: 3px 9px;
    border-radius: 20px;
    border: 0.5px solid rgba(255,255,255,0.1);
    color: #9593a0;
    background: #1a1a24;
    margin-right: 5px;
    margin-top: 5px;
}

.price-block { text-align: right; }
.price-main  { font-family: 'DM Mono', monospace; font-size: 16px; font-weight: 500; color: #f0eeea; }
.price-label { font-size: 10px; color: #5a5870; text-transform: uppercase; letter-spacing: 0.4px; }

/* Section header */
.section-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    border-bottom: 0.5px solid rgba(255,255,255,0.08);
    padding-bottom: 10px;
    margin-bottom: 12px;
    margin-top: 24px;
}
.dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
}
.section-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    color: #9593a0;
}

/* Stat cards */
.stat-row { display: flex; gap: 10px; margin-bottom: 1.5rem; }
.stat-card {
    flex: 1;
    background: #13131a;
    border: 0.5px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
}
.stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; color: #5a5870; margin-bottom: 6px; }
.stat-value { font-family: 'DM Mono', monospace; font-size: 18px; font-weight: 500; }

/* Model status badge */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 20px;
}
.model-ok   { background: rgba(29,158,117,0.12); color: #1D9E75; border: 0.5px solid rgba(29,158,117,0.25); }
.model-miss { background: rgba(232,200,74,0.10); color: #e8c84a; border: 0.5px solid rgba(232,200,74,0.25); }
.model-err  { background: rgba(226,75,74,0.10);  color: #e24b4a; border: 0.5px solid rgba(226,75,74,0.25); }
</style>
""", unsafe_allow_html=True)


# ── Load Data ────────────────────────────────────────────────
@st.cache_data(show_spinner="Memuat dataset...")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[c for c in ["index"] if c in df.columns])
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].astype(str)
    return df


# ── Load Model ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model ML...")
def load_model():
    """
    Coba load model dari MODEL_PATH.
    Return (model, status, pesan):
      status = 'ok' | 'missing' | 'error'
    """
    if not os.path.exists(MODEL_PATH):
        return None, "missing", f"File model tidak ditemukan: `{MODEL_PATH}`"
    try:
        mdl = joblib.load(MODEL_PATH)
        return mdl, "ok", f"Model dimuat dari `{MODEL_PATH}`"
    except Exception as e:
        return None, "error", f"Gagal load model: {e}"


# ── Inisialisasi ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Flight Optimizer",
    page_icon="✈️",
    layout="wide",
)

df    = load_data()
model, model_status, model_msg = load_model()


# ── Header ───────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 2])
with col_title:
    st.markdown('<div class="app-title">✈️ AI Flight Optimizer</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="app-subtitle">{len(df):,} penerbangan · '
        f'{df["airline"].nunique()} maskapai · {df["source_city"].nunique()} kota</div>',
        unsafe_allow_html=True,
    )
with col_badge:
    badge_cls = {"ok": "model-ok", "missing": "model-miss", "error": "model-err"}[model_status]
    badge_ico = {"ok": "●", "missing": "◌", "error": "✕"}[model_status]
    badge_txt = {"ok": "Model aktif", "missing": "Fallback mode", "error": "Model error"}[model_status]
    st.markdown(
        f'<br><span class="model-badge {badge_cls}">{badge_ico} {badge_txt}</span>',
        unsafe_allow_html=True,
    )
    if model_status != "ok":
        st.caption(model_msg)

st.markdown("---")

# ── Form Input ───────────────────────────────────────────────
col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    src = st.selectbox("🛫 Kota Asal", sorted(CITIES), index=sorted(CITIES).index("Delhi"))

with col2:
    dst_options = [c for c in sorted(CITIES) if c != src]
    dst = st.selectbox("🛬 Kota Tujuan", dst_options, index=dst_options.index("Mumbai") if "Mumbai" in dst_options else 0)

with col3:
    days_left = st.slider(
        "📅 Sisa Hari Sebelum Keberangkatan",
        min_value=1,
        max_value=49,
        value=10,
        step=1,
        help="Berdasarkan rentang data: 1–49 hari",
    )

# ── Search Button ────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
search = st.button("🔍 Cari Rekomendasi Terbaik", use_container_width=True, type="primary")


# ── Results ──────────────────────────────────────────────────
if search:
    if src == dst:
        st.error("⚠️ Kota asal dan tujuan tidak boleh sama.")
        st.stop()

    input_data = {
        "source_city":      src,
        "destination_city": dst,
        "days_left":        days_left,
    }

    with st.spinner("Menganalisis penerbangan..."):
        eco, biz = find_best_flights(df, model, input_data, TOP_N)

    if eco.empty and biz.empty:
        st.warning(f"Tidak ada data penerbangan untuk rute **{src} → {dst}**.")
        st.stop()

    # ── Stat Cards ───────────────────────────────────────────
    best_eco_price = format_inr(eco.iloc[0]["price"]) if not eco.empty else "—"
    best_biz_price = format_inr(biz.iloc[0]["price"]) if not biz.empty else "—"

    st.markdown(f"""
    <div class="stat-row">
      <div class="stat-card">
        <div class="stat-label">Rute</div>
        <div class="stat-value" style="color:#f0eeea;font-size:15px;">{src} → {dst}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Harga Eco Terbaik</div>
        <div class="stat-value" style="color:#1D9E75;">{best_eco_price}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Harga Bisnis Terbaik</div>
        <div class="stat-value" style="color:#7F77DD;">{best_biz_price}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Sisa Hari</div>
        <div class="stat-value" style="color:#e8c84a;">{days_left} hari</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Helper render kartu ───────────────────────────────────
    def render_section(results: pd.DataFrame, kelas: str):
        if results.empty:
            st.info(f"Tidak ada hasil kelas {kelas} untuk rute ini.")
            return

        dot_color = "#1D9E75" if kelas == "Economy" else "#7F77DD"
        label     = "Ekonomi" if kelas == "Economy" else "Bisnis"

        st.markdown(f"""
        <div class="section-hd">
          <span class="dot" style="background:{dot_color};
            box-shadow:0 0 8px {dot_color}88;"></span>
          <span class="section-label">{label} — Top {TOP_N}</span>
        </div>
        """, unsafe_allow_html=True)

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            top1_cls   = "top1" if rank == 1 else ""
            rank_cls   = "gold" if rank == 1 else ""
            stop_color = stops_color(row["stops"])
            stop_label = STOPS_LABEL.get(row["stops"], row["stops"])
            dep_label  = TIME_LABEL.get(row.get("departure_time", ""), "")
            arr_label  = TIME_LABEL.get(row.get("arrival_time", ""),  "")
            dur_str    = format_duration(row["duration"])
            price_str  = format_inr(row["price"])

            # Kolom harga prediksi (jika model aktif)
            pred_str = ""
            if model_status == "ok" and "predicted_price" in row.index:
                pred_str = (
                    f'<div class="price-label" style="margin-top:4px;">'
                    f'prediksi: {format_inr(row["predicted_price"])}</div>'
                )

            airline_display = row["airline"].replace("_", " ")

            st.markdown(f"""
            <div class="flight-card {top1_cls}">
              <div class="rank-badge {rank_cls}">{str(rank).zfill(2)}</div>
              <div class="card-body">
                <div>
                  <span class="card-airline">{airline_display}</span>
                  <span class="card-time">{dep_label}{' → ' + arr_label if arr_label else ''}</span>
                </div>
                <div style="margin-top:6px;">
                  <span class="chip" style="color:{stop_color};
                    background:rgba(0,0,0,0.2);
                    border-color:{stop_color}44;">
                    {stop_label}
                  </span>
                  <span class="chip">⏱ {dur_str}</span>
                  <span class="chip">✈️ {row['flight']}</span>
                </div>
              </div>
              <div class="price-block">
                <div class="price-main">{price_str}</div>
                <div class="price-label">harga aktual</div>
                {pred_str}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # Tampilkan Economy & Business side by side
    col_eco, col_biz = st.columns(2, gap="large")
    with col_eco:
        render_section(eco, "Economy")
    with col_biz:
        render_section(biz, "Business")

    # ── Raw data expander ─────────────────────────────────────
    with st.expander("📊 Lihat Data Mentah"):
        combined = pd.concat([eco, biz]).reset_index(drop=True)
        show_cols = ["airline", "flight", "class", "departure_time",
                     "arrival_time", "stops", "duration", "price"]
        if "predicted_price" in combined.columns:
            show_cols.append("predicted_price")
        st.dataframe(
            combined[[c for c in show_cols if c in combined.columns]],
            use_container_width=True,
        )