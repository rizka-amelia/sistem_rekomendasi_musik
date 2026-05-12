"""
Sistem Rekomendasi Musik - Content-Based Filtering
Jalankan: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os
import matplotlib.pyplot as plt

# ── konfigurasi Halaman ────────────────────────────────────
st.set_page_config(
    page_title="Sistem Rekomendasi Musik",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #0d0d0d 100%);
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1); border-radius: 16px;
        padding: 20px; box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #1DB954, #1ed760, #1DB954);
        background-size: 200% auto;
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradient-shift 3s ease-in-out infinite;
        text-align: center; margin-bottom: 0.5rem; letter-spacing: -1px;
    }
    @keyframes gradient-shift {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    .sub-title {
        text-align: center; color: #888; margin-bottom: 2rem;
        font-size: 1rem; font-weight: 400; letter-spacing: 0.5px;
    }
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 2px solid rgba(255,255,255,0.1) !important;
        border-radius: 12px !important; padding: 12px 16px !important;
        font-size: 1rem !important; color: white !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #1DB954 !important;
        box-shadow: 0 0 20px rgba(29,185,84,0.3) !important;
        background: rgba(29,185,84,0.1) !important;
    }
    .stTextInput > div > div > input::placeholder { color: rgba(255,255,255,0.4) !important; }
    .stButton > button, [data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%) !important;
        color: #000 !important; border: none !important;
        border-radius: 12px !important; padding: 8px 18px !important;
        font-weight: 700 !important; font-size: 0.82rem !important;
        letter-spacing: 0.4px;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(29,185,84,0.4) !important;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(29,185,84,0.5) !important;
    }
    .rec-card {
        background: rgba(255,255,255,0.03); backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 18px 22px; margin: 10px 0;
        transition: all 0.3s ease;
    }
    .rec-card:hover {
        background: rgba(255,255,255,0.06);
        border-color: rgba(29,185,84,0.3); transform: translateX(5px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    .input-song-card {
        background: linear-gradient(135deg, rgba(29,185,84,0.15) 0%, rgba(30,215,96,0.1) 100%);
        backdrop-filter: blur(20px); border: 1px solid rgba(29,185,84,0.3);
        border-radius: 20px; padding: 28px 32px; margin-bottom: 25px;
        color: white; box-shadow: 0 10px 40px rgba(29,185,84,0.2);
    }
    .genre-tag {
        background: linear-gradient(135deg, rgba(29,185,84,0.2) 0%, rgba(29,185,84,0.1) 100%);
        border: 1px solid rgba(29,185,84,0.3); border-radius: 20px;
        padding: 4px 14px; font-size: 0.8rem; color: #1ed760;
        margin: 3px; display: inline-block; font-weight: 500;
    }
    .section-header {
        color: #1ed760; font-size: 1.4rem; font-weight: 700;
        border-bottom: 2px solid rgba(29,185,84,0.5);
        padding-bottom: 10px; margin: 30px 0 15px 0; letter-spacing: 0.5px;
    }
    .progress-track {
        height: 6px; background: rgba(255,255,255,0.1);
        border-radius: 3px; overflow: hidden; margin: 6px 0 10px 0;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #1DB954, #1ed760, #1DB954);
        background-size: 200% auto; border-radius: 3px;
        animation: shimmer 2s linear infinite;
    }
    @keyframes shimmer {
        0%   { background-position: 100% center; }
        100% { background-position: -100% center; }
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px; padding: 20px; transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        background: rgba(255,255,255,0.05);
        border-color: rgba(29,185,84,0.3); transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] { color: #888 !important; font-size: 0.85rem !important; }
    [data-testid="stMetricValue"] { color: #1ed760 !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    .stTabs [data-testid="stTabBar"] { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 6px; }
    .stTabs [role="tabbutton"] { border-radius: 8px !important; padding: 10px 20px !important; font-weight: 600 !important; }
    .stTabs [role="tabbutton"][aria-selected="true"] {
        background: linear-gradient(135deg, #1DB954 0%, #1ed760 100%) !important; color: #000 !important;
    }
    [data-testid="stSidebar"] { background: rgba(0,0,0,0.3) !important; }
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 4px; }
    ::-webkit-scrollbar-thumb { background: rgba(29,185,84,0.5); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(29,185,84,0.8); }
</style>
""", unsafe_allow_html=True)


# ── load Model ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = 'model_rekomendasi.pkl'
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)


model_data = load_model()


# ── fungsi pembantu ────────────────────────────────────────
def get_genres_set(genre_str):
    genres = re.split(r'[,\s]+', str(genre_str).lower().strip())
    return {g for g in genres if len(g) > 2}


def search_song_fuzzy(query, song_index):
    """Cari lagu dengan fuzzy/partial match (case-insensitive)."""
    query_lower = query.lower().strip()
    if not query_lower:
        return []
    return [s for s in song_index.index if query_lower in s.lower()]


def rekomendasikan_lagu_app(judul_lagu, k=10):
    """
    Rekomendasikan lagu berdasarkan input.
    Returns: (result_df, input_info, matches_list)
      - Jika tepat 1 cocok  → (DataFrame rekomendasi, info lagu, [])
      - Jika >1 cocok       → (DataFrame kosong, None, [daftar semua cocok])
      - Jika tidak ada cocok→ (DataFrame kosong, None, [])
    """
    if model_data is None:
        return pd.DataFrame(), None, []

    df         = model_data['df']
    cosine_sim = model_data['cosine_sim_matrix']
    song_index = model_data['song_index']

    track_col  = 'Track Name Original'  if 'Track Name Original'  in df.columns else 'Track Name'
    artist_col = 'Artist Name Original' if 'Artist Name Original' in df.columns else 'Artist Name(s)'

    matches = search_song_fuzzy(judul_lagu, song_index)

    if not matches:
        return pd.DataFrame(), None, []

    # lebih dari 1 cocok → kembalikan daftar untuk dipilih user
    if len(matches) > 1:
        return pd.DataFrame(), None, matches

    # tepat 1 cocok → langsung rekomendasikan
    found_title = matches[0]
    idx_raw = song_index[found_title]
    idx     = int(idx_raw.iloc[0]) if hasattr(idx_raw, 'iloc') else int(idx_raw)

    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]  # exclude diri sendiri

    lagu_idx  = [i[0] for i in sim_scores]
    lagu_skor = [round(i[1], 6) for i in sim_scores]

    result = df.iloc[lagu_idx][[track_col, artist_col, 'Genres', 'Album Name']].copy()
    result.columns = ['Track Name', 'Artist Name(s)', 'Genres', 'Album Name']
    result['Similarity Score'] = lagu_skor
    result['Rank'] = list(range(1, k + 1))
    result = result.reset_index(drop=True)

    input_info = df.iloc[idx]
    return result, input_info, []


def evaluasi_rekomendasi(judul, k):
    """Hitung Precision@K, Recall@K, F1@K untuk satu lagu.
    
    Rumus:
    - Precision@K = TP / (TP + FP) = TP / K
    - Recall@K    = TP / (TP + FN) = TP / Total_Relevan
    """
    if model_data is None:
        return {}

    df        = model_data['df']
    track_col = 'Track Name Original' if 'Track Name Original' in df.columns else 'Track Name'

    idx_list = [i for i, s in enumerate(df[track_col]) if s == judul]
    if not idx_list:
        return {}
    idx = idx_list[0]

    genres_input = get_genres_set(df.iloc[idx]['Genres'])
    if not genres_input:
        return {'precision': 0, 'recall': 0, 'f1': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'total_relevan': 0, 'K': k}

    # ── fix: unpack 3 nilai ──
    recs, inp_info, _ = rekomendasikan_lagu_app(judul, k=k)
    if recs.empty:
        return {}

    TP = sum(bool(genres_input & get_genres_set(row['Genres']))
             for _, row in recs.iterrows())

    FP = k - TP  # False Positive = rekomendasi tidak relevan

    total_relevan = sum(
        1 for _, row in df.iterrows()
        if get_genres_set(row['Genres']) & genres_input
        and row[track_col] != judul
    )

    FN = total_relevan - TP  # False Negative = lagu relevan yang tidak masuk top-K

    precision = TP / k if k > 0 else 0
    recall    = TP / total_relevan if total_relevan > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    return {
        'precision': precision, 'recall': recall, 'f1': f1,
        'TP': TP, 'FP': FP, 'FN': FN, 'total_relevan': total_relevan, 'K': k
    }


# ══════════════════════════════════════════════════════════
# FUNGSI TAMPILKAN HASIL
# ══════════════════════════════════════════════════════════
def tampilkan_hasil_rekomendasi(recs, input_info, track_col, artist_col, top_k):
    """Render kartu lagu input + daftar rekomendasi."""
    if input_info is None:
        st.error("❌ Lagu tidak ditemukan.")
        return

    nama_lagu  = input_info[track_col]
    nama_artis = input_info[artist_col]
    nama_album = input_info.get('Album Name', '')

    spotify_search = (
        f"https://open.spotify.com/search/"
        f"{nama_lagu.replace(' ','%20')}%20{nama_artis.replace(' ','%20')}"
    )

    genres_list = str(input_info['Genres']).replace(',', ' ').split()
    genre_tags  = " ".join([f'<span class="genre-tag">{g}</span>' for g in genres_list[:6]])

    st.markdown(f"""
    <div class="input-song-card">
        <h3>🎵 {nama_lagu}</h3>
        <p>👤 {nama_artis} | 💿 {nama_album}</p>
        <div>{genre_tags}</div><br>
        <a href="{spotify_search}" target="_blank"
           style="background:#1DB954;color:white;padding:6px 16px;
                  border-radius:20px;text-decoration:none;">
           🎧 Buka di Spotify
        </a>
    </div>
    """, unsafe_allow_html=True)

    if recs.empty:
        st.warning("Tidak ada rekomendasi yang ditemukan.")
        return

    st.markdown(
        f'<div class="section-header">Top-{top_k} Rekomendasi Lagu</div>',
        unsafe_allow_html=True
    )

    for _, row in recs.iterrows():
        pct = row['Similarity Score'] * 100
        spotify_link = (
            f"https://open.spotify.com/search/"
            f"{row['Track Name'].replace(' ','%20')}%20"
            f"{row['Artist Name(s)'].replace(' ','%20')}"
        )
        rec_genres     = str(row['Genres']).replace(',', ' ').split()
        rec_genre_tags = " ".join([
            f'<span class="genre-tag">{g}</span>'
            for g in rec_genres[:5]
        ])
        st.markdown(f"""
        <div class="rec-card">
            <b>#{row['Rank']} — {row['Track Name']}</b><br>
            👤 {row['Artist Name(s)']} | 💿 {row['Album Name']}<br>
            <div style="margin: 6px 0 4px 0;">{rec_genre_tags}</div>
            <div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;">
                <small style="color:#aaa;">Similarity</small>
                <small style="color:#1ed760;font-weight:600;">{pct:.1f}%</small>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct:.1f}%;"></div>
            </div>
            <a href="{spotify_link}" target="_blank"
               style="background:#1DB954;color:white;padding:4px 12px;
                      border-radius:15px;text-decoration:none;">
               🎧 Spotify
            </a>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.download_button(
        label="📥 Download Hasil Rekomendasi (CSV)",
        data=recs.to_csv(index=False),
        file_name=f"rekomendasi_{nama_lagu[:30].replace(' ','_')}.csv",
        mime="text/csv"
    )


# ══════════════════════════════════════════════════════════
# HALAMAN UTAMA
# ══════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🎵 Sistem Rekomendasi Musik</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Content-Based Filtering · TF-IDF + Audio Features · Cosine Similarity</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;margin-bottom:2rem;opacity:0.6;font-size:0.9rem;">Temukan lagu favoritmu berdasarkan kemiripan genre dan karakteristik audio 🎧</div>', unsafe_allow_html=True)

# ── sidebar ────────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg", width=60)
    except Exception:
        st.write("🎵")

    st.title("⚙️ Pengaturan")
    st.markdown("---")
    top_k = st.slider("Jumlah Rekomendasi (K)", min_value=3, max_value=20, value=10)
    st.markdown("---")

    st.markdown("**📊 Tentang Model**")
    if model_data is not None:
        df_info = model_data['df']
        st.info(f"""
        - **Lagu** : {len(df_info):,}
        - **Fitur** : TF-IDF + 9 Audio
        - **Metrik** : Cosine Similarity
        """)
    else:
        st.warning("⚠️ Model belum ditemukan.\nJalankan `music_recommendation.py` terlebih dahulu.")

    st.markdown("---")
    st.markdown("**🔬 Fitur Audio yang Digunakan**")
    for feat in ["🎤 Danceability", "⚡ Energy", "🔊 Loudness",
                 "🗣️ Speechiness", "🎸 Acousticness", "🎻 Instrumentalness",
                 "🎭 Liveness", "😊 Valence", "⏱️ Tempo"]:
        st.markdown(f"  `{feat}`")

# ── cek model ──────────────────────────────────────────────
if model_data is None:
    st.error("❌ Model belum tersedia. Jalankan `music_recommendation.py` terlebih dahulu.")
    st.code("python music_recommendation.py", language="bash")
    st.stop()

df_app     = model_data['df']
track_col  = 'Track Name Original'  if 'Track Name Original'  in df_app.columns else 'Track Name'
artist_col = 'Artist Name Original' if 'Artist Name Original' in df_app.columns else 'Artist Name(s)'

all_songs = sorted(df_app[track_col].dropna().unique().tolist())

# ── tab navigasi ───────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎵 Rekomendasi",
    "📊 Evaluasi Sistem",
    "🔍 Eksplorasi Data",
    "📚 Cara Kerja",
    "📈 Analytics"
])

# ════════════════════════════════
# REKOMENDASI
# ════════════════════════════════
with tab1:
    st.markdown("### 🎵 Rekomendasi Lagu")
    st.caption("💡 Ketik sebagian judul lagu, lalu tekan **Enter** atau klik tombol.")

    # ── Inisialisasi session state ──
    if "matches_list" not in st.session_state:
        st.session_state.matches_list = []
    if "show_recs" not in st.session_state:
        st.session_state.show_recs = False
    if "recs_data" not in st.session_state:
        st.session_state.recs_data = None
    if "input_info_data" not in st.session_state:
        st.session_state.input_info_data = None

    # ── form: otomatis submit ──
    with st.form(key="search_form", border=False):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            query_input = st.text_input(
                "🔎 Ketik judul lagu:",
                placeholder="Contoh: Blinding (tidak perlu lengkap)",
                key="input_judul_lagu"
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            tombol_cari = st.form_submit_button("Rekomendasikan 🎵")

    # ── enter ditekan ──
    if tombol_cari:
        if not query_input.strip():
            st.error("❌ Masukkan judul lagu terlebih dahulu.")
        else:
            recs, input_info, matches = rekomendasikan_lagu_app(query_input.strip(), k=top_k)

            if input_info is None and not matches:
                # Tidak ditemukan sama sekali
                st.error(f"❌ Lagu '{query_input}' tidak ditemukan di dataset.")
                st.session_state.matches_list = []
                st.session_state.show_recs    = False

            elif matches:
                # Lebih dari 1 cocok → simpan ke session state, tampilkan dropdown
                st.session_state.matches_list = matches
                st.session_state.show_recs    = False
                st.session_state.recs_data    = None

            else:
                # Tepat 1 cocok → simpan hasil ke session state
                st.session_state.matches_list    = []
                st.session_state.recs_data       = recs
                st.session_state.input_info_data = input_info
                st.session_state.show_recs       = True

    # ── tampilkan dropdown jika ada banyak pilihan ──
    if st.session_state.matches_list:
        matches = st.session_state.matches_list
        st.success(f"🔎 Ditemukan {len(matches)} lagu yang cocok. Pilih salah satu:")

        selected_from_dropdown = st.selectbox(
            "🎵 Pilih judul lagu yang diinginkan:",
            options=matches,
            key="pilihan_lagu_dropdown"
        )

        if st.button("✅ Konfirmasi & Rekomendasikan", key="btn_konfirmasi"):
            recs_final, input_info_final, _ = rekomendasikan_lagu_app(
                selected_from_dropdown, k=top_k
            )
            # Simpan ke session state dan hapus matches list
            st.session_state.recs_data       = recs_final
            st.session_state.input_info_data = input_info_final
            st.session_state.show_recs       = True
            st.session_state.matches_list    = []
            st.rerun()

    # ── tampilkan hasil rekomendasi ──
    if st.session_state.show_recs and st.session_state.recs_data is not None:
        tampilkan_hasil_rekomendasi(
            st.session_state.recs_data,
            st.session_state.input_info_data,
            track_col,
            artist_col,
            top_k
        )


# ════════════════════════════════
# EVALUASI
# ════════════════════════════════
with tab2:
    st.markdown("### 📊 Evaluasi Sistem Rekomendasi")
    st.markdown("""
    Evaluasi menggunakan **Precision@K**, **Recall@K**, dan **F1-Score@K**.
    Sebuah rekomendasi dianggap **relevan** jika memiliki minimal satu genre yang sama.
    """)

    # ── inisialisasi session state evaluasi ──
    if "eval_matches_list" not in st.session_state:
        st.session_state.eval_matches_list  = []
    if "eval_song_selected" not in st.session_state:
        st.session_state.eval_song_selected = None

    # ── form pencarian lagu evaluasi ──
    with st.form(key="eval_search_form", border=False):
        col_e1, col_e2, col_e3 = st.columns([3, 1, 1])
        with col_e1:
            eval_query = st.text_input(
                "🔎 Ketik judul lagu:",
                placeholder="Contoh: Blinding (tidak perlu lengkap)",
                key="input_eval_lagu"
            )
        with col_e2:
            k_eval = st.number_input("K", min_value=1, max_value=20, value=10, key='k_eval')
        with col_e3:
            st.markdown("<br>", unsafe_allow_html=True)
            tombol_eval = st.form_submit_button("🔬 Hitung Evaluasi")

    # ── proses pencarian ──
    if tombol_eval:
        if not eval_query.strip():
            st.error("❌ Masukkan judul lagu terlebih dahulu.")
        else:
            matches_eval = search_song_fuzzy(eval_query.strip(), model_data['song_index'])
            if not matches_eval:
                st.error(f"❌ Lagu '{eval_query}' tidak ditemukan.")
                st.session_state.eval_matches_list  = []
                st.session_state.eval_song_selected = None
            elif len(matches_eval) > 1:
                st.session_state.eval_matches_list  = matches_eval
                st.session_state.eval_song_selected = None
            else:
                st.session_state.eval_song_selected = matches_eval[0]
                st.session_state.eval_matches_list  = []

    # ── dropdown jika banyak pilihan ──
    if st.session_state.eval_matches_list:
        st.success(f"🔎 Ditemukan {len(st.session_state.eval_matches_list)} lagu. Pilih salah satu:")
        eval_selected_dd = st.selectbox(
            "🎵 Pilih judul lagu:",
            options=st.session_state.eval_matches_list,
            key="eval_dropdown"
        )
        if st.button("✅ Konfirmasi & Evaluasi", key="btn_eval_konfirmasi"):
            st.session_state.eval_song_selected = eval_selected_dd
            st.session_state.eval_matches_list  = []
            st.rerun()

    # ── jalankan evaluasi bila lagu sudah dipilih ──
    eval_song = st.session_state.eval_song_selected
    if eval_song:
        st.info(f"🎵 Lagu dipilih: **{eval_song}**")
        with st.spinner("Menghitung evaluasi..."):
            ev             = evaluasi_rekomendasi(eval_song, k=k_eval)
            recs_ev, inp_ev, _ = rekomendasikan_lagu_app(eval_song, k=k_eval)

        if ev:
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(f"Precision@{k_eval}", f"{ev['precision']:.4f}", f"{ev['precision']*100:.1f}%")
            col_m2.metric(f"Recall@{k_eval}",    f"{ev['recall']:.6f}",   f"{ev['recall']*100:.2f}%")
            col_m3.metric(f"F1-Score@{k_eval}",  f"{ev['f1']:.4f}",       f"{ev['f1']*100:.1f}%")

            st.markdown("---")
            st.markdown("#### 🧮 Perhitungan Manual")
            st.code(
                f"Lagu Input    : {eval_song}\n"
                f"K             : {k_eval}\n"
                f"TP (relevan)  : {ev['TP']}\n"
                f"Total Relevan : {ev['total_relevan']:,}\n\n"
                f"Precision@{k_eval} = TP / K\n"
                f"             = {ev['TP']} / {k_eval}\n"
                f"             = {ev['precision']:.4f}\n\n"
                f"Recall@{k_eval}    = TP / Total Relevan dalam Dataset\n"
                f"             = {ev['TP']} / {ev['total_relevan']}\n"
                f"             = {ev['recall']:.8f}\n\n"
                f"F1@{k_eval}        = 2 x (P x R) / (P + R)\n"
                f"             = 2 x ({ev['precision']:.4f} x {ev['recall']:.8f})\n"
                f"               / ({ev['precision']:.4f} + {ev['recall']:.8f})\n"
                f"             = {ev['f1']:.8f}"
            )

            if not recs_ev.empty and inp_ev is not None:
                st.markdown("#### 📋 Detail Relevansi Tiap Rekomendasi")
                genres_input = get_genres_set(inp_ev['Genres'])
                data_tabel   = []
                for _, row in recs_ev.iterrows():
                    genres_rec = get_genres_set(row['Genres'])
                    relevan    = bool(genres_input & genres_rec)
                    irisan     = genres_input & genres_rec
                    data_tabel.append({
                        'Rank'        : int(row['Rank']),
                        'Judul Lagu'  : row['Track Name'],
                        'Artis'       : row['Artist Name(s)'],
                        'Similarity'  : f"{row['Similarity Score']:.4f}",
                        'Genre Irisan': ', '.join(irisan) if irisan else '-',
                        'Relevan'     : '✅ Ya' if relevan else '❌ Tidak'
                    })
                st.dataframe(pd.DataFrame(data_tabel), use_container_width=True)
        else:
            st.warning("Tidak dapat menghitung evaluasi untuk lagu ini.")

    st.markdown("---")
    st.markdown("### 📊 Evaluasi Batch (Multiple Lagu)")
    n_batch = st.slider("Jumlah lagu yang dievaluasi", 5, 50, 10)
    k_batch = st.number_input("K untuk batch evaluasi", 1, 20, 10, key='k_batch')

    if st.button("🚀 Jalankan Evaluasi Batch"):
        sample_songs  = all_songs[:n_batch]
        batch_results = []
        progress      = st.progress(0)

        for i, song in enumerate(sample_songs):
            res = evaluasi_rekomendasi(song, k=k_batch)
            if res:
                batch_results.append({
                    'Judul Lagu'            : song,
                    'TP'                    : res['TP'],
                    f'Precision@{k_batch}'  : round(res['precision'], 4),
                    f'Recall@{k_batch}'     : round(res['recall'],    6),
                    f'F1@{k_batch}'         : round(res['f1'],        4),
                })
            progress.progress((i + 1) / len(sample_songs))

        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            st.dataframe(batch_df, use_container_width=True)

            avg    = batch_df[[f'Precision@{k_batch}', f'Recall@{k_batch}', f'F1@{k_batch}']].mean()
            b1, b2, b3 = st.columns(3)
            b1.metric(f"Avg Precision@{k_batch}", f"{avg[f'Precision@{k_batch}']:.4f}")
            b2.metric(f"Avg Recall@{k_batch}",    f"{avg[f'Recall@{k_batch}']:.6f}")
            b3.metric(f"Avg F1@{k_batch}",        f"{avg[f'F1@{k_batch}']:.4f}")

            st.download_button(
                "📥 Download Hasil Evaluasi Batch",
                batch_df.to_csv(index=False),
                "evaluasi_batch.csv", "text/csv"
            )


# ════════════════════════════════
# EKSPLORASI DATA
# ════════════════════════════════
with tab3:
    st.markdown("### 🔍 Eksplorasi Dataset")

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    col_d1.metric("Total Lagu",       f"{len(df_app):,}")
    col_d2.metric("Unique Artis",     f"{df_app[artist_col].nunique():,}")
    col_d3.metric("Unique Genre Tag", "300+")
    col_d4.metric("Fitur Audio",      "9")

    st.markdown("---")
    st.markdown(f"**📋 Dataset Lengkap ({len(df_app):,} lagu)**")
    preview_cols = [c for c in [track_col, artist_col, 'Album Name', 'Genres',
                                 'Danceability', 'Energy', 'Valence', 'Tempo']
                    if c in df_app.columns]
    st.dataframe(df_app[preview_cols], use_container_width=True, height=500)

    st.markdown("---")
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("**🎛️ Statistik Fitur Audio**")
        AUDIO_DISPLAY = [c for c in ['Danceability', 'Energy', 'Loudness',
                                      'Speechiness', 'Acousticness', 'Instrumentalness',
                                      'Liveness', 'Valence', 'Tempo']
                         if c in df_app.columns]
        st.dataframe(df_app[AUDIO_DISPLAY].describe().round(4), use_container_width=True)

    with col_s2:
        st.markdown("**🏷️ Top 15 Genre Terpopuler**")
        all_genres_list = []
        for g in df_app['Genres'].dropna():
            all_genres_list.extend([x.strip() for x in str(g).split(',') if x.strip()])
        genre_counts_explore = pd.Series(all_genres_list).value_counts().head(15)
        st.dataframe(
            pd.DataFrame({'Genre': genre_counts_explore.index, 'Jumlah': genre_counts_explore.values}),
            use_container_width=True
        )


# ════════════════════════════════
# CARA KERJA
# ════════════════════════════════
with tab4:
    st.markdown("### 📚 Cara Kerja Sistem")

    # ── Alur Pipeline ──────────────────────────────────────
    st.markdown("""
```
Dataset CSV
    │
    ▼
[1] PRE-PROCESSING
    ├─ Data Cleaning       → hapus kolom tidak relevan, duplikat, missing values
    ├─ Case Folding & Tok  → lowercase + tokenisasi (Genres & Artist)
    └─ Normalisasi         → Min-Max Scaling pada 9 fitur audio → rentang [0, 1]
    │
    ▼
[2] REPRESENTASI DATA
    ├─ TF-IDF  (Genres + Artist)  → 300 fitur teks  × bobot 0.6
    ├─ 9 Fitur Audio (normalized) →   9 fitur audio × bobot 0.4
    └─ Digabung → 1 vektor per lagu (309 dimensi)
    │
    ▼
[3] COSINE SIMILARITY
    └─ Matriks N × N — setiap pasang lagu dihitung kesamaannya
    │
    ▼
[4] REKOMENDASI
    └─ Urutkan skor similarity → kembalikan Top-K lagu tertinggi
```
""")

    st.markdown("---")

    # ── 0. kolom yang dihapus ─────────────────────────────
    st.markdown("#### 🗑️ Kolom yang Dihapus saat Data Cleaning")
    st.markdown(
        "Dataset awal memiliki banyak kolom yang **tidak relevan** untuk sistem rekomendasi berbasis konten. "
        "Kolom-kolom berikut dihapus karena tidak digunakan sebagai fitur:"
    )
    col_del1, col_del2 = st.columns(2)
    with col_del1:
        st.markdown("""
| Kolom Dihapus | Alasan |
|---|---|
| `Track ID` | ID internal, bukan fitur konten |
| `Mode` | Digantikan fitur audio lainnya |
""")
    st.caption("💡 Selain dihapus kolom, baris dengan nilai kosong (*missing values*) dan data duplikat juga dihapus.")
    st.markdown("---")

    # ── 1. TF-IDF ─────────────────────────────────────────
    st.markdown("#### 1️⃣ TF-IDF *(Term Frequency – Inverse Document Frequency)*")
    st.markdown(
        "Digunakan untuk mengubah fitur teks **(Genres & Artist Name)** menjadi vektor numerik. "
        "Kata yang sering muncul di satu lagu tapi jarang di lagu lain mendapat bobot lebih tinggi."
    )
    st.latex(r"\text{TF-IDF}(t,\, d) = \text{TF}(t,\, d) \times \text{IDF}(t)")
    st.markdown("---")

    # ── 2. Min-Max Normalization ───────────────────────────
    st.markdown("#### 2️⃣ Min-Max Normalization")
    st.markdown(
        "Digunakan untuk **menyamakan skala** 9 fitur audio sebelum digabung dengan vektor TF-IDF. "
        "Tanpa normalisasi, fitur dengan rentang besar (misal *Tempo* 0–250 BPM) akan mendominasi "
        "fitur kecil (misal *Danceability* 0–1) sehingga hasil similarity menjadi tidak adil. "
        "Min-Max mengubah semua fitur audio ke rentang **[0, 1]**."
    )
    st.latex(r"x_{\text{norm}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}")
    st.markdown("""
| Fitur Audio | Rentang Asli | Setelah Normalisasi |
|---|---|---|
| Danceability | 0.0 – 1.0 | 0.0 – 1.0 |
| Energy | 0.0 – 1.0 | 0.0 – 1.0 |
| Loudness | -60 dB – 0 dB | 0.0 – 1.0 |
| Tempo | 0 – 250 BPM | 0.0 – 1.0 |
| *(fitur lainnya)* | berbeda-beda | 0.0 – 1.0 |
""")
    st.markdown("---")

    # ── 3. Cosine Similarity ──────────────────────────────
    st.markdown("#### 3️⃣ Cosine Similarity")
    st.markdown(
        "Mengukur **kemiripan dua lagu** berdasarkan sudut antara vektor gabungan mereka. "
        "Nilai **1.0** = identik, **0.0** = tidak mirip sama sekali."
    )
    st.latex(
        r"\text{similarity}(A, B) = \cos(\theta) = "
        r"\frac{A \cdot B}{\|A\|\,\|B\|} = "
        r"\frac{\displaystyle\sum_{i=1}^{n} A_i B_i}"
        r"{\sqrt{\displaystyle\sum_{i=1}^{n} A_i^2}\;"
        r"\sqrt{\displaystyle\sum_{i=1}^{n} B_i^2}}"
    )
    st.markdown("---")

    # ── 4. Metrik Evaluasi ────────────────────────────────
    st.markdown("#### 4️⃣ Metrik Evaluasi")

    st.markdown("**Precision@K** — dari K rekomendasi, berapa yang relevan?")
    st.latex(
        r"\text{Precision at } K = "
        r"\frac{\text{Number of relevant items in } K}"
        r"{\text{Total number of items in } K}"
    )

    st.markdown("**Recall@K** — dari semua lagu relevan di dataset, berapa yang berhasil direkomendasikan?")
    st.latex(
        r"\text{Recall}@K = "
        r"\frac{\text{Relevant items in Top-K}}"
        r"{\text{Total relevant items}}"
    )

    st.markdown("**F1@K** — rata-rata harmonis antara Precision dan Recall.")
    st.latex(
        r"F1@K = 2 \times "
        r"\frac{\text{Precision}@K \times \text{Recall}@K}"
        r"{\text{Precision}@K + \text{Recall}@K}"
    )
    st.markdown("---")

    # ── Kolom yang dipakai ────────────────────────────────
    st.markdown("#### 📋 Kolom yang Digunakan")
    st.markdown("""
| Kolom | Jenis | Peran |
|---|---|---|
| `Genres` | Teks | Fitur utama TF-IDF (bobot 60%) |
| `Artist Name(s)` | Teks | Fitur pendukung TF-IDF (bobot 60%) |
| `Danceability` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Energy` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Loudness` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Speechiness` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Acousticness` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Instrumentalness` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Liveness` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Valence` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
| `Tempo` | Audio | Dinormalisasi Min-Max → vektor audio (bobot 40%) |
""")

# ════════════════════════════════
# ANALYTICS
# ════════════════════════════════
with tab5:
    st.markdown("### 📈 Analytics - Top 10 Visualizations")

    AUDIO_FEATURES    = ['Danceability', 'Energy', 'Loudness', 'Speechiness',
                         'Acousticness', 'Instrumentalness', 'Liveness', 'Valence', 'Tempo']
    available_audio   = [col for col in AUDIO_FEATURES if col in df_app.columns]

    st.markdown("---")
    col_anal1, col_anal2 = st.columns(2)

    with col_anal1:
        st.markdown("#### 💃 Top 10 Songs by Danceability")
        try:
            top_dance = df_app.nlargest(10, 'Danceability')[
                [track_col, artist_col, 'Danceability']
            ].reset_index(drop=True)
            top_dance.columns = ['Track Name', 'Artist', 'Danceability']
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            colors1   = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 10))[::-1]
            ax1.barh(top_dance['Track Name'], top_dance['Danceability'], color=colors1)
            ax1.set_xlabel('Danceability'); ax1.set_title('Top 10 by Danceability', fontweight='bold')
            ax1.invert_yaxis(); ax1.set_xlim(0, 1.1)
            st.pyplot(fig1)
            with st.expander("View Data"):
                st.dataframe(top_dance, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    with col_anal2:
        st.markdown("#### ⚡ Top 10 Songs by Energy")
        try:
            top_energy = df_app.nlargest(10, 'Energy')[
                [track_col, artist_col, 'Energy']
            ].reset_index(drop=True)
            top_energy.columns = ['Track Name', 'Artist', 'Energy']
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            colors2   = plt.cm.Oranges(np.linspace(0.3, 0.9, 10))[::-1]
            ax2.barh(top_energy['Track Name'], top_energy['Energy'], color=colors2)
            ax2.set_xlabel('Energy'); ax2.set_title('Top 10 by Energy', fontweight='bold')
            ax2.invert_yaxis(); ax2.set_xlim(0, 1.1)
            st.pyplot(fig2)
            with st.expander("View Data"):
                st.dataframe(top_energy, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    col_anal3, col_anal4 = st.columns(2)

    with col_anal3:
        st.markdown("#### 🎤 Top 10 Most Productive Artists")
        try:
            artist_counts = df_app[artist_col].value_counts().head(10)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            colors3   = plt.cm.Spectral(np.linspace(0.2, 0.8, 10))[::-1]
            ax3.barh(artist_counts.index, artist_counts.values, color=colors3)
            ax3.set_xlabel('Number of Songs'); ax3.set_title('Top 10 Artists', fontweight='bold')
            ax3.invert_yaxis()
            st.pyplot(fig3)
            with st.expander("View Data"):
                st.dataframe(
                    artist_counts.rename('Song Count').reset_index()
                                 .rename(columns={'index': 'Artist'}),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {e}")

    with col_anal4:
        st.markdown("#### 🎼 Genre Distribution")
        try:
            all_genres_analytics = []
            for g in df_app['Genres'].dropna():
                all_genres_analytics.extend([x.strip() for x in str(g).split(',') if x.strip()])
            genre_counts_analytics = pd.Series(all_genres_analytics).value_counts().head(10)

            fig4, ax4 = plt.subplots(figsize=(8, 6))
            colors4   = plt.cm.Set3(np.linspace(0, 1, 10))
            ax4.pie(
                genre_counts_analytics.values,
                labels=genre_counts_analytics.index,
                autopct='%1.1f%%', colors=colors4,
                textprops={'fontsize': 9}, pctdistance=0.75
            )
            ax4.set_title('Top 10 Genres', fontweight='bold')
            st.pyplot(fig4)
            with st.expander("View Data"):
                st.dataframe(
                    genre_counts_analytics.rename('Count').reset_index()
                                          .rename(columns={'index': 'Genre'}),
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("### 📊 Dataset Statistics")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Total Songs",     f"{len(df_app):,}")
    col_stat2.metric("Unique Artists",  f"{df_app[artist_col].nunique():,}")
    col_stat3.metric("Unique Genres",   "300+")
    col_stat4.metric("Audio Features",  f"{len(available_audio)}")
