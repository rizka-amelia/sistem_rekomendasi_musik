# ============================================================
#  SISTEM REKOMENDASI MUSIK BERBASIS CONTENT-BASED FILTERING
#  Menggunakan TF-IDF + Audio Features + Cosine Similarity
# ============================================================

import pandas as pd
import numpy as np
import re
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import preprocessing functions
from pre_processing import (
    preprocess_full, AUDIO_COLS, TEXT_COLS, normalisasi_artist
)

print("=" * 60)
print("  SISTEM REKOMENDASI MUSIK - CONTENT-BASED FILTERING")
print("=" * 60)

# ── JALANKAN PREPROCESSING ────────────────────────────────────────
print("\n[BAB 3] PRE-PROCESSING DATA")
print("=" * 60)

preprocess_result = preprocess_full()
df_final = preprocess_result['df_norm']
scaler    = preprocess_result['scaler']

# ============================================================
# REPRESENTASI DATA
# ============================================================
print("\n REPRESENTASI DATA")
print("=" * 60)

# ── TF-IDF pada Fitur Teks ────────────────────────────
print("\n  TF-IDF (Term Frequency - Inverse Document Frequency)")

tfidf = TfidfVectorizer(
    max_features=300,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

tfidf_matrix = tfidf.fit_transform(df_final['text_combined'])
feature_names = tfidf.get_feature_names_out()

print(f"\n  Dimensi TF-IDF Matrix  : {tfidf_matrix.shape}")
print(f"  Jumlah fitur teks      : {len(feature_names)}")

# ──  representasi vektor gabungan ──────────────────────
print("\n   Penggabungan Fitur Teks (TF-IDF) + Fitur Audio")

audio_matrix = df_final[AUDIO_COLS].values

W_TEXT = 0.6
W_AUDIO = 0.4

tfidf_dense = tfidf_matrix.toarray()
combined_matrix = np.hstack([
    tfidf_dense * W_TEXT,
    audio_matrix * W_AUDIO
])

print(f"\n  Komponen vektor gabungan:")
print(f"    - Fitur TF-IDF   : {tfidf_dense.shape[1]:>5} dimensi × bobot {W_TEXT}")
print(f"    - Fitur Audio  : {audio_matrix.shape[1]:>5} dimensi × bobot {W_AUDIO}")
print(f"    - Total dimensi : {combined_matrix.shape[1]:>5}")

# ============================================================
# IMPLEMENTASI SISTEM REKOMENDASI
# ============================================================
print("\n IMPLEMENTASI SISTEM REKOMENDASI")
print("=" * 60)

# ── content- based filtering ───────────────────────────
print("\n  Content Based Filtering")
print("-" * 60)
print("""
  Content Based Filtering adalah teknik yang merekomendasikan
  item berdasarkan kemiripan konten dengan item
  yang sebelumnya pernah di-like oleh user.
  
  Setiap lagu direpresentasikan sebagai vektor fitur:
    - Fitur TF-IDF (Genres + Artist) x 60%
    - Fitur Audio (9 karakteristik) x 40%
  
  Kemiripan dihitung dengan Cosine Similarity.
""")

# ── teori cosine similarity ────────────────────────
print("\n Teori Cosine Similarity")
print("-" * 60)
print("""
  Cosine Similarity mengukur kemiripan dua vektor.
  
  Rumus:
              A · B
    cos(theta) = -----------------
              ||A|| × ||B||
  
  Interpretasi nilai:
    - 1.0 = vektor identik (paling mirip)
    - 0.0 = tidak ada kemiripan
    - -1.0 = berlawanan sama sekali
""")

# ── Contoh Perhitungan Manual ──────────────────────
print("\n  Contoh Perhitungan Manual")
print("-" * 60)

vec_a = combined_matrix[0]
vec_b = combined_matrix[1]

dot = np.dot(vec_a, vec_b)
norm_a = np.linalg.norm(vec_a)
norm_b = np.linalg.norm(vec_b)
cos_manual = dot / (norm_a * norm_b)

print(f"  Vektor A (Lagu 1): {len(vec_a)} dimensi")
print(f"  Vektor B (Lagu 2): {len(vec_b)} dimensi")
print(f"\n  Perhitungan:")
print(f"    A · B   = {dot:.6f}")
print(f"    ||A||  = {norm_a:.6f}")
print(f"    ||B||  = {norm_b:.6f}")
print(f"    cos(A,B)= {cos_manual:.6f}")
print(f"\n  Hasil: {cos_manual:.4f}")

# ─ matriks similarity ────────────────────────────
print("\n  Matriks Similarity")
print("-" * 60)
print("""
  Matriks N×N menyimpan hasil cosine similarity
  antar setiap pasang lagu.
  
  Contoh untuk 5 lagu:
              |Lagu1|Lagu2|Lagu3|Lagu4|Lagu5|
    -----------------------------------------
    Lagu1    | 1.0 | 0.8 | 0.7 | 0.6 | 0.5 |
    Lagu2    | 0.8 | 1.0 | 0.7 | 0.6 | 0.5 |
    ...
  
  Catatan:
    - Diagonal = 1.0 (lagu dengan dirinya sendiri)
    - Matriks Simetris
""")

# ── perhitungan matriks ────────────────────────
print("\n  Perhitungan Matriks")
print("-" * 60)

n_lagu = combined_matrix.shape[0]
print(f"  Jumlah lagu         : {n_lagu:,}")
print(f"  Dimensi vektor     : {combined_matrix.shape[1]}")
print(f"  Ukuran matriks     : {n_lagu} × {n_lagu}")
print(f"  Total perbandingan: {n_lagu*n_lagu:,}")

cosine_sim_matrix = cosine_similarity(combined_matrix)

print(f"\n  ✓ Matriks berhasil dihitung")
print(f"  ✓ Dimensi : {cosine_sim_matrix.shape}")

# ── sample matriks ────────────────────────────
print("\n  sample Matriks (5×5)")
print("-" * 60)

n_sample = 5
print(f"\n{'':>38}", end="")
for t in df_final['Track Name'].iloc[:n_sample]:
    print(f" {t[:10]:>10}", end="")
print()
print("  " + "-" * (40 + n_sample*12))

for i in range(n_sample):
    print(f"  {df_final['Track Name'].iloc[i][:35]:<38}", end="")
    for j in range(n_sample):
        print(f" {cosine_sim_matrix[i,j]:>10.4f}", end="")
    print()

# ──  fungsi Rekomendasi ────────────────────────
print("\n" + "=" * 60)
print("   Fungsi Pengambilan Rekomendasi")
print("=" * 60)

song_index = pd.Series(
    df_final.index,
    index=df_final['Track Name']
).drop_duplicates()

# semua lagu dalam daftar 
all_songs = sorted(song_index.index.tolist())

print(f"\n  Jumlah lagu tersedia: {len(all_songs):,}")
print("\n  Contoh (10 pertama):")
for i, s in enumerate(all_songs[:10], 1):
    print(f"    {i}. {s}")

def lihat_daftar_lagu():
    """Tampilkan semua lagu."""
    print("\n" + "=" * 60)
    print("  DAFTAR SEMUA LAGU")
    print("=" * 60)
    for i, s in enumerate(all_songs, 1):
        print(f"    {i:>4}. {s}")

def pilih_lagu():
    """
    Input lagu dengan tiga cara:
    1. Ketik nomor (contoh: 1)
    2. Ketik nama lagu (contoh: Blinding Lights)
    3. Kosongkan untuk batal
    """
    print("\n" + "=" * 60)
    print("  PILIH LAGU")
    print("=" * 60)
    print("\n  Cara input:")
    print("    1. Ketik nomor (1, 2, 3, ...)")
    print("    2. Ketik nama lagu")
    print("    3. Kosongkan untuk batal")
    
    pilihan = input("\n  Pilih lagu: ").strip()
    
    if not pilihan:
        return None
    
    # cek nomor
    if pilihan.isdigit():
        idx = int(pilihan) - 1
        if 0 <= idx < len(all_songs):
            return all_songs[idx]
        else:
            print(f"  ❌ Nomor {pilihan} tidak valid!")
            return None
    
    # cari yang mirip
    matches = [s for s in all_songs if pilihan.lower() in s.lower()]
    
    if not matches:
        print(f"  ❌ Lagu '{pilihan}' tidak ditemukan!")
        return None
    
    # tampilkan hasil pencarian
    if len(matches) > 1:
        print(f"\n  Ditemukan {len(matches)} lagu:")
        for i, m in enumerate(matches[:10], 1):
            print(f"    {i}. {m}")
        
        pil = input("\n  Pilih nomor (Enter untuk pertama): ").strip()
        if pil.isdigit() and 1 <= int(pil) <= len(matches):
            return matches[int(pil)-1]
    
    return matches[0]

def rekomendasikan_lagu(judul_lagu: str = None, k: int = 10, exclude_same_artist: bool = False) -> pd.DataFrame:
    """
    Rekomendasikan k lagu paling mirip dengan lagu input.
    
    Parameter:
        judul_lagu (str): Nama/nomor lagu (opsional)
        k (int): Jumlah rekomendasi
        exclude_same_artist (bool): Exclude lagu dari artis yang sama
    
    Returns:
        DataFrame dengan rekomendasi
    """
    # jika tidak ada input, minta pilih
    if judul_lagu is None:
        judul_lagu = pilih_lagu()
    
    if judul_lagu is None:
        print("  ❌ Tidak ada lagu yang dipilih.")
        return pd.DataFrame()
    
    # cari dalam indeks
    matches = [s for s in song_index.index if judul_lagu.lower() in s.lower()]
    
    if not matches:
        print(f"  ❌ Lagu '{judul_lagu}' tidak ditemukan!")
        return pd.DataFrame()
    
    # ambil match pertama
    found_title = matches[0]
    idx = song_index[found_title]
    
    # ambil info artis untuk filtering
    artist_input = df_final.iloc[idx]['Artist Name(s)']
    
    # ambil & urutkan skor similarity
    sim_scores = sorted(
        enumerate(cosine_sim_matrix[idx]),
        key=lambda x: x[1], reverse=True
    )[1:]  # exclude diri sendiri
    
    # jika exclude_same_artist, filter hasil
    if exclude_same_artist:
        filtered_scores = []
        for i, score in sim_scores:
            rec_artist = df_final.iloc[i]['Artist Name(s)']
            if rec_artist != artist_input:
                filtered_scores.append((i, score))
            if len(filtered_scores) >= k:
                break
        sim_scores = filtered_scores
    else:
        sim_scores = sim_scores[:k]
    
    # ambil data lagu
    rec_indices = [i for i, _ in sim_scores]
    rec_scores = [round(s, 6) for _, s in sim_scores]
    
    result = df_final.iloc[rec_indices][
        ['Track Name', 'Artist Name(s)', 'Genres', 'Album Name']
    ].copy()
    result['Similarity Score'] = rec_scores
    result['Rank'] = list(range(1, len(rec_scores) + 1))
    
    return result.reset_index(drop=True)
    
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:k+1]
    
    lagu_idx = [i[0] for i in sim_scores]
    lagu_skor = [round(i[1], 6) for i in sim_scores]
    
    result = df_final.iloc[lagu_idx][
        ['Track Name', 'Artist Name(s)', 'Genres', 'Album Name']
    ].copy()
    result['Similarity Score'] = lagu_skor
    result['Rank'] = list(range(1, k + 1))
    result = result.reset_index(drop=True)
    return result


# ============================================================
# HASIL REKOMENDASI
# ============================================================
print("\n HASIL REKOMENDASI")
print("=" * 60)

TEST_SONGS = [
    df_final['Track Name'].iloc[0],
    df_final['Track Name'].iloc[50],
    df_final['Track Name'].iloc[100],
]

for test_song in TEST_SONGS:
    print(f"\n  🎵 Lagu Input: '{test_song}'")
    recs = rekomendasikan_lagu(test_song, k=5)
    if not recs.empty:
        print(f"  {'No':>3}  {'Judul Lagu':<38} {'Skor':>8}")
        print("  " + "-" * 55)
        for _, row in recs.iterrows():
            print(f"  {row['Rank']:>3}. {row['Track Name'][:36]:<38} {row['Similarity Score']:>8.4f}")

# simpan contoh rekomendasi
recs_sample = rekomendasikan_lagu(TEST_SONGS[0], k=10)
recs_sample.to_csv('hasil_05_contoh_rekomendasi.csv', index=False)
print(f"\n  📥 File tersimpan : hasil_05_contoh_rekomendasi.csv")


print("\n" + "=" * 60)
print("EVALUASI SISTEM")
print("=" * 60)
print("""
  Definisi:
    RELEVAN  = lagu yang memiliki minimal 1 genre sama dengan lagu input
    Filter   = lagu dari artis yang sama TIDAK dihitung sebagai relevan
 
  Metrik yang digunakan:
  ┌─────────────────────────────────────────────────────┐
  │  Precision@K                                        │
  │  Recall@K                                           │
  │  F1-Score@K                                         │
  └─────────────────────────────────────────────────────┘
  (masing-masing dijelaskan & dihitung secara terpisah)
""")
 
 
# ── Helper ─────────────────────────────────────────────
def get_genres_set(genre_str: str) -> set:
    """
    Ubah string genres menjadi set kata genre individual.
    Contoh: "indonesian pop, indie rock" → {'indonesian', 'pop', 'indie', 'rock'}
    Kata ≤ 2 karakter diabaikan (noise).
    """
    genres = re.split(r'[,\s]+', str(genre_str).lower().strip())
    return {g for g in genres if len(g) > 2}
 
 
# ── Ambil nilai TP dan total_relevan ───────────────────
def hitung_tp_dan_relevan(judul: str, k: int) -> dict | None:
    """
    Hitung nilai dasar yang dibutuhkan oleh ketiga metrik:
      - TP            : jumlah rekomendasi Top-K yang relevan
      - total_relevan : total lagu relevan di seluruh dataset
 
    Kedua nilai ini dihitung SEKALI lalu dipakai terpisah
    oleh fungsi Precision, Recall, dan F1.
    """
    idx_list = [i for i, s in enumerate(df_final['Track Name']) if s == judul]
    if not idx_list:
        return None
 
    idx          = idx_list[0]
    genres_input = get_genres_set(df_final.iloc[idx]['Genres'])
 
    if not genres_input:
        return None
 
    recs = rekomendasikan_lagu(judul, k=k, exclude_same_artist=True)
    if recs.empty:
        return None
 
    # ── Hitung TP ────────────────────────────────────────
    TP = sum(
        1 for _, row in recs.iterrows()
        if genres_input & get_genres_set(row['Genres'])
    )
 
    # ── Hitung Total Relevan di Dataset ──────────────────
    artist_input  = normalisasi_artist(df_final.iloc[idx]['Artist Name(s)'])
    total_relevan = sum(
        1 for _, row in df_final.iterrows()
        if get_genres_set(row['Genres']) & genres_input
        and row['Track Name'] != judul
        and not (normalisasi_artist(row['Artist Name(s)']) & artist_input)
    )
 
    return {
        'judul'        : judul,
        'k'            : k,
        'TP'           : TP,
        'total_relevan': total_relevan,
        'genres_input' : genres_input,
    }
 
 
# ── Precision@K ────────────────────────────────────────
def hitung_precision_k(base: dict) -> float:
    """
    Rumus:
 
                      TP
      Precision@K = ─────
                      K
 
    Menjawab:
      "Dari K rekomendasi yang diberikan, berapa persen yang relevan?"
 
    Parameter:
      base (dict) : hasil dari hitung_tp_dan_relevan()
 
    Returns:
      float : nilai Precision@K  (0.0 – 1.0)
    """
    TP = base['TP']
    K  = base['k']
 
    precision = TP / K if K > 0 else 0.0
 
    print(f"\n  ── Precision@{K} ─────────────────────────────")
    print(f"  Rumus  : Precision@K = TP / K")
    print(f"         = {TP} / {K}")
    print(f"         = {precision:.4f}  ({precision*100:.2f}%)")
    print(f"""
  Interpretasi:
    Dari {K} lagu yang direkomendasikan,
    {TP} lagu relevan (punya genre sama),
    sehingga Precision = {precision*100:.1f}%
""")
 
    return precision
 
 
# ── Recall@K ───────────────────────────────────────────
def hitung_recall_k(base: dict) -> float:
    """
    Rumus:
 
                    Relevant items in Top-K       TP
      Recall@K  = ─────────────────────────── = ────────────────
                      Total relevant items       Total_Relevan
 
    Menjawab:
      "Dari SEMUA lagu relevan yang ada di dataset,
       berapa persen yang berhasil direkomendasikan dalam Top-K?"
 
    Parameter:
      base (dict) : hasil dari hitung_tp_dan_relevan()
 
    Returns:
      float : nilai Recall@K  (0.0 – 1.0)
    """
    TP            = base['TP']
    K             = base['k']
    total_relevan = base['total_relevan']
 
    recall = TP / total_relevan if total_relevan > 0 else 0.0
 
    print(f"\n  ── Recall@{K} ───────────────────────────────")
    print(f"  Rumus  : Recall@K = TP / Total_Relevan_Dataset")
    print(f"         = {TP} / {total_relevan}")
    print(f"         = {recall:.6f}  ({recall*100:.4f}%)")
    print(f"""
  Interpretasi:
    Di seluruh dataset terdapat {total_relevan:,} lagu relevan,
    namun hanya {TP} yang muncul di Top-{K},
    sehingga Recall = {recall*100:.4f}%
""")
 
    return recall
 
 
# ── F1-Score@K ─────────────────────────────────────────
def hitung_f1_k(precision: float, recall: float, k: int) -> float:
    """
    Rumus:
 
                          Precision@K × Recall@K
      F1@K  = 2 × ─────────────────────────────────
                          Precision@K + Recall@K
 
    F1 adalah rata-rata HARMONIK dari Precision dan Recall.
    Untuk menilai keseimbangan antara keduanya.
 
    Parameter:
      precision (float) : hasil hitung_precision_k()
      recall    (float) : hasil hitung_recall_k()
      k         (int)   : nilai K yang digunakan
 
    Returns:
      float : nilai F1@K  (0.0 – 1.0)
    """
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
 
    print(f"\n  ── F1-Score@{k} ─────────────────────────────")
    print(f"  Rumus  : F1@K = 2 × (Precision@K × Recall@K)")
    print(f"                      ────────────────────────")
    print(f"                      (Precision@K + Recall@K)")
    print(f"""
         = 2 × ({precision:.4f} × {recall:.6f})
               ─────────────────────────────────
               ({precision:.4f} + {recall:.6f})
 
         = 2 × {precision * recall:.8f}
               ─────────────────────────
               {precision + recall:.8f}
 
         = {f1:.8f}  ({f1*100:.4f}%)
""")
    print(f"""
  Interpretasi:
    Precision  = {precision*100:.2f}%  (akurasi rekomendasi)
    Recall     = {recall*100:.4f}%  (cakupan dari dataset)
    F1-Score   = {f1*100:.4f}%  (keseimbangan keduanya)
""")
 
    return f1
 
# ── Fungsi Evaluasi Lengkap untuk Satu Lagu ────────────
def evaluasi_satu_lagu(judul: str, k: int) -> dict | None:
    """
    Menjalankan ketiga perhitungan (Precision, Recall, F1)
    secara terpisah dan berurutan untuk satu lagu.
 
    Alur:
      1. hitung_tp_dan_relevan()  → dapatkan TP & total_relevan
      2. hitung_precision_k()     → Precision@K = TP / K
      3. hitung_recall_k()        → Recall@K    = TP / Total_Relevan
      4. hitung_f1_k()            → F1@K        = 2×(P×R)/(P+R)
    """
    print(f"\n  {'─'*55}")
    print(f"  Evaluasi Lagu: '{judul}'  |  K = {k}")
    print(f"  {'─'*55}")
 
    base = hitung_tp_dan_relevan(judul, k)
    if base is None:
        print(f"  ❌ Tidak dapat mengevaluasi '{judul}'")
        return None
 
    print(f"\n  Genres Input  : {base['genres_input']}")
    print(f"  TP (relevan)  : {base['TP']}")
    print(f"  Total Relevan : {base['total_relevan']:,} lagu di dataset")
 
    precision = hitung_precision_k(base)
    recall    = hitung_recall_k(base)
    f1        = hitung_f1_k(precision, recall, k)
 
    return {
        'judul'                 : judul,
        'k'                     : k,
        'TP'                    : base['TP'],
        'total_relevan_dataset' : base['total_relevan'],
        'precision_k'           : precision,
        'recall_k'              : recall,
        'f1_k'                  : f1,
    }
 
# ── Jalankan Evaluasi ──────────────────────────────────
K_EVAL = 10
 
eval_songs = [
    df_final['Track Name'].iloc[i]
    for i in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    if i < len(df_final)
]
 
print(f"  Mengevaluasi {len(eval_songs)} lagu dengan K = {K_EVAL}")
print("  (setiap lagu ditampilkan perhitungan lengkapnya)\n")
 
eval_results = []
for song in eval_songs:
    res = evaluasi_satu_lagu(song, k=K_EVAL)
    if res:
        eval_results.append(res)
 
# ── Tabel Rekap Hasil ─────────────────────────────────
print("\n" + "=" * 60)
print("  REKAP HASIL EVALUASI")
print("=" * 60)
print(f"\n  {'No':>3}  {'Judul Lagu':<38}  {'P@K':>6}  {'R@K':>8}  {'F1@K':>8}")
print("  " + "-" * 72)
for i, res in enumerate(eval_results, 1):
    print(
        f"  {i:>3}. {res['judul'][:36]:<38}  "
        f"{res['precision_k']:>6.4f}  "
        f"{res['recall_k']:>8.6f}  "
        f"{res['f1_k']:>8.6f}"
    )
 
avg_p  = np.mean([r['precision_k'] for r in eval_results])
avg_r  = np.mean([r['recall_k']    for r in eval_results])
avg_f1 = np.mean([r['f1_k']        for r in eval_results])
 
print("  " + "-" * 72)
print(
    f"  {'RATA-RATA':>43}  "
    f"{avg_p:>6.4f}  "
    f"{avg_r:>8.6f}  "
    f"{avg_f1:>8.6f}"
)
 
# ── Ringkasan Akhir ───────────────────────────────────
print(f"""
  {'─'*45}
  RINGKASAN EVALUASI AKHIR  (K = {K_EVAL})
  {'─'*45}
  Precision@{K_EVAL}  : {avg_p:.4f}   ({avg_p*100:.2f}%)
  Recall@{K_EVAL}     : {avg_r:.6f}  ({avg_r*100:.4f}%)
  F1-Score@{K_EVAL}   : {avg_f1:.6f}  ({avg_f1*100:.4f}%)
  Lagu diuji   : {len(eval_results)}
  {'─'*45}

""")
 
# ── Simpan ke CSV ─────────────────────────────────────
eval_df = pd.DataFrame(eval_results)
eval_df.to_csv('hasil_06_evaluasi.csv', index=False)
print(f"  📥 File tersimpan : hasil_06_evaluasi.csv")
 
 
# ============================================================
# SIMPAN MODEL (.pkl)
# ============================================================
print("\n" + "=" * 60)
print("  MENYIMPAN MODEL")
print("=" * 60)
 
model_data = {
    'df'               : df_final,
    'combined_matrix'  : combined_matrix,
    'cosine_sim_matrix': cosine_sim_matrix,
    'song_index'       : song_index,
    'tfidf'            : tfidf,
    'scaler'           : scaler,
    'AUDIO_COLS'       : AUDIO_COLS,
    'W_TEXT'           : W_TEXT,
    'W_AUDIO'          : W_AUDIO,
}
 
with open('model_rekomendasi.pkl', 'wb') as f:
    pickle.dump(model_data, f)
 
print(f"\n  📦 Model tersimpan  : model_rekomendasi.pkl")
print(f"\n" + "=" * 60)
print(f"  ✓ PROSES SELESAI — jalankan: streamlit run app.py")
print(f"=" * 60)