import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ── Konfigurasi kolom ───────────────────────────────────────────────
KOLOM_HAPUS = [
    'Track ID', 'Track URI', 'Artist URI(s)', 'Album URI',
    'Album Artist Name(s)', 'Album Release Date', 'Disc Number',
    'Track Number', 'ISRC', 'Key', 'Mode', 'Time Signature'
]

TEXT_COLS = ['Track Name', 'Artist Name(s)', 'Genres']
AUDIO_COLS = ['Danceability', 'Energy', 'Loudness',
              'Speechiness', 'Acousticness', 'Instrumentalness',
              'Liveness', 'Valence', 'Tempo']


# ── Fungsi Helper ────────────────────────────────────────────────────
def case_fold(text):
    """Case folding: ubah teks menjadi lowercase."""
    if pd.isna(text):
        return ''
    return str(text).lower()


def tokenize(text):
    """Tokenization: pisahkan kata-kata, hapus karakter khusus."""
    if pd.isna(text) or not text:
        return []
    # Hapus karakter non-alfanumerik kecuali spasi
    text = re.sub(r'[^\w\s]', ' ', str(text))
    # Pisahkan berdasarkan spasi
    tokens = text.split()
    # Filter: hanya kata dengan panjang > 2
    return [t for t in tokens if len(t) > 2]


def tokens_to_str(tokens):
    """Ubah list token menjadi string (gabungkan dengan spasi)."""
    if not tokens:
        return ''
    return ' '.join(tokens)


def normalisasi_artist(artist_str):
    """Normalisasi nama artis untuk perbandingan."""
    if pd.isna(artist_str):
        return set()
    artist_str = str(artist_str).lower()
    # Pisahkan jika ada koma (multi artist)
    artists = re.split(r'[,;\s]+', artist_str)
    return {a.strip() for a in artists if a.strip()}


# ── 1. Data Cleaning ────────────────────────────────────────────────
def clean_data(df):
    df_clean = df.drop(columns=KOLOM_HAPUS, errors='ignore')
    
    # hapus duplikat berdasarkan Track Name
    df_clean = df_clean.drop_duplicates(subset=['Track Name'], keep='first')
    
    for col in TEXT_COLS:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('')
    
    for col in AUDIO_COLS:
        if col in df_clean.columns:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
    
    df_clean = df_clean.reset_index(drop=True)
    return df_clean


# ── 2. Case Folding & Tokenization ─────────────────────────────────
def tokenize_data(df_token):
    for col in TEXT_COLS:
        if col in df_token.columns:
            # Case folding
            df_token[col] = df_token[col].apply(case_fold)
            # Tokenization
            df_token[col] = df_token[col].apply(
                lambda x: tokens_to_str(tokenize(x))
            )
    
    # Kolom text_combined untuk TF-IDF
    df_token['genres_clean'] = df_token['Genres'].apply(
        lambda x: x.replace(',', ' ')
    )
    df_token['text_combined'] = (
        df_token['genres_clean'] + ' ' + df_token['Artist Name(s)']
    )
    
    return df_token


# ── 3. Normalisasi ──────────────────────────────────────────────────
def normalize_data(df_norm):
    scaler = MinMaxScaler()
    df_norm[AUDIO_COLS] = scaler.fit_transform(df_norm[AUDIO_COLS])
    return df_norm, scaler


# ── Fungsi Utama ───────────────────────────────────────────────
def preprocess_full():
    """
    Jalankan semua langkah preprocessing.
    
    Returns:
        dict dengan kunci:
            - 'df_clean': DataFrame setelah cleaning
            - 'df_token': DataFrame setelah tokenisasi
            - 'df_norm': DataFrame setelah normalisasi
            - 'scaler': objek MinMaxScaler
    """
    print("\n[1] MEMUAT DATA")
    print("-" * 40)
    
    # Load dataset - menggunakan sep=';' karena file CSV menggunakan semicolon
    df = pd.read_csv('Song.csv', sep=';', on_bad_lines='skip', encoding='utf-8')
    print(f"  Dataset asli : {len(df):,} baris")
    
    print("\n[2] DATA CLEANING")
    print("-" * 40)
    df_clean = clean_data(df)
    print(f"  Setelah cleaning : {len(df_clean):,} baris")
    
    # Simpan hasil cleaning
    df_clean.to_csv('hasil_01_data_cleaning.csv', index=False)
    print("  [OK] hasil_01_data_cleaning.csv")
    
    print("\n[3] CASE FOLDING & TOKENISASI")
    print("-" * 40)
    df_token = df_clean.copy()
    df_token = tokenize_data(df_token)
    print("  [OK] Case folding & tokenisasi selesai")
    
    # Simpan hasil tokenisasi
    df_token.to_csv('hasil_02_case_folding_tokenisasi.csv', index=False)
    print("  [OK] hasil_02_case_folding_tokenisasi.csv")
    
    print("\n[4] NORMALISASI")
    print("-" * 40)
    df_norm = df_token.copy()
    df_norm, scaler = normalize_data(df_norm)
    print("  [OK] Normalisasi Min-Max (fitur audio)")
    
    # Simpan hasil normalisasi
    df_norm.to_csv('hasil_03_normalisasi.csv', index=False)
    print("  [OK] hasil_03_normalisasi.csv")
    
    return {
        'df_clean': df_clean,
        'df_token': df_token,
        'df_norm': df_norm,
        'scaler': scaler,
    }