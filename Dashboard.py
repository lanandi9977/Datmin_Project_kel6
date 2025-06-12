import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier # Import DecisionTreeClassifier

# ======================================================================================
# KONFIGURASI HALAMAN
# ======================================================================================
st.set_page_config(
    page_title="Prediksi Kepribadian | Home",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================================================
# FUNGSI-FUNGSI UTAMA (TETAP SAMA)
# ======================================================================================
@st.cache_data
def load_data():
    """Fungsi untuk memuat dan membersihkan data dari file CSV."""
    df = pd.read_csv('personality_dataset.csv')
    df_processed = df.copy()
    for col in df_processed.columns:
        if df_processed[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_processed[col]):
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
            else:
                df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
    le = LabelEncoder()
    for col in ['Stage_fear', 'Drained_after_socializing', 'Personality']:
        df_processed[col] = le.fit_transform(df_processed[col])
    return df, df_processed

@st.cache_resource
def train_models(data):
    """Fungsi untuk melatih model dan mengembalikan hasil evaluasi."""
    X = data.drop('Personality', axis=1)
    y = data['Personality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    target_names = ['Extrovert', 'Introvert']

    # --- KNN Model ---
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn, target_names=target_names)
    cm_knn = confusion_matrix(y_test, y_pred_knn)

    # --- Gaussian Naive Bayes Model ---
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    report_nb = classification_report(y_test, y_pred_nb, target_names=target_names)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    # --- Decision Tree Model (Penambahan Baru) ---
    dt = DecisionTreeClassifier(random_state=42) # Inisialisasi model Decision Tree
    dt.fit(X_train, y_train) # Latih model Decision Tree
    y_pred_dt = dt.predict(X_test) # Lakukan prediksi
    accuracy_dt = accuracy_score(y_test, y_pred_dt) # Hitung akurasi
    report_dt = classification_report(y_test, y_pred_dt, target_names=target_names) # Laporan klasifikasi
    cm_dt = confusion_matrix(y_test, y_pred_dt) # Confusion matrix

    return knn, nb, dt, accuracy_knn, report_knn, cm_knn, accuracy_nb, report_nb, cm_nb, accuracy_dt, report_dt, cm_dt # Tambahkan dt dan hasilnya

@st.cache_data
def get_average_profiles(_df_processed):
    """Menghitung profil rata-rata untuk Introvert dan Extrovert."""
    # Pastikan 'Personality' adalah kolom yang sudah di-encode (0 atau 1)
    # Asumsi 0=Extrovert, 1=Introvert berdasarkan bagaimana Anda melatih LabelEncoder
    extrovert_profile = _df_processed[_df_processed['Personality'] == 0].mean(numeric_only=True)
    introvert_profile = _df_processed[_df_processed['Personality'] == 1].mean(numeric_only=True)
    return introvert_profile.drop('Personality'), extrovert_profile.drop('Personality')


# ======================================================================================
# MEMUAT DATA DAN MODEL SEKALI SAJA MENGGUNAKAN SESSION STATE
# ======================================================================================
# Ini adalah kunci utama: data dan model hanya akan dimuat jika belum ada di session state.
if 'models_loaded' not in st.session_state:
    with st.spinner("Memuat data dan melatih model... Ini hanya butuh beberapa detik."):
        df_raw, df_processed = load_data()
        # Perbarui pemanggilan train_models untuk menyertakan Decision Tree
        model_knn, model_nb, model_dt, acc_knn, report_knn, cm_knn, acc_nb, report_nb, cm_nb, acc_dt, report_dt, cm_dt = train_models(df_processed)
        introvert_avg, extrovert_avg = get_average_profiles(df_processed)

        # Simpan semua variabel yang dibutuhkan ke session_state
        st.session_state.df_raw = df_raw
        st.session_state.df_processed = df_processed
        st.session_state.model_knn = model_knn
        st.session_state.model_nb = model_nb
        st.session_state.model_dt = model_dt # Simpan model Decision Tree
        st.session_state.acc_knn = acc_knn
        st.session_state.acc_nb = acc_nb
        st.session_state.acc_dt = acc_dt # Simpan akurasi Decision Tree
        st.session_state.report_knn = report_knn
        st.session_state.report_nb = report_nb
        st.session_state.report_dt = report_dt # Simpan laporan Decision Tree
        st.session_state.cm_knn = cm_knn
        st.session_state.cm_nb = cm_nb
        st.session_state.cm_dt = cm_dt # Simpan confusion matrix Decision Tree
        st.session_state.introvert_avg = introvert_avg
        st.session_state.extrovert_avg = extrovert_avg
        st.session_state.models_loaded = True
    st.success("Data dan model berhasil dimuat!")

# ======================================================================================
# KONTEN HALAMAN UTAMA (HOME)
# ======================================================================================
st.title("Selamat Datang di Dashboard Prediksi Kepribadian! 👋")
st.markdown("---")
st.header("Tentang Aplikasi Ini")
st.markdown("""
Aplikasi ini dirancang untuk memprediksi kecenderungan kepribadian seseorang (Introvert atau Extrovert) berdasarkan serangkaian kebiasaan dan preferensi sosial. Dasbor ini terdiri dari tiga bagian utama:

1.  **Analisis Data (EDA):** Menjelajahi dataset yang digunakan, termasuk visualisasi distribusi data dan hubungan antar fitur.
2.  **Hasil Pelatihan Model:** Menampilkan performa dari tiga model machine learning (**KNN**, **Naive Bayes**, dan **Decision Tree**) yang telah dilatih untuk melakukan prediksi.
3.  **Lakukan Prediksi:** Formulir interaktif di mana Anda dapat memasukkan data Anda sendiri dan mendapatkan prediksi kepribadian secara langsung.

Silakan pilih halaman yang ingin Anda tuju melalui navigasi di sidebar.
""")

st.info("Pilih halaman dari sidebar di sebelah kiri untuk memulai.", icon="👈")
