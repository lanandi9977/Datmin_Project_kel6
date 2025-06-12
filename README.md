# Dasbor Prediksi Kepribadian (Introvert vs. Extrovert)

## ✨ Fitur Utama

Aplikasi ini dibagi menjadi tiga halaman utama dengan berbagai fitur:

* **🏠 Halaman Utama:** Halaman selamat datang yang menjelaskan fungsionalitas aplikasi.
* **📊 Analisis Data (EDA):**
    * Menampilkan dataset mentah dan statistik deskriptif.
    * Visualisasi interaktif distribusi data menggunakan **Plotly**, termasuk diagram lingkaran dan diagram batang.
    * Peta korelasi (heatmap) menggunakan **Seaborn** untuk melihat hubungan antar fitur.
* **🤖 Hasil Pelatihan Model:**
    * Membandingkan performa dua model klasifikasi: **K-Nearest Neighbors (KNN)** dan **Gaussian Naive Bayes**.
    * Menampilkan metrik evaluasi lengkap, termasuk Akurasi, Laporan Klasifikasi (Presisi, Recall, F1-Score), dan **Confusion Matrix** visual.
* **🔮 Lakukan Prediksi:**
    * Formulir interaktif bagi pengguna untuk memasukkan data mereka sendiri.
    * Menampilkan hasil prediksi dari kedua model beserta tingkat kepercayaannya (probabilitas).
    * **Radar Chart** yang membandingkan profil pengguna dengan profil rata-rata Introvert dan Extrovert untuk wawasan yang lebih dalam.


## 🛠️ Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3
* **Framework Web:** Streamlit
* **Analisis Data:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Visualisasi Data:** Plotly, Matplotlib, Seaborn

---

## 🚀 Instalasi dan Cara Menjalankan

Untuk menjalankan aplikasi ini di komputer lokal Anda, ikuti langkah-langkah berikut:

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-anda/nama-repositori-anda.git](https://github.com/username-anda/nama-repositori-anda.git)
    ```

2.  **Masuk ke direktori proyek:**
    ```bash
    cd nama-repositori-anda
    ```

3.  **Buat dan aktifkan virtual environment:**
    ```bash
    # Membuat environment
    python -m venv venv

    # Mengaktifkan di Windows
    .\venv\Scripts\activate

    # Mengaktifkan di MacOS/Linux
    source venv/bin/activate
    ```

4.  **Instal semua library yang dibutuhkan:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Jalankan aplikasi Streamlit:**
    ```bash
    streamlit run app.py
    ```

Aplikasi akan otomatis terbuka di browser Anda.

---

## 📁 Struktur Proyek

Struktur file dan folder proyek ini diatur sebagai berikut untuk mendukung aplikasi multi-halaman Streamlit.
