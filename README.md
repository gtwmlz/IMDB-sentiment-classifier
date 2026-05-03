# Analisis Sentimen Ulasan Film (IMDB)

Proyek ini melakukan **klasifikasi sentimen biner** (positif / negatif) pada teks ulasan film. Pelatihan model dan evaluasi dilakukan di Jupyter Notebook; aplikasi web sederhana memuat model terlatih untuk prediksi interaktif dan batch melalui CSV.

## Isi proyek

| Bagian | Penjelasan |
|--------|------------|
| **Notebook (`mv.ipynb`)** | Pembersihan teks, pembagian data, vektorisasi **TF-IDF**, pelatihan **Logistic Regression** dan **Naive Bayes**, metrik (`accuracy`, `balanced accuracy`, `classification_report`), serta **confusion matrix** lewat fungsi `evaluate_model`. |
| **Aplikasi (`app.py`)** | UI **Streamlit**: prediksi satu teks (dengan probabilitas) dan unggah CSV berisi kolom `review` untuk prediksi massal + diagram pie distribusi sentimen. Saat ini aplikasi memakai model **Logistic Regression** (`sentiment_log_model.pkl`). |
| **Data** | `IMDB Dataset.csv` — ~50.000 ulasan (kolom `review`, `sentiment`). File ini besar dan biasanya tidak di-commit (lihat `.gitignore`). |
| **Artefak model** | `sentiment_log_model.pkl`, `sentiment_nb_model.pkl`, `tfidf_vectorizer.pkl` — dihasilkan setelah menjalankan notebook; juga diabaikan Git agar repo tetap ringan. |

## Struktur file (sementara)

```
str/
├── app.py                    # Aplikasi Streamlit
├── mv.ipynb                  # Pipeline ML & evaluasi
├── IMDB Dataset.csv          # Dataset (lokal; di .gitignore)
├── sentiment_log_model.pkl   # Model LR (lokal; di .gitignore)
├── sentiment_nb_model.pkl    # Model Naive Bayes (lokal; di .gitignore)
├── tfidf_vectorizer.pkl      # Vectorizer TF-IDF (lokal; di .gitignore)
├── .gitignore
└── README.md
```

Folder `.venv/` dipakai untuk lingkungan virtual Python lokal dan tidak perlu di-versioning.

## Stack teknologi

- **Bahasa:** Python 3.x  
- **ML / data:** [scikit-learn](https://scikit-learn.org/) (TF-IDF, Logistic Regression, MultinomialNB), [pandas](https://pandas.pydata.org/)  
- **NLP:** [NLTK](https://www.nltk.org/) (stopwords)  
- **Serialisasi model:** [joblib](https://joblib.readthedocs.io/)  
- **Notebook:** Jupyter  
- **Visualisasi (notebook & app):** [Matplotlib](https://matplotlib.org/)  
- **Web UI:** [Streamlit](https://streamlit.io/)  

> **Catatan:** Belum ada `requirements.txt` di repo. Untuk menyiapkan lingkungan, instal paket yang dipakai di notebook dan di `app.py`, misalnya: `streamlit`, `pandas`, `scikit-learn`, `joblib`, `matplotlib`, `jupyter`, `nltk`.

## Menjalankan proyek

1. **Lingkungan virtual (disarankan)**  
   Buat dan aktifkan venv, lalu instal dependensi yang diperlukan.

2. **Latih ulang / hasilkan artefak**  
   Buka `mv.ipynb`, jalankan sel dari awal sampai penyimpanan `joblib` agar file `.pkl` dan vectorizer tersedia di folder proyek.

3. **Jalankan aplikasi web**  
   Dari root proyek (pastikan file model & vectorizer ada):

   ```bash
   streamlit run app.py
   ```

4. **Dataset**  
   Letakkan `IMDB Dataset.csv` di folder yang sama dengan notebook jika Anda ingin mengulang pelatihan dari data mentah.

## Rencana ke depan (perbaikan UI)

Hal-hal yang masuk akal untuk iterasi berikutnya pada antarmuka Streamlit:

- **Tampilan:** layout dua kolom (input vs hasil), tema/warna konsisten, tipografi dan spacing yang lebih rapi; opsi dark mode jika diperlukan.  
- **UX:** validasi input kosong dengan pesan yang jelas, indikator loading saat prediksi batch, pratinjau beberapa baris CSV setelah upload.  
- **Fitur:** pilihan model (Logistic Regression vs Naive Bayes) lewat `st.selectbox`, tampilan probabilitas untuk kedua kelas, unduh CSV hasil prediksi.  
- **Grafik:** ganti atau lengkapi pie chart (misalnya bar chart atau metrik ringkas) agar lebih mudah dibaca pada dataset tidak seimbang.  
- **Deployment:** persiapkan `requirements.txt` / Docker ringan dan variabel lingkungan untuk path model agar siap di-hosting.

---

Jika Anda menambahkan file baru (misalnya `requirements.txt` atau folder `assets/`), struktur di atas bisa diperbarui di bagian README ini.
