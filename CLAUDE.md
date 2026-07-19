# Gold Price Prediction Project

## Tujuan
Dashboard Streamlit untuk prediksi harga emas (XAU/USD) harian/mingguan/bulanan
menggunakan model LSTM. Data historis & harga terkini diambil dari
Yahoo Finance (yfinance) — gratis, tanpa API key.

## Struktur Project
- `data_fetch.py`    -> ambil data historis & harga terkini (yfinance, simbol GC=F)
- `model_train.py`   -> bangun dataset, latih LSTM, simpan model + scaler
- `app.py`           -> dashboard Streamlit (chart harga + prediksi)
- `requirements.txt` -> daftar dependency
- `README.md`        -> dokumentasi cara pakai

File hasil training (dibuat otomatis, jangan diedit manual). Satu set per horizon
(day / week / month), format .keras (bukan .h5 yang sudah deprecated di Keras 3):
- `model_lstm_day.keras`, `model_lstm_week.keras`, `model_lstm_month.keras` -> model tersimpan
- `scaler_day.pkl`, `scaler_week.pkl`, `scaler_month.pkl`                    -> scaler normalisasi
- `series_day.pkl`, `series_week.pkl`, `series_month.pkl`                    -> data terakhir dipakai training

## Cara Menjalankan
```bash
pip install -r requirements.txt
python model_train.py --horizon day     # atau --horizon week / --horizon month
streamlit run app.py
```

## Aturan Project
- Selalu pakai virtual environment (venv), jangan install ke global Python
- Jangan hardcode API key di kode — kalau nanti tambah API berbayar (GoldAPI dll),
  simpan key di `.env` dan load pakai `python-dotenv`
- Lookback default LSTM = 60 periode (didefinisikan di `model_train.py` sebagai
  `LOOKBACK`) — jangan diubah tanpa alasan jelas, karena mempengaruhi ukuran
  input model yang sudah tersimpan
- Setiap kali `model_train.py` diubah, jalankan ulang training dan laporkan
  perbandingan train loss vs val loss (cek overfitting) sebelum dianggap selesai
- Model horizon "day", "week", dan "month" disimpan terpisah — jangan dicampur
- Jangan commit file `.keras`, `.h5`, `.pkl`, `.env`, atau folder `venv/` ke git (lihat .gitignore)

## Konvensi Kode
- Komentar & docstring pakai Bahasa Indonesia
- Gaya kode function-based (bukan class-based), konsisten dengan file yang sudah ada
- Setiap fungsi baru sebaiknya punya docstring singkat yang menjelaskan input/output

## Konteks Penting
- "Realtime" di project ini artinya dashboard mengambil data terbaru setiap
  kali dibuka/refresh (bukan streaming tick-by-tick), karena horizon prediksi
  mingguan/bulanan tidak butuh update per detik
- Prediksi adalah alat bantu riset/edukasi, bukan rekomendasi finansial —
  selalu pertahankan disclaimer ini di UI dashboard
- Kalau menambah sumber data baru, pastikan tetap kompatibel dengan format
  yang dipakai `resample_data()` di `data_fetch.py` (index datetime, kolom harga)

## Fitur Dashboard (Rencana Final)

Urutan tampilan di `app.py`, dari atas ke bawah:

1. **Header** — judul, pilihan horizon (day/week/month), status kesegaran data
   (timestamp data terakhir diambil + indikator warna hijau/kuning/merah)
2. **Kartu ringkasan** — harga sekarang, prediksi periode depan (dengan rentang
   interval kepercayaan, bukan angka tunggal), persentase perubahan
3. **Grafik interaktif** — harga historis + titik/pita prediksi, opsi tampilan
   garis atau candlestick
4. **Panel akurasi model** — MAPE model vs MAPE baseline naive (dan XGBoost
   kalau sudah dibuat) dari hasil backtesting, supaya user tahu model ini
   benar-benar lebih baik dari sekadar "tebak harga terakhir" atau tidak
5. **Riwayat prediksi vs realisasi** — log prediksi yang pernah dibuat model,
   dibandingkan dengan harga aktual setelah waktunya lewat (jejak rekam nyata,
   beda dari backtesting yang pakai data lama)
6. **Warning banner** — kalau fetch data dari yfinance gagal/kosong, tampilkan
   peringatan jelas dan jangan diam-diam pakai data basi
7. **Info training** — kapan model terakhir dilatih + tombol "Latih ulang
   model sekarang"
8. **Level support/resistance sederhana** — dari data historis, sebagai konteks
   tambahan
9. **Tombol unduh CSV** — hasil prediksi + data historis
10. **Disclaimer** — selalu tampil di bagian bawah (lihat Konteks Penting)

Auto-refresh dashboard pakai `streamlit-autorefresh`, interval disarankan
5-15 menit (bukan per detik) — training model TIDAK dijalankan ulang tiap
refresh, hanya load model yang sudah ada dan hitung prediksi baru dari data
terkini. Training ulang hanya lewat tombol manual atau dijadwalkan berkala.

## Ide Pengembangan yang Sedang Dipertimbangkan
- Backtesting akurasi (MAPE) terhadap data historis — WAJIB, jadi bagian dari
  Fitur Dashboard poin 4 di atas
- Bandingkan LSTM vs Prophet vs XGBoost
- Tambah fitur eksternal (DXY index, suku bunga, harga minyak) sebagai input model
- Deploy ke Streamlit Community Cloud
