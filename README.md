# Prediksi Harga Emas (XAU/USD) — LSTM + Streamlit

Proyek untuk memprediksi harga emas harian/mingguan/bulanan menggunakan model LSTM,
dengan dashboard interaktif berbasis Streamlit.

## Cara Menjalankan

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Latih model** (opsional lewat command line, atau bisa langsung dari dashboard)
   ```bash
   python model_train.py --horizon day
   python model_train.py --horizon week
   python model_train.py --horizon month
   ```

3. **Jalankan dashboard**
   ```bash
   streamlit run app.py
   ```
   Buka browser ke `http://localhost:8501`

4. Di dashboard, kalau model belum ada, klik tombol **"Latih model sekarang"**
   di sidebar — proses training akan berjalan otomatis.

## Cara Kerja

- **Data**: diambil dari Yahoo Finance (`yfinance`), simbol `GC=F` (Gold Futures)
  sebagai proxy harga emas dunia (XAU/USD).
- **Preprocessing**: data dipakai harian apa adanya, atau diagregasi jadi
  mingguan/bulanan (`resample`), lalu dinormalisasi dengan `MinMaxScaler`.
- **Model**: LSTM 2 layer dengan dropout, menggunakan 60 periode terakhir
  (lookback) untuk memprediksi 1 periode berikutnya.
- **Dashboard**: menampilkan harga historis, harga terkini, dan hasil
  prediksi periode berikutnya, lengkap dengan grafik interaktif.

## Struktur File

```
gold-prediction/
├── data_fetch.py     # ambil data historis & harga terkini
├── model_train.py     # bangun & latih model LSTM
├── app.py              # dashboard Streamlit
├── requirements.txt
└── README.md
```

Setelah training, akan muncul file tambahan (satu set per horizon: day/week/month):
- `model_lstm_day.keras` / `model_lstm_week.keras` / `model_lstm_month.keras` — model tersimpan
- `scaler_day.pkl` / `scaler_week.pkl` / `scaler_month.pkl` — scaler untuk normalisasi
- `series_day.pkl` / `series_week.pkl` / `series_month.pkl` — data terakhir yang dipakai training

## Catatan Penting

- **"Realtime"** di sini berarti dashboard mengambil data terbaru setiap kali
  dibuka/refresh, lalu prediksi dihitung ulang — sesuai untuk horizon
  mingguan/bulanan. Ini beda dengan prediksi per-detik/menit yang butuh
  data streaming tick-by-tick.
- Kualitas prediksi LSTM untuk harga aset finansial punya keterbatasan
  inheren — harga emas dipengaruhi faktor makro (suku bunga, inflasi,
  geopolitik) yang sulit ditangkap murni dari pola historis harga.
  Gunakan sebagai alat bantu belajar/riset, bukan dasar keputusan investasi.
- Kalau ingin data lebih presisi/realtime (per menit), bisa ganti
  `data_fetch.py` untuk memakai API berbayar seperti GoldAPI.io atau
  Metals-API, lalu sesuaikan pipeline datanya.

## Ide Pengembangan Lanjutan

- Tambah fitur eksternal (DXY index, suku bunga The Fed, harga minyak)
  sebagai input tambahan ke model
- Bandingkan performa LSTM vs Prophet vs XGBoost
- Tambah backtesting untuk mengukur akurasi prediksi historis
- Deploy ke Streamlit Community Cloud biar bisa diakses online