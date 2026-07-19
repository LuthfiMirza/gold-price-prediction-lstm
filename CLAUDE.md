# Gold Price Prediction Project

## Tujuan
Dashboard Streamlit untuk riset prediksi harga emas (XAU/USD) harian/mingguan/bulanan
menggunakan LSTM, baseline naive, evaluasi MAPE, dan perbandingan model. Data historis
& harga terkini diambil dari Yahoo Finance (`yfinance`) — gratis, tanpa API key.

## Struktur Project
- `data_fetch.py`    -> ambil data historis GC=F, harga terkini, fitur makro, dan resampling OHLC
- `backtest.py`      -> baseline naive forecaster + walk-forward MAPE
- `model_train.py`   -> bangun dataset, latih LSTM univariate/multivariate, simpan model/scaler/metadata
- `evaluate.py`      -> evaluasi LSTM vs baseline naive per horizon
- `model_compare.py` -> bandingkan naive, LSTM, XGBoost, dan Prophet
- `prediction_log.py`-> log prediksi lokal dan bandingkan dengan realisasi harga
- `app.py`           -> dashboard Streamlit
- `requirements.txt` -> daftar dependency pinned
- `README.md`        -> dokumentasi cara pakai

## Cara Menjalankan
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python model_train.py --horizon day     # atau --horizon week / --horizon month
python model_train.py --horizon day --multivariate  # opsional fitur makro
python evaluate.py
python model_compare.py
streamlit run app.py
```

## Aturan Project
- Selalu pakai virtual environment (venv), jangan install ke global Python
- Jangan hardcode API key di kode — kalau nanti tambah API berbayar, simpan key di `.env`
- Lookback default LSTM = 60 periode (`LOOKBACK` di `model_train.py`); jangan ubah tanpa retrain semua horizon
- Setiap kali `model_train.py` diubah, jalankan ulang training dan laporkan train loss vs val loss
- Model horizon `day`, `week`, dan `month` disimpan terpisah
- Jangan commit `.keras`, `.h5`, `.pkl`, `.env`, `metadata_*.json`, `prediction_log.csv`, atau folder venv

## Konvensi Kode
- Komentar & docstring pakai Bahasa Indonesia
- Gaya kode function-based, bukan class-heavy
- Setiap fungsi baru sebaiknya punya docstring singkat yang menjelaskan input/output

## Konteks Penting
- "Realtime" berarti dashboard mengambil data terbaru saat refresh/auto-refresh, bukan streaming tick-by-tick
- Auto-refresh hanya menjalankan fetch + predict; training ulang hanya lewat tombol manual atau CLI
- Prediksi adalah alat bantu riset/edukasi, bukan rekomendasi finansial — disclaimer harus selalu tampil di UI
- Baseline naive wajib ditampilkan/dipertimbangkan karena audit saat ini menunjukkan naive masih mengalahkan LSTM
- Prophet bisa gagal jika backend Stan tidak tersedia; jangan sembunyikan kegagalan environment ini

## Fitur Dashboard Saat Ini
1. Header, pilihan horizon, dan status kesegaran data
2. Kartu ringkasan harga sekarang, prediksi, rentang MAPE, dan perubahan persen
3. Grafik interaktif line/candlestick dengan prediksi, confidence band, support, dan resistance
4. Panel akurasi model naive/LSTM/XGBoost/Prophet
5. Riwayat prediksi vs realisasi
6. Warning banner saat fetch data gagal
7. Info training dan tombol "Latih ulang model sekarang"
8. Tombol unduh CSV
9. Disclaimer finansial di bagian bawah semua horizon
