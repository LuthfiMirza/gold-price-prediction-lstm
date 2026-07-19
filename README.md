# Prediksi Harga Emas (XAU/USD) — LSTM + Streamlit

Proyek portfolio untuk riset prediksi harga emas harian/mingguan/bulanan menggunakan LSTM,
baseline naive, evaluasi MAPE, dan dashboard Streamlit. Data diambil dari Yahoo Finance
melalui `yfinance` dengan simbol utama `GC=F`.

## Cara Menjalankan

1. **Buat dan aktifkan virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Latih model per horizon**
   ```bash
   python model_train.py --horizon day
   python model_train.py --horizon week
   python model_train.py --horizon month
   ```

4. **Opsional: latih model multivariate**
   ```bash
   python model_train.py --horizon day --multivariate
   python model_train.py --horizon week --multivariate
   python model_train.py --horizon month --multivariate
   ```

5. **Evaluasi model**
   ```bash
   python evaluate.py
   python model_compare.py
   ```

6. **Jalankan dashboard**
   ```bash
   streamlit run app.py
   ```
   Buka browser ke `http://localhost:8501`.

## Struktur File

```text
gold-prediction/
├── data_fetch.py       # ambil data emas, resampling OHLC, dan fitur makro
├── backtest.py         # baseline naive + walk-forward MAPE
├── model_train.py      # training LSTM univariate/multivariate + metadata
├── evaluate.py         # evaluasi LSTM vs baseline naive
├── model_compare.py    # perbandingan naive, LSTM, XGBoost, Prophet
├── prediction_log.py   # log prediksi vs realisasi harga
├── app.py              # dashboard Streamlit
├── requirements.txt
├── PLAN.md
└── CLAUDE.md
```

## Artifact Lokal

File berikut dibuat saat training/runtime dan tidak dicommit:
- `model_lstm_*.keras` dan `model_lstm_multi_*.keras`
- `scaler_*.pkl` dan `scaler_multi_*.pkl`
- `series_*.pkl` dan `series_multi_*.pkl`
- `metadata_*.json`
- `prediction_log.csv`

## Fitur Dashboard

- Sidebar horizon Harian/Mingguan/Bulanan.
- Harga terkini dan status kesegaran data.
- Warning banner saat fetch Yahoo Finance gagal.
- Kartu ringkasan harga sekarang, prediksi periode depan, rentang MAPE, dan perubahan persen.
- Grafik Plotly line/candlestick dengan titik prediksi, confidence band, support, dan resistance.
- CSV export data historis + prediksi.
- Riwayat prediksi vs harga realisasi.
- Tombol retrain manual dan metadata training.
- Auto-refresh 10 menit untuk fetch + predict, bukan retraining otomatis.
- Panel perbandingan model naive/LSTM/XGBoost/Prophet.
- Disclaimer finansial selalu tampil di bagian bawah dashboard.

## Catatan Evaluasi Saat Ini

Audit awal menunjukkan baseline naive masih mengalahkan LSTM pada semua horizon.
Artinya project ini valid sebagai alat belajar/riset, tetapi model belum layak dianggap
unggul untuk keputusan finansial.

## Disclaimer

Prediksi ini hanya untuk riset dan edukasi, bukan rekomendasi finansial atau ajakan
membeli/menjual aset apa pun.
