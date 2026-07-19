"""Modul untuk menjalankan dashboard prediksi harga emas."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

from data_fetch import DataFetchError, fetch_historical, get_latest_price, resample_data
from model_train import LOOKBACK

HORIZON_LABELS = {
    "day": "Harian",
    "week": "Mingguan",
    "month": "Bulanan",
}
MAPE_BAND = {
    "day": 2.3519,
    "week": 4.3433,
    "month": 10.5858,
}


def _selected_horizon() -> tuple[str, str]:
    """Mengambil pilihan horizon dari sidebar."""
    st.sidebar.header("Pengaturan")
    selected_label = st.sidebar.radio("Pilih horizon", list(HORIZON_LABELS.values()))
    selected_horizon = next(key for key, value in HORIZON_LABELS.items() if value == selected_label)
    return selected_horizon, selected_label


def _artifact_path(prefix: str, horizon: str, suffix: str) -> Path:
    """Membuat path artifact model berdasarkan horizon."""
    return Path(f"{prefix}_{horizon}.{suffix}")


def _artifacts_ready(horizon: str) -> bool:
    """Memeriksa apakah model, scaler, dan series horizon sudah tersedia."""
    return all(
        path.exists()
        for path in (
            _artifact_path("model_lstm", horizon, "keras"),
            _artifact_path("scaler", horizon, "pkl"),
            _artifact_path("series", horizon, "pkl"),
        )
    )


def _load_prediction_artifacts(horizon: str):
    """Memuat model, scaler, dan series untuk prediksi dashboard."""
    model = load_model(_artifact_path("model_lstm", horizon, "keras"))
    with _artifact_path("scaler", horizon, "pkl").open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with _artifact_path("series", horizon, "pkl").open("rb") as series_file:
        series = pickle.load(series_file)
    return model, scaler, series.dropna().astype(float)


def predict_next_price(horizon: str) -> tuple[float, pd.Series]:
    """Memprediksi harga absolut periode berikutnya dari LOOKBACK data terakhir."""
    model, scaler, series = _load_prediction_artifacts(horizon)
    if len(series) < LOOKBACK:
        raise ValueError("series tidak cukup panjang untuk prediksi")

    last_window = series.iloc[-LOOKBACK:].values.reshape(-1, 1)
    scaled_window = scaler.transform(last_window)
    scaled_prediction = model.predict(scaled_window.reshape(1, LOOKBACK, 1), verbose=0)
    predicted_price = scaler.inverse_transform(scaled_prediction)[0][0]
    return float(predicted_price), series


def render_prediction_cards(current_price: float, predicted_price: float, horizon: str) -> None:
    """Menampilkan kartu harga terkini, prediksi, dan perubahan persen."""
    confidence_band = MAPE_BAND[horizon] / 100
    lower_bound = predicted_price * (1 - confidence_band)
    upper_bound = predicted_price * (1 + confidence_band)
    percent_change = ((predicted_price - current_price) / current_price) * 100

    current_col, prediction_col, change_col = st.columns(3)
    current_col.metric("Harga sekarang", f"US$ {current_price:,.2f}")
    prediction_col.metric(
        "Prediksi periode depan",
        f"US$ {predicted_price:,.2f}",
        help=f"Rentang memakai band MAPE LSTM backtest ±{MAPE_BAND[horizon]:.2f}%.",
    )
    prediction_col.caption(f"Rentang: US$ {lower_bound:,.2f} – US$ {upper_bound:,.2f}")
    change_col.metric("Perubahan prediksi", f"{percent_change:+.2f}%")



st.set_page_config(page_title="Prediksi Harga Emas", page_icon="🏆", layout="wide")

selected_horizon, selected_label = _selected_horizon()

st.title("Prediksi Harga Emas (XAU/USD)")
st.subheader(f"Horizon terpilih: {selected_label}")

try:
    current_price, latest_timestamp = get_latest_price()
    st.caption(f"Timestamp data terakhir: {latest_timestamp}")

    if not _artifacts_ready(selected_horizon):
        st.warning("Model belum dilatih untuk horizon ini. Jalankan training terlebih dahulu.")
        st.code(f"python model_train.py --horizon {selected_horizon}")
    else:
        predicted_price, _ = predict_next_price(selected_horizon)
        render_prediction_cards(current_price, predicted_price, selected_horizon)
except DataFetchError as error:
    st.warning(f"Gagal mengambil data terbaru: {error}")
except ValueError as error:
    st.warning(str(error))
