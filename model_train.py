"""Modul untuk melatih model prediksi harga emas."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from data_fetch import fetch_historical, resample_data

LOOKBACK = 60
TRAIN_SPLIT_RATIO = 0.8
EPOCHS = 20
BATCH_SIZE = 32
VALID_HORIZONS = ("day", "week", "month")
TRAIN_PERIOD_BY_HORIZON = {
    "day": "10y",
    "week": "10y",
    "month": "20y",
}


# LOOKBACK menentukan bentuk input model tersimpan; jangan diubah tanpa retrain semua horizon.
def build_dataset(series: pd.Series | np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Membentuk dataset supervised X dan y dari deret harga tertutup."""
    if lookback <= 0:
        raise ValueError("lookback harus lebih besar dari 0")

    values = np.asarray(series, dtype=float).reshape(-1, 1)
    if len(values) <= lookback:
        raise ValueError("panjang series harus lebih besar dari lookback")

    features = []
    targets = []
    for current_index in range(lookback, len(values)):
        features.append(values[current_index - lookback : current_index])
        targets.append(values[current_index])

    return np.array(features), np.array(targets)


def build_model() -> Sequential:
    """Membangun model LSTM dua layer dengan dropout untuk regresi harga."""
    model = Sequential(
        [
            Input(shape=(LOOKBACK, 1)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _artifact_path(prefix: str, horizon: str, suffix: str) -> Path:
    """Membuat path artifact training berdasarkan horizon."""
    return Path(f"{prefix}_{horizon}.{suffix}")


def _prepare_scaled_series(close_series: pd.Series) -> tuple[np.ndarray, MinMaxScaler, int]:
    """Melakukan scaling dengan scaler yang fit hanya pada train split."""
    values = close_series.dropna().astype(float).values.reshape(-1, 1)
    train_size = int(len(values) * TRAIN_SPLIT_RATIO)
    if train_size <= LOOKBACK or len(values) - train_size < 1:
        raise ValueError("data tidak cukup untuk train/validation split")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values[:train_size])
    scaled_values = scaler.transform(values)
    return scaled_values, scaler, train_size


def train(horizon: str):
    """Melatih model LSTM yang memprediksi harga absolut periode berikutnya."""
    normalized_horizon = horizon.lower()
    if normalized_horizon not in VALID_HORIZONS:
        valid_values = ", ".join(VALID_HORIZONS)
        raise ValueError(f"horizon harus salah satu dari: {valid_values}")

    historical_df = fetch_historical(period=TRAIN_PERIOD_BY_HORIZON[normalized_horizon])
    horizon_df = resample_data(historical_df, normalized_horizon)
    close_series = horizon_df["Close"]

    scaled_values, scaler, train_size = _prepare_scaled_series(close_series)
    train_values = scaled_values[:train_size]
    validation_values = scaled_values[train_size - LOOKBACK :]

    x_train, y_train = build_dataset(train_values, LOOKBACK)
    x_val, y_val = build_dataset(validation_values, LOOKBACK)

    model = build_model()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    model.save(_artifact_path("model_lstm", normalized_horizon, "keras"))
    with _artifact_path("scaler", normalized_horizon, "pkl").open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with _artifact_path("series", normalized_horizon, "pkl").open("wb") as series_file:
        pickle.dump(close_series, series_file)

    return history


def parse_args() -> argparse.Namespace:
    """Membaca argumen CLI untuk memilih horizon training."""
    parser = argparse.ArgumentParser(description="Latih model LSTM prediksi harga emas.")
    parser.add_argument("--horizon", choices=VALID_HORIZONS, required=True)
    return parser.parse_args()


def main() -> None:
    """Menjalankan training dari command line."""
    args = parse_args()
    train(args.horizon)


if __name__ == "__main__":
    main()
