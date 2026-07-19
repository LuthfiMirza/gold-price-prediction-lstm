"""Modul untuk melatih model prediksi harga emas."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential

from data_fetch import fetch_historical, fetch_multivariate_data, resample_data

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


def build_feature_dataset(values: np.ndarray, lookback: int, target_column: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Membentuk dataset supervised multivariate dengan target satu kolom."""
    if lookback <= 0:
        raise ValueError("lookback harus lebih besar dari 0")
    if len(values) <= lookback:
        raise ValueError("panjang values harus lebih besar dari lookback")

    features = []
    targets = []
    for current_index in range(lookback, len(values)):
        features.append(values[current_index - lookback : current_index])
        targets.append(values[current_index, target_column])

    return np.array(features), np.array(targets)


def build_model(feature_count: int = 1) -> Sequential:
    """Membangun model LSTM dua layer dengan dropout untuk regresi harga."""
    model = Sequential(
        [
            Input(shape=(LOOKBACK, feature_count)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def _artifact_path(prefix: str, horizon: str, suffix: str, multivariate: bool = False) -> Path:
    """Membuat path artifact training berdasarkan horizon."""
    mode = "multi_" if multivariate else ""
    return Path(f"{prefix}_{mode}{horizon}.{suffix}")


def compute_returns(values: np.ndarray) -> np.ndarray:
    """Menghitung return persentase antar periode berturut-turut dari deret harga/fitur."""
    prices = np.asarray(values, dtype=float)
    if len(prices) < 2:
        raise ValueError("butuh minimal 2 titik data untuk menghitung return")
    return (prices[1:] - prices[:-1]) / prices[:-1]


def _validation_mape(
    model: Sequential,
    scaler: MinMaxScaler,
    x_val: np.ndarray,
    base_prices_val: np.ndarray,
    actual_next_prices_val: np.ndarray,
) -> float:
    """Menghitung MAPE validasi dalam skala harga asli dari prediksi return."""
    scaled_predictions = model.predict(x_val, verbose=0).reshape(-1)
    target_min = scaler.data_min_[0]
    target_range = scaler.data_range_[0]
    predicted_returns = scaled_predictions * target_range + target_min
    predicted_prices = base_prices_val * (1 + predicted_returns)
    non_zero_mask = actual_next_prices_val != 0
    actual_values = actual_next_prices_val[non_zero_mask]
    predictions = predicted_prices[non_zero_mask]
    return float(np.mean(np.abs((actual_values - predictions) / actual_values)) * 100)


def _prepare_scaled_series(close_series: pd.Series) -> tuple[np.ndarray, MinMaxScaler, int, np.ndarray]:
    """Mengubah harga jadi return, lalu scaling dengan scaler yang fit hanya pada train split."""
    prices = close_series.dropna().astype(float).values
    returns = compute_returns(prices).reshape(-1, 1)
    train_size = int(len(returns) * TRAIN_SPLIT_RATIO)
    if train_size <= LOOKBACK or len(returns) - train_size < 1:
        raise ValueError("data tidak cukup untuk train/validation split")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(returns[:train_size])
    scaled_returns = scaler.transform(returns)
    return scaled_returns, scaler, train_size, prices


def _prepare_scaled_frame(feature_df: pd.DataFrame) -> tuple[np.ndarray, MinMaxScaler, int, np.ndarray]:
    """Mengubah semua kolom fitur jadi return, lalu scaling dengan scaler fit hanya pada train split."""
    values = feature_df.dropna().astype(float).values
    returns = compute_returns(values)
    train_size = int(len(returns) * TRAIN_SPLIT_RATIO)
    if train_size <= LOOKBACK or len(returns) - train_size < 1:
        raise ValueError("data multivariate tidak cukup untuk train/validation split")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(returns[:train_size])
    scaled_returns = scaler.transform(returns)
    return scaled_returns, scaler, train_size, values[:, 0]


def train(horizon: str, multivariate: bool = False):
    """Melatih model LSTM yang memprediksi return (persentase perubahan) periode berikutnya.

    Target return dipilih daripada harga absolut karena LSTM harga absolut mudah
    "menjiplak" nilai terakhir (mirip baseline naive) dan sulit mengalahkannya pada
    backtest MAPE. Prediksi harga akhir tetap direkonstruksi dari harga asli terakhir
    dikali (1 + return prediksi), jadi API luar (app.py, evaluate.py) tetap bicara
    dalam skala harga, bukan return.
    """
    normalized_horizon = horizon.lower()
    if normalized_horizon not in VALID_HORIZONS:
        valid_values = ", ".join(VALID_HORIZONS)
        raise ValueError(f"horizon harus salah satu dari: {valid_values}")

    if multivariate:
        feature_df = fetch_multivariate_data(period=TRAIN_PERIOD_BY_HORIZON[normalized_horizon])
        feature_df = feature_df.resample({"day": "B", "week": "W-FRI", "month": "ME"}[normalized_horizon]).last().ffill().dropna()
        close_series = feature_df["GoldClose"]
        scaled_returns, scaler, train_size, target_prices = _prepare_scaled_frame(feature_df)
        train_values = scaled_returns[:train_size]
        validation_values = scaled_returns[train_size - LOOKBACK :]
        x_train, y_train = build_feature_dataset(train_values, LOOKBACK)
        x_val, y_val = build_feature_dataset(validation_values, LOOKBACK)
        feature_count = feature_df.shape[1]
    else:
        historical_df = fetch_historical(period=TRAIN_PERIOD_BY_HORIZON[normalized_horizon])
        horizon_df = resample_data(historical_df, normalized_horizon)
        close_series = horizon_df["Close"]
        scaled_returns, scaler, train_size, target_prices = _prepare_scaled_series(close_series)
        train_values = scaled_returns[:train_size]
        validation_values = scaled_returns[train_size - LOOKBACK :]
        x_train, y_train = build_dataset(train_values, LOOKBACK)
        x_val, y_val = build_dataset(validation_values, LOOKBACK)
        feature_count = 1

    # target_prices[train_size:] adalah harga dasar (t) untuk tiap sampel validasi,
    # target_prices[train_size+1:] adalah harga aktual periode berikutnya (t+1) —
    # dipakai merekonstruksi prediksi return kembali ke skala harga untuk MAPE.
    base_prices_val = target_prices[train_size : train_size + len(y_val)]
    actual_next_prices_val = target_prices[train_size + 1 : train_size + 1 + len(y_val)]

    model = build_model(feature_count)
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
    )

    model.save(_artifact_path("model_lstm", normalized_horizon, "keras", multivariate))
    with _artifact_path("scaler", normalized_horizon, "pkl", multivariate).open("wb") as scaler_file:
        pickle.dump(scaler, scaler_file)
    with _artifact_path("series", normalized_horizon, "pkl", multivariate).open("wb") as series_file:
        pickle.dump(feature_df if multivariate else close_series, series_file)

    metadata = {
        "horizon": normalized_horizon,
        "multivariate": multivariate,
        "prediction_target": "return",
        "last_trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
        "validation_mape": _validation_mape(model, scaler, x_val, base_prices_val, actual_next_prices_val),
        "lookback": LOOKBACK,
        "train_period": TRAIN_PERIOD_BY_HORIZON[normalized_horizon],
    }
    _artifact_path("metadata", normalized_horizon, "json", multivariate).write_text(json.dumps(metadata, indent=2))

    return history


def parse_args() -> argparse.Namespace:
    """Membaca argumen CLI untuk memilih horizon training."""
    parser = argparse.ArgumentParser(description="Latih model LSTM prediksi harga emas.")
    parser.add_argument("--horizon", choices=VALID_HORIZONS, required=True)
    parser.add_argument("--multivariate", action="store_true", help="Gunakan fitur DXY, proxy Fed rate, dan oil.")
    return parser.parse_args()


def main() -> None:
    """Menjalankan training dari command line."""
    args = parse_args()
    train(args.horizon, multivariate=args.multivariate)


if __name__ == "__main__":
    main()
