"""Modul untuk mengevaluasi performa model LSTM terhadap baseline naive."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from backtest import naive_forecast, walk_forward_mape
from model_train import LOOKBACK, VALID_HORIZONS


def _artifact_path(prefix: str, horizon: str, suffix: str, multivariate: bool = False) -> Path:
    """Membuat path artifact model berdasarkan horizon."""
    mode = "multi_" if multivariate else ""
    return Path(f"{prefix}_{mode}{horizon}.{suffix}")


def _load_series(horizon: str, multivariate: bool = False):
    """Memuat series harga yang disimpan saat training."""
    with _artifact_path("series", horizon, "pkl", multivariate).open("rb") as series_file:
        return pickle.load(series_file)


def _load_scaler(horizon: str, multivariate: bool = False):
    """Memuat scaler yang dipakai saat training."""
    with _artifact_path("scaler", horizon, "pkl", multivariate).open("rb") as scaler_file:
        return pickle.load(scaler_file)


def lstm_walk_forward_mape(series: pd.Series, horizon: str, window: int) -> float:
    """Menghitung MAPE walk-forward LSTM dengan prediksi batch agar efisien."""
    clean_series = series.dropna().astype(float)
    if len(clean_series) <= window:
        raise ValueError("panjang series harus lebih besar dari window")

    model = load_model(_artifact_path("model_lstm", horizon, "keras"))
    scaler = _load_scaler(horizon)
    values = clean_series.values.reshape(-1, 1)
    scaled_values = scaler.transform(values)

    features = []
    actual_values = []
    for current_index in range(window, len(scaled_values)):
        actual_value = float(values[current_index][0])
        if actual_value == 0:
            continue
        features.append(scaled_values[current_index - window : current_index])
        actual_values.append(actual_value)

    if not features:
        raise ValueError("MAPE tidak bisa dihitung karena semua nilai aktual bernilai 0")

    scaled_predictions = model.predict(np.array(features), verbose=0)
    predictions = scaler.inverse_transform(scaled_predictions).reshape(-1)
    actual_array = np.asarray(actual_values, dtype=float)
    return float(np.mean(np.abs((actual_array - predictions) / actual_array)) * 100)


def multivariate_lstm_walk_forward_mape(feature_df: pd.DataFrame, horizon: str, window: int) -> float:
    """Menghitung MAPE walk-forward LSTM multivariate dengan prediksi batch."""
    clean_df = feature_df.dropna().astype(float)
    if len(clean_df) <= window:
        raise ValueError("panjang feature_df harus lebih besar dari window")

    model = load_model(_artifact_path("model_lstm", horizon, "keras", True))
    scaler = _load_scaler(horizon, True)
    values = clean_df.values
    scaled_values = scaler.transform(values)

    features = []
    actual_values = []
    for current_index in range(window, len(scaled_values)):
        actual_value = float(values[current_index, 0])
        if actual_value == 0:
            continue
        features.append(scaled_values[current_index - window : current_index])
        actual_values.append(actual_value)

    scaled_predictions = model.predict(np.array(features), verbose=0).reshape(-1)
    predictions = scaled_predictions * scaler.data_range_[0] + scaler.data_min_[0]
    actual_array = np.asarray(actual_values, dtype=float)
    return float(np.mean(np.abs((actual_array - predictions) / actual_array)) * 100)


def evaluate_horizon(horizon: str) -> dict[str, float | str]:
    """Menghitung MAPE LSTM dan baseline naive untuk satu horizon."""
    normalized_horizon = horizon.lower()
    if normalized_horizon not in VALID_HORIZONS:
        valid_values = ", ".join(VALID_HORIZONS)
        raise ValueError(f"horizon harus salah satu dari: {valid_values}")

    series = _load_series(normalized_horizon).dropna().astype(float)
    lstm_mape = lstm_walk_forward_mape(series, normalized_horizon, LOOKBACK)
    naive_mape = walk_forward_mape(series, naive_forecast, LOOKBACK)
    winner = "LSTM" if lstm_mape < naive_mape else "naive"

    return {
        "horizon": normalized_horizon,
        "lstm_mape": lstm_mape,
        "naive_mape": naive_mape,
        "winner": winner,
    }


def print_evaluation_table(results: list[dict[str, float | str]]) -> None:
    """Mencetak tabel ringkas hasil evaluasi model."""
    print("horizon | LSTM MAPE | naive MAPE | winner")
    print("--- | ---: | ---: | ---")
    for result in results:
        print(
            f"{result['horizon']} | "
            f"{result['lstm_mape']:.4f} | "
            f"{result['naive_mape']:.4f} | "
            f"{result['winner']}"
        )


def evaluate_multivariate_comparison() -> list[dict[str, float | str]]:
    """Membandingkan MAPE LSTM univariate dan multivariate per horizon."""
    rows = []
    for horizon in VALID_HORIZONS:
        univariate_series = _load_series(horizon).dropna().astype(float)
        multivariate_frame = _load_series(horizon, True)
        univariate_mape = lstm_walk_forward_mape(univariate_series, horizon, LOOKBACK)
        multivariate_mape = multivariate_lstm_walk_forward_mape(multivariate_frame, horizon, LOOKBACK)
        winner = "multivariate" if multivariate_mape < univariate_mape else "univariate"
        rows.append(
            {
                "horizon": horizon,
                "univariate_mape": univariate_mape,
                "multivariate_mape": multivariate_mape,
                "winner": winner,
            }
        )
    return rows


def print_multivariate_table(results: list[dict[str, float | str]]) -> None:
    """Mencetak tabel perbandingan univariate dan multivariate."""
    print("horizon | univariate LSTM MAPE | multivariate LSTM MAPE | winner")
    print("--- | ---: | ---: | ---")
    for result in results:
        print(
            f"{result['horizon']} | "
            f"{result['univariate_mape']:.4f} | "
            f"{result['multivariate_mape']:.4f} | "
            f"{result['winner']}"
        )


def main() -> None:
    """Menjalankan evaluasi untuk semua horizon."""
    results = [evaluate_horizon(horizon) for horizon in VALID_HORIZONS]
    print_evaluation_table(results)
    if all(_artifact_path("model_lstm", horizon, "keras", True).exists() for horizon in VALID_HORIZONS):
        print()
        print_multivariate_table(evaluate_multivariate_comparison())


if __name__ == "__main__":
    main()
