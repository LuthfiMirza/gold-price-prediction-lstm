"""Modul untuk menguji baseline prediksi harga emas secara walk-forward."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


PredictFn = Callable[[pd.Series], float]


def naive_forecast(series: pd.Series) -> float:
    """Memprediksi nilai berikutnya memakai nilai terakhir yang tersedia."""
    clean_series = series.dropna()
    if clean_series.empty:
        raise ValueError("series tidak boleh kosong")
    return float(clean_series.iloc[-1])


def walk_forward_mape(series: pd.Series, predict_fn: PredictFn, window: int) -> float:
    """Menghitung MAPE dengan validasi walk-forward pada jendela bergerak."""
    if window <= 0:
        raise ValueError("window harus lebih besar dari 0")

    clean_series = series.dropna().astype(float)
    if len(clean_series) <= window:
        raise ValueError("panjang series harus lebih besar dari window")

    percentage_errors = []
    for current_index in range(window, len(clean_series)):
        history_window = clean_series.iloc[current_index - window : current_index]
        actual_value = float(clean_series.iloc[current_index])
        if actual_value == 0:
            continue

        predicted_value = float(predict_fn(history_window))
        percentage_error = abs((actual_value - predicted_value) / actual_value)
        percentage_errors.append(percentage_error)

    if not percentage_errors:
        raise ValueError("MAPE tidak bisa dihitung karena semua nilai aktual bernilai 0")

    return float(np.mean(percentage_errors) * 100)
