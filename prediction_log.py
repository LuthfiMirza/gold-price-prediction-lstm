"""Modul untuk mencatat prediksi dan membandingkannya dengan realisasi."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_fetch import fetch_historical, resample_data

LOG_PATH = Path("prediction_log.csv")
LOG_COLUMNS = ["logged_at", "horizon", "target_date", "predicted_value"]


def _read_log() -> pd.DataFrame:
    """Membaca log prediksi lokal atau membuat tabel kosong."""
    if not LOG_PATH.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(LOG_PATH, parse_dates=["logged_at", "target_date"])


def log_prediction(horizon: str, predicted_value: float, target_date) -> None:
    """Mencatat prediksi baru dengan dedup berdasarkan horizon dan target_date."""
    target_timestamp = pd.Timestamp(target_date).normalize()
    log_df = _read_log()
    if not log_df.empty:
        same_prediction = (log_df["horizon"] == horizon) & (log_df["target_date"] == target_timestamp)
        if same_prediction.any():
            return

    new_row = pd.DataFrame(
        [
            {
                "logged_at": pd.Timestamp.now(tz="UTC"),
                "horizon": horizon,
                "target_date": target_timestamp,
                "predicted_value": float(predicted_value),
            }
        ]
    )
    pd.concat([log_df, new_row], ignore_index=True).to_csv(LOG_PATH, index=False)


def get_realized_comparisons(horizon: str) -> pd.DataFrame:
    """Mengambil prediksi yang sudah lewat dan menghitung error realisasi."""
    log_df = _read_log()
    if log_df.empty:
        return pd.DataFrame(columns=["target_date", "predicted_value", "actual_value", "error_pct"])

    horizon_log = log_df[log_df["horizon"] == horizon].copy()
    if horizon_log.empty:
        return pd.DataFrame(columns=["target_date", "predicted_value", "actual_value", "error_pct"])

    actual_df = resample_data(fetch_historical(period="20y"), horizon)
    latest_date = actual_df.index.max().normalize()
    horizon_log = horizon_log[horizon_log["target_date"] <= latest_date]

    rows = []
    for _, prediction in horizon_log.iterrows():
        target_date = pd.Timestamp(prediction["target_date"]).normalize()
        actual_candidates = actual_df[actual_df.index.normalize() >= target_date]
        if actual_candidates.empty:
            continue

        actual_value = float(actual_candidates.iloc[0]["Close"])
        predicted_value = float(prediction["predicted_value"])
        error_pct = abs((actual_value - predicted_value) / actual_value) * 100 if actual_value else None
        rows.append(
            {
                "target_date": target_date.date().isoformat(),
                "predicted_value": predicted_value,
                "actual_value": actual_value,
                "error_pct": error_pct,
            }
        )

    return pd.DataFrame(rows)
