"""Modul untuk membandingkan naive, LSTM, XGBoost, dan Prophet."""

from __future__ import annotations

import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

from backtest import naive_forecast, walk_forward_mape
from evaluate import evaluate_horizon
from model_train import LOOKBACK, VALID_HORIZONS


def _supervised_rows(series: pd.Series, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Membentuk fitur lag untuk model tabular seperti XGBoost."""
    values = series.dropna().astype(float).values
    features = []
    targets = []
    for current_index in range(window, len(values)):
        features.append(values[current_index - window : current_index])
        targets.append(values[current_index])
    return np.array(features), np.array(targets)


def xgboost_walk_forward_mape(series: pd.Series, window: int) -> float:
    """Menghitung MAPE XGBoost dengan fitur lag dan split kronologis."""
    features, targets = _supervised_rows(series, window)
    split_index = int(len(features) * 0.8)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, objective="reg:squarederror")
    model.fit(features[:split_index], targets[:split_index])
    predictions = model.predict(features[split_index:])
    actual_values = targets[split_index:]
    return float(np.mean(np.abs((actual_values - predictions) / actual_values)) * 100)


def prophet_mape(series: pd.Series) -> float:
    """Menghitung MAPE Prophet dengan split kronologis sederhana."""
    clean_series = series.dropna().astype(float)
    prophet_df = pd.DataFrame({"ds": clean_series.index.tz_localize(None), "y": clean_series.values})
    split_index = int(len(prophet_df) * 0.8)
    try:
        model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        model.fit(prophet_df.iloc[:split_index])
        future_df = prophet_df.iloc[split_index:][["ds"]]
        forecast = model.predict(future_df)
        actual_values = prophet_df.iloc[split_index:]["y"].values
        predictions = forecast["yhat"].values
        return float(np.mean(np.abs((actual_values - predictions) / actual_values)) * 100)
    except Exception as error:
        print(f"Prophet unavailable: {error}")
        return float("nan")


def compare_models() -> pd.DataFrame:
    """Membuat tabel perbandingan MAPE semua model untuk semua horizon."""
    rows = []
    for horizon in VALID_HORIZONS:
        evaluation = evaluate_horizon(horizon)
        series = pd.read_pickle(f"series_{horizon}.pkl").dropna().astype(float)
        rows.extend(
            [
                {"horizon": horizon, "model": "naive", "mape": evaluation["naive_mape"]},
                {"horizon": horizon, "model": "LSTM", "mape": evaluation["lstm_mape"]},
                {"horizon": horizon, "model": "XGBoost", "mape": xgboost_walk_forward_mape(series, LOOKBACK)},
                {"horizon": horizon, "model": "Prophet", "mape": prophet_mape(series)},
            ]
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Menjalankan perbandingan model dan mencetak tabel MAPE."""
    comparison_df = compare_models()
    print(comparison_df.pivot(index="horizon", columns="model", values="mape").round(4).to_string())


if __name__ == "__main__":
    main()
