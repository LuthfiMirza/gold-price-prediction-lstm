"""Modul untuk mengambil data harga emas dari sumber eksternal."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

GOLD_SYMBOL = "GC=F"
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


class DataFetchError(Exception):
    """Exception khusus saat pengambilan data harga emas gagal."""


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Merapikan kolom MultiIndex dari yfinance menjadi kolom standar."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _validate_price_data(df: pd.DataFrame, context: str) -> pd.DataFrame:
    """Memastikan data harga berisi kolom OHLC yang dibutuhkan."""
    if df.empty:
        raise DataFetchError(f"Data harga emas kosong saat {context}.")

    missing_columns = [column for column in OHLC_COLUMNS if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise DataFetchError(f"Data harga emas tidak memiliki kolom wajib: {missing}.")

    cleaned_df = df.dropna(subset=OHLC_COLUMNS).copy()
    if cleaned_df.empty:
        raise DataFetchError(f"Data harga emas tidak valid setelah pembersihan saat {context}.")

    cleaned_df.index = pd.to_datetime(cleaned_df.index)
    return cleaned_df.sort_index()


def fetch_historical(period: str = "5y") -> pd.DataFrame:
    """Mengambil data historis emas GC=F dari Yahoo Finance."""
    try:
        df = yf.download(GOLD_SYMBOL, period=period, progress=False, auto_adjust=False)
        return _validate_price_data(_flatten_columns(df), "mengambil data historis")
    except DataFetchError:
        raise
    except Exception as error:
        raise DataFetchError(f"Gagal mengambil data historis emas: {error}") from error


def resample_data(df: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Mengubah data harga menjadi horizon day, week, atau month secara OHLC."""
    normalized_horizon = horizon.lower()
    if normalized_horizon == "day":
        return _validate_price_data(_flatten_columns(df), "memproses data harian")

    resample_rules = {
        "week": "W-FRI",
        "month": "ME",
    }
    if normalized_horizon not in resample_rules:
        raise ValueError("horizon harus salah satu dari: day, week, month")

    clean_df = _validate_price_data(_flatten_columns(df), f"memproses data {normalized_horizon}")
    aggregation = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
    }
    if "Volume" in clean_df.columns:
        aggregation["Volume"] = "sum"

    resampled_df = clean_df.resample(resample_rules[normalized_horizon]).agg(aggregation)
    return _validate_price_data(resampled_df, f"resampling data {normalized_horizon}")


def get_latest_price() -> tuple[float, pd.Timestamp]:
    """Mengambil harga emas terbaru beserta timestamp data terakhir."""
    try:
        df = yf.download(GOLD_SYMBOL, period="5d", progress=False, auto_adjust=False)
        clean_df = _validate_price_data(_flatten_columns(df), "mengambil harga terbaru")
        latest_row = clean_df.iloc[-1]
        return float(latest_row["Close"]), pd.Timestamp(clean_df.index[-1])
    except DataFetchError:
        raise
    except Exception as error:
        raise DataFetchError(f"Gagal mengambil harga emas terbaru: {error}") from error
