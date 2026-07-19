"""Modul untuk menjalankan dashboard prediksi harga emas."""

from __future__ import annotations

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh
from tensorflow.keras.models import load_model

from data_fetch import DataFetchError, fetch_historical, get_latest_price, resample_data
from model_train import LOOKBACK, compute_returns
from model_train import train as train_model
from prediction_log import get_realized_comparisons, log_prediction

try:
    from model_compare import compare_models
except Exception:
    compare_models = None

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
AUTO_REFRESH_INTERVAL_MS = 10 * 60 * 1000


def render_tradingview_widget() -> None:
    """Menampilkan widget chart live TradingView (embed gratis, tanpa API key).

    Widget ini murni tampilan pihak ketiga untuk konteks pasar real-time —
    bukan sumber data model. Prediksi dashboard tetap dihitung dari yfinance
    lewat data_fetch.py, karena widget embed tidak menyediakan data terprogram.
    """
    st.caption("Live market (TradingView) — konteks pasar real-time, terpisah dari data prediksi di atas.")
    components.html(
        """
        <div class="tradingview-widget-container">
          <div class="tradingview-widget-container__widget"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-symbol-overview.js" async>
          {
            "symbols": [["OANDA:XAUUSD|1D"]],
            "chartOnly": false,
            "width": "100%",
            "height": "220",
            "locale": "id_ID",
            "colorTheme": "light",
            "autosize": true,
            "showVolume": false,
            "showMA": false,
            "hideDateRanges": false,
            "hideMarketStatus": false,
            "hideSymbolLogo": false,
            "scalePosition": "right",
            "scaleMode": "Normal",
            "fontFamily": "-apple-system, BlinkMacSystemFont, sans-serif",
            "noTimeScale": false,
            "valuesTracking": "1",
            "changeMode": "price-and-percent",
            "chartType": "area"
          }
          </script>
        </div>
        """,
        height=240,
    )


def render_freshness_badge(latest_timestamp: pd.Timestamp) -> None:
    """Menampilkan status kesegaran data berdasarkan timestamp terbaru."""
    latest_time = pd.Timestamp(latest_timestamp)
    if latest_time.tzinfo is None:
        latest_time = latest_time.tz_localize("UTC")
    now = pd.Timestamp.now(tz="UTC")
    minutes_old = max((now - latest_time).total_seconds() / 60, 0)

    if minutes_old < 15:
        st.success(f"Data segar - {minutes_old:.0f} menit lalu")
    elif minutes_old < 60:
        st.warning(f"Data cukup segar - {minutes_old:.0f} menit lalu")
    else:
        st.error(f"Data lama - {minutes_old / 60:.1f} jam lalu")


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


def load_training_metadata(horizon: str) -> dict | None:
    """Memuat metadata training untuk horizon terpilih jika tersedia."""
    metadata_path = _artifact_path("metadata", horizon, "json")
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text())


def render_training_controls(horizon: str) -> None:
    """Menampilkan metadata model dan tombol retrain manual."""
    metadata = load_training_metadata(horizon)
    if metadata:
        st.sidebar.caption(f"Model terakhir dilatih: {metadata['last_trained_at']}")
        st.sidebar.caption(f"Validation MAPE: {metadata['validation_mape']:.2f}%")
    else:
        st.sidebar.caption("Model belum memiliki metadata training.")

    if st.sidebar.button("Latih ulang model sekarang"):
        with st.spinner("Melatih ulang model, mohon tunggu..."):
            train_model(horizon)
        st.success("Training selesai. Refresh dashboard untuk memuat artifact terbaru.")


def _load_prediction_artifacts(horizon: str):
    """Memuat model, scaler, dan series untuk prediksi dashboard."""
    model = load_model(_artifact_path("model_lstm", horizon, "keras"))
    with _artifact_path("scaler", horizon, "pkl").open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    with _artifact_path("series", horizon, "pkl").open("rb") as series_file:
        series = pickle.load(series_file)
    return model, scaler, series.dropna().astype(float)


def predict_next_price(horizon: str) -> tuple[float, pd.Series]:
    """Memprediksi harga periode berikutnya lewat return, lalu direkonstruksi ke skala harga."""
    model, scaler, series = _load_prediction_artifacts(horizon)
    if len(series) < LOOKBACK + 1:
        raise ValueError("series tidak cukup panjang untuk prediksi")

    prices = series.values
    returns = compute_returns(prices)
    last_window = returns[-LOOKBACK:].reshape(-1, 1)
    scaled_window = scaler.transform(last_window)
    scaled_prediction = model.predict(scaled_window.reshape(1, LOOKBACK, 1), verbose=0)
    predicted_return = scaler.inverse_transform(scaled_prediction)[0][0]
    last_price = float(prices[-1])
    predicted_price = last_price * (1 + predicted_return)
    return float(predicted_price), series


def render_horizon_overview() -> None:
    """Menampilkan ringkasan prediksi besok, minggu depan, dan bulan depan sekaligus.

    Supaya user tidak perlu gonta-ganti radio horizon cuma buat lihat gambaran
    besar: naik/turun dan berapa persen untuk ketiga horizon dalam satu pandangan.
    """
    st.subheader("Ringkasan cepat: besok, minggu depan, bulan depan")
    columns = st.columns(3)
    for column, (horizon, label) in zip(columns, HORIZON_LABELS.items()):
        with column:
            if not _artifacts_ready(horizon):
                st.caption(f"{label}: model belum dilatih.")
                continue
            try:
                predicted_price, series = predict_next_price(horizon)
            except ValueError as error:
                st.caption(f"{label}: {error}")
                continue
            last_price = float(series.iloc[-1])
            percent_change = ((predicted_price - last_price) / last_price) * 100
            st.metric(label, f"US$ {predicted_price:,.2f}", f"{percent_change:+.2f}%")


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


def render_price_chart(series: pd.Series, predicted_price: float, horizon: str) -> None:
    """Menampilkan grafik historis dan titik prediksi, dengan garis penghubung arah dan rentang error bar."""
    chart_type = st.toggle("Tampilkan candlestick", value=False)
    history_df = resample_data(fetch_historical(), horizon).tail(180)
    last_date = history_df.index[-1]
    last_price = float(history_df["Close"].iloc[-1])
    offset = pd.tseries.frequencies.to_offset({"day": "B", "week": "W-FRI", "month": "ME"}[horizon])
    prediction_date = last_date + offset
    confidence_band = MAPE_BAND[horizon] / 100
    lower_bound = predicted_price * (1 - confidence_band)
    upper_bound = predicted_price * (1 + confidence_band)
    is_up = predicted_price >= last_price
    # Palet warna disamakan dengan skema TradingView: biru untuk garis harga,
    # hijau/merah teal khas TradingView untuk naik/turun (candlestick & arah prediksi).
    tv_up_color = "#26a69a"
    tv_down_color = "#ef5350"
    tv_line_color = "#2962ff"
    trend_color = tv_up_color if is_up else tv_down_color

    fig = go.Figure()
    if chart_type:
        fig.add_trace(
            go.Candlestick(
                x=history_df.index,
                open=history_df["Open"],
                high=history_df["High"],
                low=history_df["Low"],
                close=history_df["Close"],
                name="Harga historis",
                increasing_line_color=tv_up_color,
                decreasing_line_color=tv_down_color,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=history_df.index,
                y=history_df["Close"],
                mode="lines",
                name="Harga historis",
                line=dict(color=tv_line_color, width=1.5),
            )
        )

    # Garis putus-putus penghubung harga terakhir -> prediksi, supaya arahnya kelihatan jelas.
    fig.add_trace(
        go.Scatter(
            x=[last_date, prediction_date],
            y=[last_price, predicted_price],
            mode="lines",
            line=dict(color=trend_color, dash="dash", width=2),
            name="Arah prediksi",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[prediction_date],
            y=[predicted_price],
            mode="markers+text",
            marker=dict(color=trend_color, size=14, symbol="diamond"),
            text=[f"{'▲' if is_up else '▼'} US$ {predicted_price:,.0f}"],
            textposition="top center" if is_up else "bottom center",
            error_y=dict(
                type="data",
                symmetric=False,
                array=[upper_bound - predicted_price],
                arrayminus=[predicted_price - lower_bound],
                color=trend_color,
                thickness=2,
                width=8,
            ),
            name="Prediksi",
        )
    )

    # Beri ruang kosong di sisi kanan supaya titik prediksi tidak mepet ke tepi grafik.
    x_padding = (prediction_date - history_df.index[0]) * 0.05
    fig.update_xaxes(range=[history_df.index[0], prediction_date + x_padding])

    # Warna abu-abu netral dan label di kiri, supaya tidak bentrok dengan warna arah
    # prediksi (hijau/merah) atau tertutup label prediksi yang ada di tepi kanan.
    support_level = float(history_df["Low"].tail(60).min())
    resistance_level = float(history_df["High"].tail(60).max())
    fig.add_hline(
        y=support_level,
        line_dash="dot",
        line_color="rgba(120,120,120,0.6)",
        annotation_text="Support",
        annotation_position="bottom left",
    )
    fig.add_hline(
        y=resistance_level,
        line_dash="dot",
        line_color="rgba(120,120,120,0.6)",
        annotation_text="Resistance",
        annotation_position="top left",
    )
    # Layout ala TradingView: latar putih bersih, sumbu harga di kanan, gridline
    # horizontal tipis saja (vertikal dimatikan), garis crosshair mengikuti kursor.
    fig.update_layout(
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="-apple-system, BlinkMacSystemFont, Trebuchet MS, Roboto, sans-serif", color="#131722", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=60, t=30, b=10),
    )
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#e0e3eb",
        showspikes=True,
        spikemode="across",
        spikecolor="#758696",
        spikethickness=1,
        title_text="",
    )
    fig.update_yaxes(
        side="right",
        showgrid=True,
        gridcolor="#f0f3fa",
        showline=False,
        title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def next_target_date(series: pd.Series, horizon: str) -> pd.Timestamp:
    """Menghitung tanggal target prediksi berikutnya berdasarkan horizon."""
    offset = pd.tseries.frequencies.to_offset({"day": "B", "week": "W-FRI", "month": "ME"}[horizon])
    return pd.Timestamp(series.index[-1]) + offset


def render_prediction_history(horizon: str) -> None:
    """Menampilkan riwayat prediksi yang sudah punya harga realisasi."""
    st.subheader("Riwayat prediksi vs realisasi")
    comparison_df = get_realized_comparisons(horizon)
    if comparison_df.empty:
        st.caption("Belum ada prediksi lama yang bisa dibandingkan dengan realisasi.")
        return
    st.dataframe(comparison_df, use_container_width=True)


def render_model_comparison() -> None:
    """Menampilkan panel akurasi model jika dependency perbandingan tersedia."""
    st.subheader("Panel akurasi model")
    if compare_models is None:
        st.caption("Panel perbandingan belum tersedia di environment ini.")
        return
    if st.button("Hitung perbandingan model"):
        with st.spinner("Menghitung MAPE naive, LSTM, XGBoost, dan Prophet..."):
            comparison_df = compare_models()
        st.dataframe(comparison_df, use_container_width=True)


def render_csv_download(horizon: str, predicted_price: float, target_date: pd.Timestamp) -> None:
    """Menyediakan unduhan CSV berisi data historis dan prediksi."""
    historical_df = resample_data(fetch_historical(), horizon).copy()
    export_df = historical_df.reset_index().rename(columns={"Date": "date"})
    prediction_row = pd.DataFrame(
        [
            {
                "date": target_date,
                "Open": np.nan,
                "High": np.nan,
                "Low": np.nan,
                "Close": predicted_price,
                "Volume": np.nan,
                "type": "prediction",
            }
        ]
    )
    export_df["type"] = "historical"
    export_df = pd.concat([export_df, prediction_row], ignore_index=True)
    st.download_button(
        "Unduh CSV",
        data=export_df.to_csv(index=False),
        file_name=f"gold_prediction_{horizon}.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="Prediksi Harga Emas", page_icon="🏆", layout="wide")

# Auto-refresh hanya menjalankan ulang path fetch + predict Streamlit, tidak memanggil train_model().
st_autorefresh(interval=AUTO_REFRESH_INTERVAL_MS, key="gold_dashboard_autorefresh")

selected_horizon, selected_label = _selected_horizon()
render_training_controls(selected_horizon)

st.title("Prediksi Harga Emas (XAU/USD)")
render_tradingview_widget()
render_horizon_overview()
st.divider()
st.subheader(f"Horizon terpilih: {selected_label}")

try:
    current_price, latest_timestamp = get_latest_price()
    st.caption(f"Timestamp data terakhir: {latest_timestamp}")
    render_freshness_badge(latest_timestamp)

    if not _artifacts_ready(selected_horizon):
        st.warning("Model belum dilatih untuk horizon ini. Jalankan training terlebih dahulu.")
        st.code(f"python model_train.py --horizon {selected_horizon}")
    else:
        predicted_price, prediction_series = predict_next_price(selected_horizon)
        target_date = next_target_date(prediction_series, selected_horizon)
        log_prediction(selected_horizon, predicted_price, target_date)
        render_prediction_cards(current_price, predicted_price, selected_horizon)
        render_price_chart(prediction_series, predicted_price, selected_horizon)
        render_csv_download(selected_horizon, predicted_price, target_date)
        render_model_comparison()
        render_prediction_history(selected_horizon)
except DataFetchError as error:
    st.warning(f"⚠️ Gagal mengambil data terbaru dari Yahoo Finance: {error}")
except ValueError as error:
    st.warning(str(error))

st.divider()
st.caption(
    "Disclaimer: prediksi ini hanya untuk riset dan edukasi, bukan rekomendasi finansial "
    "atau ajakan membeli/menjual aset apa pun."
)
