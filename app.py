"""Modul untuk menjalankan dashboard prediksi harga emas."""

from __future__ import annotations

import streamlit as st

from data_fetch import DataFetchError, get_latest_price

HORIZON_LABELS = {
    "day": "Harian",
    "week": "Mingguan",
    "month": "Bulanan",
}


st.set_page_config(page_title="Prediksi Harga Emas", page_icon="🏆", layout="wide")

st.sidebar.header("Pengaturan")
selected_label = st.sidebar.radio("Pilih horizon", list(HORIZON_LABELS.values()))
selected_horizon = next(key for key, value in HORIZON_LABELS.items() if value == selected_label)

st.title("Prediksi Harga Emas (XAU/USD)")
st.subheader(f"Horizon terpilih: {selected_label}")

try:
    current_price, latest_timestamp = get_latest_price()
    st.metric("Harga emas terkini", f"US$ {current_price:,.2f}")
    st.caption(f"Timestamp data terakhir: {latest_timestamp}")
except DataFetchError as error:
    st.warning(f"Gagal mengambil harga terbaru: {error}")

st.info("Prediksi belum ditampilkan pada fase ini. Dashboard baru menampilkan data live.")
