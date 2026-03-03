# =========================
# IMPORT LIBRARY
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf


# =========================
# CONFIG
# =========================

FITUR = ["TAVG", "RH_AVG", "RR"]
TIMESTEP = 25

plt.style.use("seaborn-v0_8-darkgrid")
st.set_page_config(layout="wide", page_title="Prediksi Curah Hujan LSTM")

MODEL_META = {
    "path": "model_34_ep100_lr0.0001_ts25.h5",
    "epoch": 100,
    "lr": 0.0001,
    "durasi": "±31 menit",
    "timestep": 25
}


# =========================
# TITLE
# =========================

st.title("🌧️ Prediksi Curah Hujan Menggunakan LSTM")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dataset",
        "Interpolasi Linear",
        "Normalisasi",
        "Load Model",
        "Prediksi Test",
        "Prediksi Masa Depan"
    ]
)


# =========================
# SESSION STATE
# =========================

keys = [
    "df_asli",
    "df_interpolasi",
    "scaled_data",
    "scaler",
    "model",
    "x_test",
    "y_test",
    "model_meta_loaded"
]

for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None


# =========================
# MENU 1 — DATASET
# =========================

if menu == "Dataset":

    uploaded_file = st.file_uploader(
        "Upload dataset (Excel / CSV)",
        type=["xlsx", "xls", "csv"]
    )

    if uploaded_file is None:
        st.info("Silakan upload dataset terlebih dahulu.")
        st.stop()

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df.columns = df.columns.str.strip()

    kolom_wajib = ["Tanggal"] + FITUR
    kolom_hilang = [c for c in kolom_wajib if c not in df.columns]
    if kolom_hilang:
        st.error(f"Kolom berikut wajib ada: {kolom_hilang}")
        st.stop()

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True)
    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Tanggal"]).reset_index(drop=True)

    st.session_state.df_asli = df

    st.subheader("Dataset Asli")
    st.write(f"Jumlah data: **{len(df)} baris**")
    st.write(f"Periode: **{df['Tanggal'].min().date()}** s.d. **{df['Tanggal'].max().date()}**")
    st.dataframe(df, use_container_width=True, height=500)

    st.success("Dataset berhasil dimuat.")


# =========================
# MENU 2 — INTERPOLASI
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df_asli
    if df is None:
        st.error("Upload dataset dulu.")
        st.stop()

    df_interp = df.copy()
    df_interp[FITUR] = df_interp[FITUR].interpolate(method="linear")
    df_interp[FITUR] = df_interp[FITUR].bfill().ffill()

    st.session_state.df_interpolasi = df_interp

    st.subheader("Data Setelah Interpolasi")
    st.dataframe(df_interp, use_container_width=True, height=500)

    st.success("Interpolasi selesai.")


# =========================
# MENU 3 — NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df_interpolasi
    if df is None:
        st.error("Lakukan interpolasi dulu.")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FITUR])

    st.session_state.scaler = scaler
    st.session_state.scaled_data = scaled

    df_scaled = pd.DataFrame(scaled, columns=FITUR)
    df_scaled.insert(0, "Tanggal", df["Tanggal"].values)

    st.subheader("Data Setelah Normalisasi")
    st.dataframe(df_scaled, use_container_width=True, height=500)

    st.success("Normalisasi berhasil.")


# =========================
# MENU 4 — LOAD MODEL
# =========================

elif menu == "Load Model":

    if st.button("Load Model Terbaik"):
        try:
            model = load_model(MODEL_META["path"], compile=False)
            model.compile(optimizer="adam", loss="mse")

            x_test = pd.read_csv("X_test_34.csv").values
            y_test = pd.read_csv("y_test_34.csv").values
            x_test = x_test.reshape(x_test.shape[0], TIMESTEP, len(FITUR))

            st.session_state.model = model
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test
            st.session_state.model_meta_loaded = MODEL_META

            st.success("Model berhasil di-load")

        except Exception as e:
            st.error(f"Gagal load model: {e}")


# =========================
# MENU 5 — PREDIKSI TEST
# =========================

elif menu == "Prediksi Test":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test

    if model is None:
        st.warning("Load Model dulu.")
        st.stop()

    pred = model.predict(x_test, verbose=0)

    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:, 2] = pred.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, 2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:, 2] = y_test.flatten()
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, 2]

    rmse = np.sqrt(np.mean((actual_inverse - pred_inverse) ** 2))
    st.metric("RMSE", f"{rmse:.3f}")

    hasil = pd.DataFrame({
        "Aktual RR": actual_inverse,
        "Prediksi RR": pred_inverse
    })

    st.dataframe(hasil, use_container_width=True)


# =========================
# MENU 6 — PREDIKSI MASA DEPAN
# =========================

elif menu == "Prediksi Masa Depan":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df_interpolasi

    if model is None:
        st.warning("Load Model dulu.")
        st.stop()

    st.subheader("Pilih prediksi selanjutnya:")

    n = st.selectbox(
        "",
        options=[1, 3, 7, 14, 30, 90, 180, 365],
        index=2
    )

    last = x_test[-1:]
    future_scaled = []

    for _ in range(n):
        pred = model.predict(last, verbose=0)
        future_scaled.append(pred[0][0])

        new_row = last[:, -1, :].copy()
        new_row[0][2] = pred[0][0]

        last = np.concatenate(
            [last[:, 1:, :], new_row.reshape(1, 1, len(FITUR))],
            axis=1
        )

    dummy = np.zeros((n, len(FITUR)))
    dummy[:, 2] = np.array(future_scaled)

    future_inverse = scaler.inverse_transform(dummy)[:, 2]

    st.subheader("Prediksi Selanjutnya:")

    hasil_future = pd.DataFrame({
        "Prediksi": np.round(future_inverse, 2)
    })

    st.dataframe(hasil_future, height=300)

    st.markdown("---")

    tanggal_future = pd.date_range(
        start=df["Tanggal"].iloc[-1],
        periods=n + 1
    )[1:]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tanggal_future, future_inverse, marker="o")
    ax.set_title("Grafik Prediksi Masa Depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    plt.xticks(rotation=45)

    st.pyplot(fig, use_container_width=True)
