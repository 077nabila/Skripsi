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

st.set_page_config(layout="wide")


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
    "y_test"
]

for k in keys:
    if k not in st.session_state:
        st.session_state[k] = None


# =========================
# MENU 1 — DATASET
# =========================

if menu == "Dataset":

    df = pd.read_excel("dataset_skripsi (3).xlsx")
    df.columns = df.columns.str.strip()

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df = df.dropna(subset=["Tanggal"])

    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.reset_index(drop=True)

    st.session_state.df_asli = df

    st.subheader("Dataset Asli")
    st.dataframe(df, use_container_width=True)

    st.success("Dataset berhasil di-load")


# =========================
# MENU 2 — INTERPOLASI
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df_asli

    if df is None:
        st.error("Load Dataset dulu")
        st.stop()

    df_interp = df.copy()
    df_interp[FITUR] = df_interp[FITUR].interpolate(method="linear")
    df_interp[FITUR] = df_interp[FITUR].bfill()
    df_interp[FITUR] = df_interp[FITUR].ffill()

    st.session_state.df_interpolasi = df_interp

    st.subheader("Data Setelah Interpolasi")
    st.dataframe(df_interp, use_container_width=True)

    st.success("Interpolasi berhasil")


# =========================
# MENU 3 — NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df_interpolasi

    if df is None:
        st.error("Lakukan interpolasi dulu")
        st.stop()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FITUR])

    st.session_state.scaler = scaler
    st.session_state.scaled_data = scaled

    df_scaled = pd.DataFrame(scaled, columns=FITUR)
    df_scaled.insert(0, "Tanggal", df["Tanggal"].values)

    st.subheader("Data Setelah Normalisasi")
    st.dataframe(df_scaled, use_container_width=True)

    st.success("Normalisasi berhasil")


# =========================
# MENU 4 — LOAD MODEL
# =========================

elif menu == "Load Model":

    if st.button("Load Model"):

        model = load_model(
            "model_34_ep100_lr0.0001_ts25.h5",
            compile=False
        )

        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=["mae"]
        )

        x_test = pd.read_csv("X_test_34.csv").values
        y_test = pd.read_csv("y_test_34.csv").values

        x_test = x_test.reshape(
            x_test.shape[0],
            TIMESTEP,
            len(FITUR)
        )

        st.session_state.model = model
        st.session_state.x_test = x_test
        st.session_state.y_test = y_test

        st.success("Model berhasil di-load")


# =========================
# MENU 5 — PREDIKSI TEST
# =========================

elif menu == "Prediksi Test":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    df = st.session_state.df_interpolasi

    if model is None or scaler is None or df is None:
        st.error("Load dataset, normalisasi, dan model dulu")
        st.stop()

    pred = model.predict(x_test, verbose=0)

    # Inverse scaling
    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:, 2] = pred.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, 2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:, 2] = y_test.flatten()
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, 2]

    tanggal = df["Tanggal"].iloc[TIMESTEP: TIMESTEP + len(actual_inverse)].reset_index(drop=True)

    hasil = pd.DataFrame({
        "Tanggal": tanggal,
        "Aktual RR": actual_inverse,
        "Prediksi RR": pred_inverse
    }).sort_values("Tanggal")

    st.subheader("Hasil Prediksi Test")
    st.dataframe(hasil, use_container_width=True)

    # RMSE
    rmse = np.sqrt(np.mean((actual_inverse - pred_inverse) ** 2))
    st.metric("RMSE", f"{rmse:.3f}")

    # ======================
    # GRAFIK BAGUS
    # ======================

    smooth_actual = hasil["Aktual RR"].rolling(7, center=True).mean()
    smooth_pred = hasil["Prediksi RR"].rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(16,6), dpi=140)

    ax.plot(
        hasil["Tanggal"],
        smooth_actual,
        label="Aktual",
        linewidth=2.8
    )

    ax.plot(
        hasil["Tanggal"],
        smooth_pred,
        label="Prediksi",
        linewidth=2.8,
        linestyle="--"
    )

    ax.fill_between(
        hasil["Tanggal"],
        smooth_actual,
        smooth_pred,
        alpha=0.15
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set_title(
        "Perbandingan Aktual vs Prediksi Curah Hujan",
        fontsize=18,
        fontweight="bold"
    )

    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)


# =========================
# MENU 6 — PREDIKSI MASA DEPAN
# =========================

elif menu == "Prediksi Masa Depan":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df_interpolasi

    if model is None or scaler is None or df is None:
        st.error("Load dataset, normalisasi, dan model dulu")
        st.stop()

    n = st.selectbox("Jumlah hari prediksi", [1, 7, 14, 30, 90, 180, 365])

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

    future_scaled = np.array(future_scaled)

    dummy = np.zeros((n, len(FITUR)))
    dummy[:, 2] = future_scaled
    future_inverse = scaler.inverse_transform(dummy)[:, 2]

    tanggal_future = pd.date_range(
        start=df["Tanggal"].iloc[-1],
        periods=n + 1
    )[1:]

    hasil_future = pd.DataFrame({
        "Tanggal": tanggal_future,
        "Prediksi RR": future_inverse
    })

    st.subheader("Prediksi Masa Depan")
    st.dataframe(hasil_future, use_container_width=True)

    fig, ax = plt.subplots(figsize=(16,6), dpi=140)

    ax.plot(
        tanggal_future,
        future_inverse,
        marker="o",
        linewidth=2.5
    )

    ax.set_title("Prediksi Curah Hujan Masa Depan", fontsize=18, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
