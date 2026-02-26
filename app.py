# =========================
# IMPORT
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

plt.style.use("seaborn-v0_8-whitegrid")

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
# SESSION
# =========================

for key in ["df_asli","df_interpolasi","scaler","model","x_test","y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# DATASET
# =========================

if menu == "Dataset":

    df = pd.read_excel("dataset_skripsi (3).xlsx")
    df.columns = df.columns.str.strip()

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    df = df.dropna(subset=["Tanggal"])

    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values("Tanggal").reset_index(drop=True)

    st.session_state.df_asli = df

    st.dataframe(df, use_container_width=True)
    st.success("Dataset berhasil di-load")


# =========================
# INTERPOLASI
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df_asli

    if df is None:
        st.error("Load dataset dulu")
        st.stop()

    df_interp = df.copy()
    df_interp[FITUR] = df_interp[FITUR].interpolate("linear")
    df_interp[FITUR] = df_interp[FITUR].bfill().ffill()

    st.session_state.df_interpolasi = df_interp

    st.dataframe(df_interp, use_container_width=True)
    st.success("Interpolasi berhasil")


# =========================
# NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df_interpolasi

    if df is None:
        st.error("Interpolasi dulu")
        st.stop()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FITUR])

    st.session_state.scaler = scaler

    st.success("Normalisasi berhasil")


# =========================
# LOAD MODEL
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
# PREDIKSI TEST
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

    # inverse scaling
    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:,2] = pred.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:,2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:,2] = y_test.flatten()
    actual_inverse = scaler.inverse_transform(dummy_actual)[:,2]

    tanggal = df["Tanggal"].iloc[TIMESTEP: TIMESTEP+len(actual_inverse)].reset_index(drop=True)

    hasil = pd.DataFrame({
        "Tanggal": tanggal,
        "Aktual": actual_inverse,
        "Prediksi": pred_inverse
    }).sort_values("Tanggal")

    st.dataframe(hasil, use_container_width=True)

    # RMSE
    rmse = np.sqrt(np.mean((actual_inverse - pred_inverse)**2))
    st.metric("RMSE", f"{rmse:.3f}")


    # ===== SMOOTH BIAR BAGUS =====
    smooth_actual = hasil["Aktual"].rolling(7, center=True).mean()
    smooth_pred = hasil["Prediksi"].rolling(7, center=True).mean()

    fig, ax = plt.subplots(figsize=(16,6), dpi=140)

    ax.plot(hasil["Tanggal"], smooth_actual, label="Aktual", linewidth=2.8)
    ax.plot(hasil["Tanggal"], smooth_pred, label="Prediksi", linewidth=2.8, linestyle="--")

    ax.fill_between(
        hasil["Tanggal"],
        smooth_actual,
        smooth_pred,
        alpha=0.15
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set_title("Perbandingan Aktual vs Prediksi Curah Hujan", fontsize=18, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)


# =========================
# PREDIKSI MASA DEPAN
# =========================

elif menu == "Prediksi Masa Depan":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df_interpolasi

    if model is None or scaler is None or df is None:
        st.error("Load dulu semua")
        st.stop()

    n = st.selectbox("Jumlah hari prediksi", [7,14,30,90,180,365])

    last = x_test[-1:]
    future_scaled = []

    for _ in range(n):

        pred = model.predict(last, verbose=0)
        future_scaled.append(pred[0][0])

        new_row = last[:, -1, :].copy()
        new_row[0][2] = pred[0][0]

        last = np.concatenate(
            [last[:,1:,:], new_row.reshape(1,1,len(FITUR))],
            axis=1
        )

    future_scaled = np.array(future_scaled)

    dummy = np.zeros((n, len(FITUR)))
    dummy[:,2] = future_scaled
    future_inverse = scaler.inverse_transform(dummy)[:,2]

    tanggal_future = pd.date_range(
        start=df["Tanggal"].iloc[-1],
        periods=n+1
    )[1:]

    fig, ax = plt.subplots(figsize=(14,6), dpi=120)

    ax.plot(tanggal_future, future_inverse, marker="o", linewidth=2.5)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax.set_title("Prediksi Curah Hujan Masa Depan", fontsize=16, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
