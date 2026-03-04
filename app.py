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

# Metadata model (sesuaikan dengan model kamu)
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
# MENU 1 — DATASET (UPLOAD)
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

    st.subheader("Info Awal Dataset")
    st.write("Jumlah baris awal:", len(df))

  
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True).dt.date
    gagal = df["Tanggal"].isna().sum()
    if gagal > 0:
        st.warning(f"{gagal} baris bukan data tanggal & dibuang.")

    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Tanggal"]).reset_index(drop=True)

    st.session_state.df_asli = df

    st.subheader("Dataset Asli")
    st.write(f"Jumlah data: **{len(df)} baris**")
    st.write(f"Periode: **{df['Tanggal'].min()}** s.d. **{df['Tanggal'].max()}**")
    st.dataframe(df, use_container_width=True, height=600)

    st.success("Dataset berhasil dimuat lengkap.")


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

    mask_missing = df[FITUR].isna().any(axis=1)
    df_missing = df[mask_missing].copy()
    df_after = df_interp.loc[df_missing.index].copy()

    st.subheader("Data Setelah Dilakukan Interpolasi :")

    compare = df_missing.copy()
    after = df_after.copy()
    compare.insert(0, "No", compare.index)
    after.insert(0, "No", after.index)

    hasil_compare = compare.copy()
    for col in FITUR:
        hasil_compare[col] = (
            compare[col].astype(str)
            + "  →  "
            + after[col].round(3).astype(str)
        )

    st.dataframe(
        hasil_compare[["No", "Tanggal"] + FITUR],
        use_container_width=True,
        height=450
    )

    st.info(f"Jumlah data yang diinterpolasi: {mask_missing.sum()} baris")

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.plot(df["Tanggal"], df["RR"], color="red", alpha=0.6, label="Sebelum Interpolasi")
    ax.plot(df_interp["Tanggal"], df_interp["RR"], color="blue", linewidth=2, label="Sesudah Interpolasi")
    ax.set_title("Perbandingan Curah Hujan (RR) Sebelum vs Sesudah Interpolasi")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

    st.success("Interpolasi selesai.")


# =========================
# MENU 3 — NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df_interpolasi
    if df is None:
        st.error("Lakukan interpolasi dulu.")
        st.stop()

    st.subheader("Nilai Minimum & Maksimum (Sebelum Normalisasi)")
    minmax = pd.DataFrame({"Min": df[FITUR].min(), "Max": df[FITUR].max()})
    st.dataframe(minmax, use_container_width=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FITUR])

    st.session_state.scaler = scaler
    st.session_state.scaled_data = scaled

    df_scaled = pd.DataFrame(scaled, columns=FITUR)
    df_scaled.insert(0, "Tanggal", df["Tanggal"].values)

    st.subheader("Data Setelah Normalisasi (MinMax 0–1)")
    st.dataframe(df_scaled, use_container_width=True, height=500)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.plot(df_scaled["Tanggal"], df_scaled["RR"], color="blue", label="RR (Normalisasi)")
    ax.set_title("Grafik Curah Hujan (RR) Setelah Normalisasi")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Nilai Normalisasi")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

    st.success("Normalisasi berhasil (MinMaxScaler 0–1).")


# =========================
# MENU 5 — PREDIKSI TEST
# =========================

elif menu == "Prediksi Test":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    y_test = st.session_state.y_test
    df = st.session_state.df_interpolasi

    if model is None:
        st.warning("Load Model dulu di menu 'Load Model'.")
        st.stop()

    if scaler is None or df is None or x_test is None or y_test is None:
        st.error("Pastikan dataset, normalisasi, dan data test sudah siap.")
        st.stop()

    st.subheader("Model yang Digunakan")
    meta = st.session_state.model_meta_loaded
    if meta:
        st.write(f"Epoch: {meta['epoch']} | LR: {meta['lr']} | Durasi: {meta['durasi']} | Timestep: {meta['timestep']}")

    pred = model.predict(x_test, verbose=0)

    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:, 2] = pred.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, 2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:, 2] = y_test.flatten()
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, 2]

    rmse = np.sqrt(np.mean((actual_inverse - pred_inverse) ** 2))
    st.metric("RMSE (Data Test)", f"{rmse:.3f}")

    tanggal = df["Tanggal"].iloc[TIMESTEP: TIMESTEP + len(actual_inverse)].reset_index(drop=True)
    hasil = pd.DataFrame({"Tanggal": tanggal, "Aktual RR": actual_inverse, "Prediksi RR": pred_inverse})

    st.subheader("Hasil Prediksi Data Test")
    st.dataframe(hasil, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.plot(hasil["Tanggal"], hasil["Aktual RR"], color="blue", label="Aktual")
    ax.plot(hasil["Tanggal"], hasil["Prediksi RR"], color="red", linestyle="--", label="Prediksi")
    ax.set_title("Perbandingan Aktual vs Prediksi Curah Hujan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)


# =========================
# MENU 6 — PREDIKSI MASA DEPAN
# =========================

elif menu == "Prediksi Masa Depan":

    model = st.session_state.model
    scaler = st.session_state.scaler
    x_test = st.session_state.x_test
    df = st.session_state.df_interpolasi

    if model is None:
        st.warning("Load Model dulu di menu 'Load Model'.")
        st.stop()

    if scaler is None or df is None or x_test is None:
        st.error("Pastikan dataset, normalisasi, dan data test sudah siap.")
        st.stop()

    horizon = st.radio("Periode Waktu ke Depan", ["Short-term (1–14 hari)", "Long-term (30–365 hari)"])
    if horizon == "Short-term (1–14 hari)":
        n = st.selectbox("Pilih jumlah hari", [1, 7, 14])
    else:
        n = st.selectbox("Pilih jumlah hari", [30, 90, 180, 365])

    last = x_test[-1:]
    future_scaled = []

    for _ in range(n):
        pred = model.predict(last, verbose=0)
        future_scaled.append(pred[0][0])
        new_row = last[:, -1, :].copy()
        new_row[0][2] = pred[0][0]
        last = np.concatenate([last[:, 1:, :], new_row.reshape(1, 1, len(FITUR))], axis=1)

    dummy = np.zeros((n, len(FITUR)))
    dummy[:, 2] = np.array(future_scaled)
    future_inverse = scaler.inverse_transform(dummy)[:, 2]

    # Hapus detik pada tanggal
    tanggal_future = pd.date_range(start=df["Tanggal"].iloc[-1], periods=n + 1)[1:].date
    hasil_future = pd.DataFrame({"Tanggal": tanggal_future, "Prediksi RR": future_inverse})

    st.subheader("Prediksi Curah Hujan Masa Depan")
    st.dataframe(hasil_future, use_container_width=True)

    fig, ax = plt.subplots(figsize=(14, 5), dpi=140)
    ax.plot(tanggal_future, future_inverse, color="blue", marker="o")
    ax.set_title("Prediksi Curah Hujan Masa Depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    plt.xticks(rotation=45)
    st.pyplot(fig, use_container_width=True)

