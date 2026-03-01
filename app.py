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
# MENU 1 — DATASET (UPLOAD)
# =========================

if menu == "Dataset":

    uploaded_file = st.file_uploader(
        "📂 Upload dataset (Excel / CSV)",
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
        st.error(f"Kolom berikut wajib ada di dataset: {kolom_hilang}")
        st.stop()

    st.subheader("🔎 Info Awal Dataset")
    st.write("Jumlah baris awal:", len(df))

    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce", dayfirst=True)
    gagal_parse = df["Tanggal"].isna().sum()
    if gagal_parse > 0:
        st.warning(f"Ada {gagal_parse} baris bukan data (keterangan) & dibuang.")

    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Tanggal"]).reset_index(drop=True)

    st.session_state.df_asli = df

    st.subheader("📄 Dataset Setelah Cleaning")
    st.write(f"Jumlah data: **{len(df)} baris**")
    st.write(f"Periode: **{df['Tanggal'].min().date()}** s.d. **{df['Tanggal'].max().date()}**")
    st.dataframe(df, use_container_width=True, height=600)

    st.success("Dataset siap digunakan")


# =========================
# MENU 2 — INTERPOLASI (FINAL)
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df_asli

    if df is None:
        st.error("Upload dataset dulu di menu Dataset.")
        st.stop()

    mask_na = df[FITUR].isna().any(axis=1)
    df_missing = df.loc[mask_na, ["Tanggal"] + FITUR]

    st.subheader("🔎 Data yang Mengalami Missing Value (Sebelum Interpolasi)")
    st.write(f"Jumlah baris yang diinterpolasi: **{len(df_missing)} baris**")

    if len(df_missing) == 0:
        st.success("Tidak ada missing value. Tidak perlu interpolasi.")
        st.stop()

    st.dataframe(df_missing, use_container_width=True, height=400)

    st.subheader("📐 Metode Interpolasi Linear")
    st.latex(r"x_t = x_a + (x_b - x_a) \times \frac{t - t_a}{t_b - t_a}")

    df_interp = df.copy()
    df_interp[FITUR] = df_interp[FITUR].interpolate(method="linear")
    df_interp[FITUR] = df_interp[FITUR].bfill().ffill()

    st.session_state.df_interpolasi = df_interp

    df_after = df_interp.loc[mask_na, ["Tanggal"] + FITUR]

    st.subheader("📊 Hasil Setelah Interpolasi (Nomor Urut = Posisi Data Asli)")

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
        height=400
    )

    st.subheader("📈 Grafik Sebelum vs Sesudah Interpolasi (RR)")

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)

    ax.plot(df["Tanggal"], df["RR"], label="Sebelum Interpolasi", color="#1f77b4", linewidth=2, alpha=0.7)  # biru
    ax.plot(df_interp["Tanggal"], df_interp["RR"], label="Sesudah Interpolasi", color="#d62728", linewidth=2.8)  # merah

    ax.scatter(df_missing["Tanggal"], df_missing["RR"], color="#9467bd", label="Titik Missing", s=50, zorder=5)

    ax.set_title("Perbandingan Curah Hujan Sebelum vs Sesudah Interpolasi", fontsize=18, fontweight="bold")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.success("Interpolasi selesai — nomor urut & visual sudah rapi")


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

    st.subheader("📊 Data Setelah Normalisasi")
    st.dataframe(df_scaled, use_container_width=True)

    st.success("Normalisasi berhasil")


# =========================
# MENU 4 — LOAD MODEL
# =========================

elif menu == "Load Model":

    if st.button("🚀 Load Model"):
        try:
            model = load_model("model_34_ep100_lr0.0001_ts25.h5", compile=False)
            model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError(), metrics=["mae"])

            x_test = pd.read_csv("X_test_34.csv").values
            y_test = pd.read_csv("y_test_34.csv").values

            x_test = x_test.reshape(x_test.shape[0], TIMESTEP, len(FITUR))

            st.session_state.model = model
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test

            st.success("Model & data test berhasil di-load")
        except Exception as e:
            st.error(f"Gagal load model atau data test: {e}")


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
        st.error("Pastikan dataset, normalisasi, dan model sudah di-load.")
        st.stop()

    pred = model.predict(x_test, verbose=0)

    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:, 2] = pred.flatten()
    pred_inverse = scaler.inverse_transform(dummy_pred)[:, 2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:, 2] = y_test.flatten()
    actual_inverse = scaler.inverse_transform(dummy_actual)[:, 2]

    tanggal = df["Tanggal"].iloc[TIMESTEP: TIMESTEP + len(actual_inverse)].reset_index(drop=True)

    hasil = pd.DataFrame({"Tanggal": tanggal, "Aktual RR": actual_inverse, "Prediksi RR": pred_inverse})

    st.subheader("📉 Hasil Prediksi Data Test")
    st.dataframe(hasil, use_container_width=True)

    rmse = np.sqrt(np.mean((actual_inverse - pred_inverse) ** 2))
    st.metric("RMSE", f"{rmse:.3f}")

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    ax.plot(hasil["Tanggal"], hasil["Aktual RR"], label="Aktual", linewidth=2.5)
    ax.plot(hasil["Tanggal"], hasil["Prediksi RR"], label="Prediksi", linewidth=2.5, linestyle="--")
    ax.set_title("Perbandingan Aktual vs Prediksi Curah Hujan", fontsize=18, fontweight="bold")
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
        st.error("Pastikan dataset, normalisasi, dan model sudah di-load.")
        st.stop()

    n = st.selectbox("Jumlah hari prediksi", [1, 7, 14, 30, 90, 180, 365])

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

    tanggal_future = pd.date_range(start=df["Tanggal"].iloc[-1], periods=n + 1)[1:]
    hasil_future = pd.DataFrame({"Tanggal": tanggal_future, "Prediksi RR": future_inverse})

    st.subheader("🔮 Prediksi Curah Hujan Masa Depan")
    st.dataframe(hasil_future, use_container_width=True)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    ax.plot(tanggal_future, future_inverse, marker="o", linewidth=2.5)
    ax.set_title("Prediksi Curah Hujan Masa Depan", fontsize=18, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
