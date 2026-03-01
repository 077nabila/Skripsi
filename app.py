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
# MENU 1 — DATASET
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
        st.warning(f"Ada {gagal_parse} baris bukan data & dibuang.")

    df[FITUR] = df[FITUR].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Tanggal"]).reset_index(drop=True)

    st.session_state.df_asli = df

    st.subheader("📄 Dataset")
    st.write(f"Jumlah data: **{len(df)} baris**")
    st.write(f"Periode: **{df['Tanggal'].min().date()}** s.d. **{df['Tanggal'].max().date()}**")
    st.dataframe(df, use_container_width=True, height=600)

    st.success("Dataset siap diproses")


# =========================
# MENU 2 — INTERPOLASI (FINAL)
# =========================

elif menu == "Interpolasi Linear":

    df = st.session_state.df_asli

    if df is None:
        st.error("Upload dataset dulu di menu Dataset.")
        st.stop()

    st.subheader("🔎 Data yang Mengalami Missing Value")

    mask_na = df[FITUR].isna().any(axis=1)
    df_na = df.loc[mask_na, ["Tanggal"] + FITUR]

    st.write(f"Jumlah baris yang memiliki missing value: **{len(df_na)} baris**")

    if len(df_na) > 0:
        st.dataframe(df_na, use_container_width=True)
    else:
        st.success("Tidak ada missing value")

    st.subheader("📐 Metode Interpolasi Linear")
    st.latex(r"x_t = x_a + (x_b - x_a) \times \frac{t - t_a}{t_b - t_a}")

    df_interp = df.copy()
    df_interp[FITUR] = df_interp[FITUR].interpolate(method="linear")
    df_interp[FITUR] = df_interp[FITUR].bfill().ffill()

    st.session_state.df_interpolasi = df_interp

    if len(df_na) > 0:
        st.subheader("📊 Perbandingan Sebelum → Sesudah Interpolasi")

        before = df.loc[mask_na, ["Tanggal"] + FITUR].reset_index(drop=True)
        after = df_interp.loc[mask_na, ["Tanggal"] + FITUR].reset_index(drop=True)

        compare = before.copy()
        for col in FITUR:
            compare[col] = before[col].astype(str) + " → " + after[col].round(3).astype(str)

        st.dataframe(compare, use_container_width=True)

    st.subheader("📈 Grafik Before vs After (RR)")

    fig, ax = plt.subplots(figsize=(16, 5), dpi=140)
    ax.plot(df["Tanggal"], df["RR"], label="Sebelum", alpha=0.6)
    ax.plot(df_interp["Tanggal"], df_interp["RR"], label="Sesudah", linewidth=2.5)

    ax.set_title("Curah Hujan Sebelum vs Sesudah Interpolasi")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)

    st.success("Interpolasi selesai")


# =========================
# MENU 3 — NORMALISASI
# =========================

elif menu == "Normalisasi":

    df = st.session_state.df_interpolasi

    if df is None:
        st.error("Lakukan interpolasi dulu.")
        st.stop()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[FITUR])

    st.session_state.scaler = scaler
    st.session_state.scaled_data = scaled

    df_scaled = pd.DataFrame(scaled, columns=FITUR)
    df_scaled.insert(0, "Tanggal", df["Tanggal"].values)

    st.subheader("📊 Data Setelah Normalisasi")
    st.dataframe(df_scaled, use_container_width=True)

    st.success("Normalisasi selesai")


# =========================
# MENU 4 — LOAD MODEL
# =========================

elif menu == "Load Model":

    if st.button("🚀 Load Model"):
        try:
            model = load_model("model_34_ep100_lr0.0001_ts25.h5", compile=False)
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])

            x_test = pd.read_csv("X_test_34.csv").values.reshape(-1, TIMESTEP, len(FITUR))
            y_test = pd.read_csv("y_test_34.csv").values

            st.session_state.model = model
            st.session_state.x_test = x_test
            st.session_state.y_test = y_test

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
    df = st.session_state.df_interpolasi

    if None in (model, scaler, x_test, y_test, df):
        st.error("Pastikan semua tahap sudah dijalankan.")
        st.stop()

    pred = model.predict(x_test, verbose=0)

    dummy_pred = np.zeros((len(pred), len(FITUR)))
    dummy_pred[:, 2] = pred.flatten()
    pred_inv = scaler.inverse_transform(dummy_pred)[:, 2]

    dummy_actual = np.zeros((len(y_test), len(FITUR)))
    dummy_actual[:, 2] = y_test.flatten()
    actual_inv = scaler.inverse_transform(dummy_actual)[:, 2]

    tanggal = df["Tanggal"].iloc[TIMESTEP: TIMESTEP + len(actual_inv)].reset_index(drop=True)

    hasil = pd.DataFrame({
        "Tanggal": tanggal,
        "Aktual RR": actual_inv,
        "Prediksi RR": pred_inv
    })

    st.subheader("📉 Hasil Prediksi Data Test")
    st.dataframe(hasil, use_container_width=True)

    rmse = np.sqrt(np.mean((actual_inv - pred_inv) ** 2))
    st.metric("RMSE", f"{rmse:.3f}")

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    ax.plot(hasil["Tanggal"], hasil["Aktual RR"], label="Aktual", linewidth=2.5)
    ax.plot(hasil["Tanggal"], hasil["Prediksi RR"], label="Prediksi", linestyle="--", linewidth=2.5)

    ax.set_title("Aktual vs Prediksi Curah Hujan")
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

    if None in (model, scaler, x_test, df):
        st.error("Pastikan semua tahap sudah dijalankan.")
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
    dummy[:, 2] = future_scaled
    future_inv = scaler.inverse_transform(dummy)[:, 2]

    tanggal_future = pd.date_range(start=df["Tanggal"].iloc[-1], periods=n + 1)[1:]

    hasil_future = pd.DataFrame({"Tanggal": tanggal_future, "Prediksi RR": future_inv})

    st.subheader("🔮 Prediksi Curah Hujan Masa Depan")
    st.dataframe(hasil_future, use_container_width=True)

    fig, ax = plt.subplots(figsize=(16, 6), dpi=140)
    ax.plot(tanggal_future, future_inv, marker="o", linewidth=2.5)
    ax.set_title("Prediksi Curah Hujan Masa Depan")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Curah Hujan (mm)")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
