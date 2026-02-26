import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import load_model

# ================= CONFIG =================
TIMESTEP = 25

FITUR = [
    "Curah Hujan (RR)",
    "Kelembapan Rata-rata (RH_avg)",
    "Suhu Rata-rata (T_avg)"
]

TARGET_INDEX = 2  # suhu

# ================= LOAD MODEL & SCALER =================
@st.cache_resource
def load_all():
    model = load_model("model_lstm.h5")
    scaler_X = joblib.load("scaler_X.save")
    scaler_y = joblib.load("scaler_y.save")
    return model, scaler_X, scaler_y

model, scaler_X, scaler_y = load_all()

# ================= FUNCTION =================
def create_dataset(data, timestep=25):
    X, y = [], []
    for i in range(len(data) - timestep):
        X.append(data[i:i+timestep])
        y.append(data[i+timestep, TARGET_INDEX])
    return np.array(X), np.array(y)

# ================= UI =================
st.title("Prediksi Suhu LSTM")

uploaded = st.file_uploader("Upload Data Excel", type=["xlsx"])

if uploaded:

    df = pd.read_excel(uploaded)

    st.write("Data Awal")
    st.dataframe(df.head())

    # ================= NORMALISASI =================
    scaled = scaler_X.transform(df[FITUR])
    X_all, y_all = create_dataset(scaled, TIMESTEP)

    # ================= SPLIT =================
    split = int(len(X_all) * 0.8)

    X_train, X_test = X_all[:split], X_all[split:]
    y_train, y_test = y_all[:split], y_all[split:]

    # ================= PREDIKSI TEST =================
    pred_test = model.predict(X_test)

    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1,1))[:,0]
    pred_test_inv = scaler_y.inverse_transform(pred_test)[:,0]

    # ================= PLOT =================
    st.subheader("Grafik Prediksi vs Aktual")

    plt.figure(figsize=(14,6))
    plt.plot(y_test_inv, label="Aktual")
    plt.plot(pred_test_inv, label="Prediksi")
    plt.legend()
    plt.grid(True)

    st.pyplot(plt)

    # ================= RMSE =================
    rmse = np.sqrt(np.mean((y_test_inv - pred_test_inv)**2))
    st.success(f"RMSE: {rmse:.4f}")

    # ================= PREDIKSI MASA DEPAN =================
    st.subheader("Prediksi Masa Depan")

    jumlah_hari = st.number_input("Jumlah Hari Diprediksi", 1, 365, 30)

    if st.button("Prediksi"):

        last_window = scaled[-TIMESTEP:]
        future_preds = []

        current_window = last_window.copy()

        for _ in range(jumlah_hari):

            pred = model.predict(current_window.reshape(1, TIMESTEP, len(FITUR)))
            future_preds.append(pred[0,0])

            new_row = current_window[-1].copy()
            new_row[TARGET_INDEX] = pred[0,0]

            current_window = np.vstack([current_window[1:], new_row])

        future_preds = scaler_y.inverse_transform(
            np.array(future_preds).reshape(-1,1)
        )[:,0]

        # Plot future
        plt.figure(figsize=(14,6))
        plt.plot(future_preds, marker="o", label="Prediksi Masa Depan")
        plt.legend()
        plt.grid(True)

        st.pyplot(plt)

        df_future = pd.DataFrame({
            "Hari": np.arange(1, jumlah_hari+1),
            "Prediksi Suhu": future_preds
        })

        st.dataframe(df_future)
