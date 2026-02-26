import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Prediksi Curah Hujan", layout="wide")

st.title("Aplikasi Prediksi Curah Hujan")

menu = st.sidebar.selectbox(
    "Menu",
    ["Upload Data", "Prediksi & Evaluasi", "Prediksi Masa Depan"]
)

# =============================
# UPLOAD DATA
# =============================
if menu == "Upload Data":

    st.header("Upload Dataset")

    file = st.file_uploader("Upload file CSV", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        st.success("Data berhasil diupload")
        st.dataframe(data)

        st.session_state["data"] = data


# =============================
# PREDIKSI & EVALUASI
# =============================
elif menu == "Prediksi & Evaluasi":

    st.header("Prediksi & Evaluasi Model")

    if "data" not in st.session_state:
        st.warning("Silakan upload data dulu")
    else:

        data = st.session_state["data"]

        target_column = st.selectbox(
            "Pilih Kolom Target (Curah Hujan)",
            data.columns
        )

        if st.button("Proses Prediksi"):

            df = data.copy()

            # Buat fitur lag sederhana
            df["lag1"] = df[target_column].shift(1)
            df["lag2"] = df[target_column].shift(2)
            df["lag3"] = df[target_column].shift(3)

            df = df.dropna()

            X = df[["lag1", "lag2", "lag3"]]
            y = df[target_column]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            model = RandomForestRegressor(n_estimators=200, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            st.metric("Nilai RMSE", f"{rmse:.3f}")

            # =============================
            # Grafik Aktual vs Prediksi
            # =============================
            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(y_test.values, label="Aktual", marker="o")
            ax.plot(y_pred, label="Prediksi", marker="s")

            ax.set_title("Perbandingan Aktual vs Prediksi")
            ax.set_xlabel("Index Data")
            ax.set_ylabel("Curah Hujan")
            ax.legend()

            st.pyplot(fig)

            # Simpan model & data
            st.session_state["model"] = model
            st.session_state["last_data"] = df
            st.session_state["target"] = target_column


# =============================
# PREDIKSI MASA DEPAN
# =============================
elif menu == "Prediksi Masa Depan":

    st.header("Prediksi Curah Hujan Masa Depan")

    if "model" not in st.session_state:
        st.warning("Silakan lakukan prediksi terlebih dahulu")
    else:

        model = st.session_state["model"]
        df = st.session_state["last_data"]
        target_column = st.session_state["target"]

        n_future = st.number_input(
            "Jumlah Periode Prediksi",
            min_value=1,
            max_value=24,
            value=6
        )

        if st.button("Prediksi"):

            last_values = list(df[target_column].values[-3:])
            predictions = []

            for _ in range(n_future):

                X_input = np.array(last_values[-3:]).reshape(1, -1)
                pred = model.predict(X_input)[0]

                predictions.append(pred)
                last_values.append(pred)

            future_df = pd.DataFrame({
                "Periode": range(1, n_future + 1),
                "Prediksi Curah Hujan": predictions
            })

            st.subheader("Tabel Hasil Prediksi")
            st.dataframe(future_df, use_container_width=True)

            # =============================
            # Grafik Prediksi Masa Depan
            # =============================
            fig2, ax2 = plt.subplots(figsize=(12, 5))

            ax2.plot(future_df["Periode"],
                     future_df["Prediksi Curah Hujan"],
                     marker="o",
                     label="Prediksi")

            ax2.set_title("Prediksi Curah Hujan Masa Depan")
            ax2.set_xlabel("Periode")
            ax2.set_ylabel("Curah Hujan")
            ax2.legend()

            st.pyplot(fig2)

            # Download
            csv = future_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Hasil Prediksi",
                data=csv,
                file_name="prediksi_curah_hujan.csv",
                mime="text/csv"
            )
