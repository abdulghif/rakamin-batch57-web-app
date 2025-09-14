import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
from pathlib import Path
import time

# --- Asumsi Struktur Proyek ---
# Pastikan Anda memiliki skrip-skrip ini di dalam folder 'scripts'
# agar import di bawah ini berfungsi dengan baik.
from scripts.analysis import run_analysis
from scripts.generate_dummy_data import generate_dummy_data_func
from scripts.train_model import train_churn_model

# --- Placeholder Functions (Jika skrip asli tidak tersedia) ---
# Hapus atau komentari bagian ini jika Anda sudah memiliki skripnya
# def run_analysis():
#     st.info("Halaman Analisis Data sedang dalam pengembangan.")
#     st.image("https://placehold.co/800x400/EEE/31343C?text=Grafik+Analisis+Data", use_column_width=True)

# def generate_dummy_data_func():
#     Path("data").mkdir(exist_ok=True)
#     data = {
#         'age': [35, 45, 25, 55, 30],
#         'gender': ['Female', 'Male', 'Male', 'Female', 'Male'],
#         'purchase_amount': [100, 200, 50, 300, 150],
#         'tenure': [12, 24, 6, 36, 18],
#         'churn': [0, 1, 0, 1, 0]
#     }
#     df = pd.DataFrame(data)
#     df.to_csv("data/customer_data.csv", index=False)
#     time.sleep(1) # Simulasi proses
#     return "Data dummy berhasil dibuat."

# def train_churn_model():
#     Path("models").mkdir(exist_ok=True)
#     # Ini adalah model palsu, hanya untuk demonstrasi
#     from sklearn.linear_model import LogisticRegression
#     model = LogisticRegression()
#     # Simulasikan model yang sudah dilatih
#     with open("models/churn_model.pkl", 'wb') as f:
#         pickle.dump(model, f)
#     time.sleep(2) # Simulasi proses training
#     return model
# --- Akhir Placeholder Functions ---


# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="👋",
    layout="wide"
)

# --- Path & Setup ---
DATA_PATH = Path("data/customer_data.csv")
MODEL_PATH = Path("models/churn_model.pkl")

# --- Fungsi Helper ---
@st.cache_resource
def load_model(path):
    """Memuat model prediksi churn dari file pickle."""
    if not path.exists():
        return None
    with open(path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict_churn(model, age, gender, purchase_amount, tenure):
    """Melakukan prediksi churn berdasarkan input pengguna."""
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'purchase_amount': [purchase_amount],
        'tenure': [tenure]
    })
    # Placeholder: model asli Anda mungkin butuh encoding untuk 'gender'
    # contoh: input_data['gender'] = input_data['gender'].map({'Male': 1, 'Female': 0})
    
    # Untuk model placeholder, kita akan generate probabilitas acak
    # import numpy as np
    # prediction_proba = np.random.rand()
    # prediction = 1 if prediction_proba >= 0.5 else 0
    
    # Jika menggunakan model asli, baris di atas diganti dengan:
    prediction_proba = model.predict_proba(input_data)[0][1]
    prediction = 1 if prediction_proba >= 0.5 else 0
    
    return prediction, prediction_proba

# --- Definisi Halaman ---

def prediction_page():
    """Halaman utama untuk prediksi churn."""
    st.title("🔮 Aplikasi Prediksi Churn Pelanggan")
    st.markdown("""
    Selamat datang! Aplikasi ini menggunakan *Machine Learning* untuk memprediksi kemungkinan seorang pelanggan akan berhenti berlangganan (churn).
    Masukkan data pelanggan di bawah ini untuk melihat hasilnya.
    """)

    model = load_model(MODEL_PATH)

    if model is None:
        st.error("❌ Model tidak ditemukan. Silakan pergi ke halaman **'Persiapan Data & Model'** untuk melatih model terlebih dahulu.", icon="🚨")
        st.stop()

    col1, col2 = st.columns([1, 1])

    with col1:
        with st.form("prediction_form"):
            st.header("📋 Input Data Pelanggan")
            
            age = st.slider('Usia', min_value=18, max_value=80, value=35, step=1)
            gender = st.selectbox('Gender', options=['Male', 'Female'])
            purchase_amount = st.number_input('Jumlah Pembelian (Rp)', min_value=0, value=500000, step=100000)
            tenure = st.slider('Lama Berlangganan (bulan)', min_value=1, max_value=120, value=24, step=1)
            
            predict_button = st.form_submit_button('✨ Prediksi Churn!', use_container_width=True, type="primary")

    with col2:
        st.header("📊 Hasil Prediksi")
        if not predict_button:
            st.info("Hasil prediksi akan muncul di sini setelah Anda menekan tombol.")
        else:
            prediction, probability = predict_churn(model, age, gender, purchase_amount, tenure)
            
            # Gauge Chart dengan Plotly
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilitas Churn", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#d62728" if prediction == 1 else "#2ca02c"},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(44, 160, 44, 0.2)'},
                        {'range': [50, 100], 'color': 'rgba(214, 39, 40, 0.2)'}],
                }))
            fig.update_layout(height=250, margin=dict(t=40, b=40, l=40, r=40))
            st.plotly_chart(fig, use_container_width=True)

            if prediction == 1:
                st.error(f"**PELANGGAN BERPOTENSI CHURN** (Probabilitas: {probability:.1%})", icon="🔥")
                with st.expander("Lihat Rekomendasi Tindakan"):
                    st.write("""
                    - **Hubungi Segera:** Lakukan kontak personal untuk memahami kebutuhan dan masalah yang dihadapi.
                    - **Tawarkan Insentif:** Berikan diskon khusus, bonus, atau paket retensi.
                    - **Minta Feedback:** Gali masukan mengenai produk/layanan untuk perbaikan.
                    """)
            else:
                st.success(f"**PELANGGAN KEMUNGKINAN LOYAL** (Probabilitas Churn: {probability:.1%})", icon="💚")
                with st.expander("Lihat Rekomendasi Tindakan"):
                     st.write("""
                    - **Jaga Hubungan:** Pertahankan komunikasi yang baik dan rutin.
                    - **Program Loyalitas:** Tawarkan keuntungan eksklusif bagi pelanggan setia.
                    - **Upselling/Cross-selling:** Perkenalkan produk atau layanan lain yang relevan.
                    """)


def analysis_page():
    """Halaman untuk menampilkan analisis data."""
    st.title("📈 Analisis Data Pelanggan")
    st.markdown("Halaman ini menampilkan visualisasi dan analisis dari data pelanggan yang ada.")
    
    if not DATA_PATH.exists():
        st.warning("Data belum tersedia. Silakan generate data di halaman 'Persiapan Data & Model'.", icon="⚠️")
        st.stop()
    
    run_analysis() # Memanggil fungsi dari skrip analisis

def preparation_page():
    """Halaman untuk persiapan data dan training model."""
    st.title("⚙️ Persiapan Data & Model")
    st.markdown("Gunakan halaman ini untuk membuat data dummy dan melatih model prediksi churn.")

    # Status Cek
    data_exists = DATA_PATH.exists()
    model_exists = MODEL_PATH.exists()
    
    st.subheader("Status Saat Ini")
    status_col1, status_col2 = st.columns(2)
    status_col1.metric("Status Data", "✅ Tersedia" if data_exists else "❌ Kosong")
    status_col2.metric("Status Model", "✅ Terlatih" if model_exists else "❌ Belum Ada")

    st.divider()

    # Langkah 1: Generate Data
    with st.container(border=True):
        st.subheader("Langkah 1: Generate Data Dummy")
        st.write("Klik tombol di bawah ini untuk membuat file `customer_data.csv` baru di folder `data/`.")
        if st.button("Generate Data", use_container_width=True):
            with st.spinner('Membuat data dummy...'):
                result = generate_dummy_data_func()
                st.success(f"✅ {result}", icon="📄")
            if DATA_PATH.exists():
                df = pd.read_csv(DATA_PATH)
                st.write("Preview Data:")
                st.dataframe(df.head())
                st.rerun()

    # Langkah 2: Latih Model
    with st.container(border=True):
        st.subheader("Langkah 2: Latih Model Machine Learning")
        st.write("Gunakan data yang ada untuk melatih model. Model akan disimpan di `models/churn_model.pkl`.")
        if st.button("Latih Model", use_container_width=True, disabled=not data_exists):
            with st.spinner('Model sedang dilatih, mohon tunggu...'):
                train_churn_model()
                st.success("✅ Model berhasil dilatih dan disimpan!", icon="🤖")
                st.rerun()
        elif not data_exists:
            st.warning("Tombol Latih Model akan aktif setelah data digenerate.", icon="💡")

# --- Navigasi Aplikasi ---
pg = st.navigation({
    "Prediksi": [st.Page(prediction_page, title="Prediksi Churn", icon="🔮")],
    "Data": [
        st.Page(analysis_page, title="Analisis Data", icon="📊"),
        st.Page(preparation_page, title="Persiapan Data & Model", icon="⚙️")
    ]
})
pg.run()
