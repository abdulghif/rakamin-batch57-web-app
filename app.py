import streamlit as st
import pandas as pd
import plotly.figure_factory as ff
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from pathlib import Path

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Aplikasi Prediksi Iris",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Path untuk Data dan Model ---
DATA_PATH = Path("data/iris.csv")
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "iris_classifier.joblib"

# --- Caching Functions ---
@st.cache_data
def load_data(path):
    """Memuat dataset Iris dari path yang diberikan."""
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di '{path}'. Pastikan file 'iris.csv' ada di dalam folder 'data'.")
        return None

@st.cache_resource
def load_model(path):
    """Memuat model yang sudah di-training."""
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        return None

# --- Halaman Training Model ---
def training_page():
    """Halaman untuk melatih dan mengevaluasi model."""
    st.title("🔬 Pelatihan Model Klasifikasi Iris")
    st.markdown("""
        Di halaman ini, Anda dapat melatih model *Random Forest Classifier* menggunakan dataset Iris.
        Klik tombol di bawah untuk memulai proses.
    """)

    df = load_data(DATA_PATH)
    if df is None:
        st.stop()

    if st.button("🚀 Latih Model Sekarang!", type="primary", use_container_width=True):
        with st.spinner("Mohon tunggu, model sedang dilatih..."):
            # 1. Pemisahan Fitur (X) dan Target (y)
            X = df.drop(['Id', 'Species'], axis=1)
            y = df['Species']
            class_names = y.unique().tolist()
            
            # 2. Pembagian Data Training dan Testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            # 3. Inisialisasi dan Pelatihan Model
            rfc = RandomForestClassifier(n_estimators=100, random_state=42)
            rfc.fit(X_train, y_train)
            
            # 4. Prediksi
            y_pred = rfc.predict(X_test)
            
            # 5. Evaluasi Performa
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            # 6. Simpan Model
            MODEL_DIR.mkdir(exist_ok=True)
            joblib.dump(rfc, MODEL_PATH)

        st.success("✅ Model berhasil dilatih dan disimpan!")
        st.balloons()
        
        st.subheader("📊 Performa Model")
        
        # Tampilkan Metrik
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi", f"{accuracy:.2%}")
        col2.metric("Presisi", f"{precision:.3f}")
        col3.metric("Recall", f"{recall:.3f}")
        col4.metric("F1-Score", f"{f1:.3f}")

        # Tampilkan Confusion Matrix
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=class_names)
        fig_cm = ff.create_annotated_heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True
        )
        fig_cm.update_layout(
            title_text='<b>Confusion Matrix</b>',
            xaxis_title='Prediksi Label',
            yaxis_title='Label Aktual',
            margin=dict(t=50, l=0, r=0, b=0)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        with st.expander("Lihat detail data yang digunakan"):
            st.write("Contoh Data Asli:")
            st.dataframe(df.sample(5))
            st.write("Dimensi X_train:", X_train.shape)
            st.write("Dimensi X_test:", X_test.shape)

# --- Halaman Prediksi ---
def prediction_page():
    """Halaman untuk melakukan prediksi bunga Iris."""
    st.title("🌸 Prediksi Spesies Bunga Iris")
    st.markdown("""
        Masukkan nilai fitur bunga Iris di bawah ini untuk memprediksi spesiesnya.
        Pastikan Anda telah melatih model di halaman **Training** terlebih dahulu.
    """)
    
    # Muat model
    rfc = load_model(MODEL_PATH)
    
    if rfc is None:
        st.warning("Model belum dilatih. Silakan pergi ke halaman 'Training' untuk melatih model terlebih dahulu.", icon="⚠️")
        st.stop()
    
    # Form untuk input pengguna
    with st.form("prediction_form"):
        st.subheader("Masukkan Fitur Bunga:")
        
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.number_input("Panjang Sepal (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1, format="%.1f")
            sepal_width = st.number_input("Lebar Sepal (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1, format="%.1f")
        with col2:
            petal_length = st.number_input("Panjang Petal (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1, format="%.1f")
            petal_width = st.number_input("Lebar Petal (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1, format="%.1f")
            
        submitted = st.form_submit_button("🔮 Prediksi!", use_container_width=True, type="primary")

    if submitted:
        # Buat data input untuk prediksi
        X_pred = [[sepal_length, sepal_width, petal_length, petal_width]]
        
        # Lakukan prediksi
        prediction = rfc.predict(X_pred)[0]
        prediction_proba = rfc.predict_proba(X_pred)
        
        st.success(f"**Hasil Prediksi:** Bunga ini kemungkinan besar adalah **{prediction}**.")
        
        st.write("Probabilitas Prediksi:")
        prob_df = pd.DataFrame(prediction_proba, columns=rfc.classes_, index=["Probabilitas"])
        st.dataframe(prob_df)

# --- Navigasi Aplikasi ---
pg = st.navigation([
    st.Page(training_page, title="Training", icon="🔬"),
    st.Page(prediction_page, title="Prediksi", icon="🌸"),
])
pg.run()
