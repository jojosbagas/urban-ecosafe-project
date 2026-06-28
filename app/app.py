import streamlit as st

# Setup Konfigurasi Halaman
st.set_page_config(page_title="Urban EcoSafe Dashboard", layout="wide", page_icon="🏙️")

# Judul Utama
st.title("🏙️ Urban EcoSafe")
st.markdown("**Dashboard Cerdas Pemantauan Keselamatan & Lingkungan Kota**")
st.divider()

# Sidebar untuk Navigasi Menu
with st.sidebar:
    st.header("Modul Sistem")
    menu = st.radio(
        "Pilih Fitur:",
        ["🚦 Klasifikasi Kecelakaan (Project 1)", "☁️ Forecasting Polusi Udara (Project 2)"]
    )
    st.info("Project UAS Data Mining - Kelompok X")

# ---------------------------------------------------------
# MENU 1: KLASIFIKASI KECELAKAAN (PROJECT 1)
# ---------------------------------------------------------
if menu == "🚦 Klasifikasi Kecelakaan (Project 1)":
    st.header("Prediksi Tingkat Keparahan Kecelakaan")
    st.write("Masukkan kondisi lingkungan untuk memprediksi tingkat keparahan (Severity).")
    
    # Membagi layout menjadi 2 kolom
    col1, col2 = st.columns(2)
    
    with col1:
        suhu = st.number_input("Suhu (Fahrenheit)", min_value=-20.0, max_value=120.0, value=75.0)
        kelembaban = st.number_input("Kelembaban (%)", min_value=0.0, max_value=100.0, value=50.0)
    
    with col2:
        visibilitas = st.number_input("Jarak Pandang (mil)", min_value=0.0, max_value=20.0, value=10.0)
        cuaca = st.selectbox("Kondisi Cuaca", ["Clear", "Rain", "Snow", "Fog"])
        
    if st.button("Prediksi Keparahan", type="primary"):
        # Nanti bagian ini diganti dengan model.predict() dari Ivan
        st.success("Tingkat Keparahan Prediksi: Level 3 (Berdampak Signifikan pada Lalu Lintas)")

# ---------------------------------------------------------
# MENU 2: FORECASTING POLUSI UDARA (PROJECT 2)
# ---------------------------------------------------------
elif menu == "☁️ Forecasting Polusi Udara (Project 2)":
    st.header("Peramalan Kualitas Udara (PM2.5) dengan LSTM")
    st.write("Masukkan riwayat data polusi udara 7 hari terakhir.")
    
    # Input data 7 hari berjejer ke samping
    cols = st.columns(7)
    input_data = []
    
    for i in range(7):
        with cols[i]:
            val = st.number_input(f"H-{7-i}", value=50.0 + (i*2), step=1.0)
            input_data.append(val)
            
    if st.button("Prediksi Polusi Esok Hari", type="primary"):
        # Nanti bagian ini diganti dengan model.predict() dari Charis
        st.warning("Prediksi PM2.5 Besok: 78.4 (Kualitas Udara Sedang)")