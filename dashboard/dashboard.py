import streamlit as st
import pandas as pd
import numpy as np
import os

# Impor komponen modular dari folder components
from components.overview_tab import render_overview
from components.eda_tab import render_eda
from components.business_questions import render_bq
from components.profiling_tab import render_profiling
from components.comparative_tab import render_comparative
from components.explorer_tab import render_explorer
from components.predict_tab import render_predict

# Konfigurasi Halaman Utama
st.set_page_config(
    layout="wide", 
    page_title="EduPredict AI", 
    page_icon="🎓"
)

# Palet warna resmi sesuai spesifikasi
COLOR_MAP = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}

@st.cache_data
def load_historical_data():
    """Memuat data historis untuk visualisasi"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "processed", "edupredict_cleaned.csv")
    
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        # Jika file tidak ditemukan, buat dummy data untuk keperluan pengembangan dan demo
        np.random.seed(42)
        n_rows = 6607
        dummy_data = {
            'Hours_Studied': np.random.randint(1, 30, n_rows),
            'Attendance': np.random.randint(50, 100, n_rows),
            'Sleep_Hours': np.random.randint(4, 10, n_rows),
            'Previous_Scores': np.random.randint(40, 100, n_rows),
            'Tutoring_Sessions': np.random.randint(0, 5, n_rows),
            'Physical_Activity': np.random.randint(0, 7, n_rows),
            'Parental_Involvement': np.random.choice(['Low', 'Medium', 'High'], n_rows),
            'Access_to_Resources': np.random.choice(['Low', 'Medium', 'High'], n_rows),
            'Motivation_Level': np.random.choice(['Low', 'Medium', 'High'], n_rows),
            'Internet_Access': np.random.choice(['Yes', 'No'], n_rows, p=[0.85, 0.15]),
            'Family_Income': np.random.choice(['Low', 'Medium', 'High'], n_rows, p=[0.4, 0.4, 0.2]),
            'Teacher_Quality': np.random.choice(['Low', 'Medium', 'High'], n_rows),
            'Peer_Influence': np.random.choice(['Negative', 'Neutral', 'Positive'], n_rows),
            'Parental_Education_Level': np.random.choice(['High School', 'Bachelor', 'Master'], n_rows),
            'Gender': np.random.choice(['Male', 'Female'], n_rows),
            'School_Type': np.random.choice(['Public', 'Private'], n_rows),
            'Learning_Disabilities': np.random.choice(['Yes', 'No'], n_rows, p=[0.05, 0.95]),
            'Distance_from_Home': np.random.choice(['Near', 'Moderate', 'Far'], n_rows),
            'Extracurricular_Activities': np.random.choice(['Yes', 'No'], n_rows),
            'Exam_Score': np.random.randint(30, 100, n_rows),
            'Risk_Category': np.random.choice(['High', 'Medium', 'Low'], n_rows, p=[0.01, 0.83, 0.16])
        }
        df_dummy = pd.DataFrame(dummy_data)
        df_dummy.loc[df_dummy['Risk_Category'] == 'High', 'Family_Income'] = np.random.choice(['Low', 'Medium'], size=len(df_dummy[df_dummy['Risk_Category'] == 'High']), p=[0.55, 0.45])
        df_dummy.loc[df_dummy['Risk_Category'] == 'High', 'Motivation_Level'] = np.random.choice(['Low', 'Medium'], size=len(df_dummy[df_dummy['Risk_Category'] == 'High']), p=[0.53, 0.47])
        return df_dummy

df_raw = load_historical_data()

# sidebar navigasi utama
st.sidebar.title("🎓 EduPredict AI")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Pilih Halaman Navigasi:",
    [
        "Overview", 
        "Exploratory Data Analysis", 
        "Business Questions", 
        "Profil High Risk", 
        "Uji Komparatif (Attendance)", 
        "Data Explorer",
        "Prediksi Risiko Real-Time"
    ]
)
st.sidebar.markdown("---")

# filter global untuk halaman Overview, EDA, dan Business Questions agar tetap konsisten dalam analisis
if page in ["Overview", "Exploratory Data Analysis", "Business Questions"]:
    st.sidebar.header("Filter Global")
    global_risk = st.sidebar.multiselect(
        "Kategori Risiko:", options=['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low']
    )
    global_school = st.sidebar.selectbox(
        "Tipe Sekolah:", options=["All"] + list(df_raw['School_Type'].unique()), index=0
    )
    
    df_filtered = df_raw[df_raw['Risk_Category'].isin(global_risk)]
    if global_school != "All":
        df_filtered = df_filtered[df_filtered['School_Type'] == global_school]
        
    st.sidebar.metric("Data Aktif:", f"{len(df_filtered)} / {len(df_raw)} siswa")
    
    if len(df_filtered) == 0:
        st.title(page)
        st.warning("⚠️ Tidak ada data historis yang sesuai dengan kombinasi filter global.")
        st.stop()
else:
    df_filtered = df_raw

# render halaman sesuai pilihan navigasi
if page == "Overview":
    render_overview(df_filtered, COLOR_MAP)
elif page == "Exploratory Data Analysis":
    render_eda(df_filtered, COLOR_MAP)
elif page == "Business Questions":
    render_bq(df_filtered, COLOR_MAP)
elif page == "Profil High Risk":
    render_profiling(df_raw, COLOR_MAP)
elif page == "Uji Komparatif (Attendance)":
    render_comparative(df_raw)
elif page == "Data Explorer":
    render_explorer(df_raw)
elif page == "Prediksi Risiko Real-Time":
    render_predict(COLOR_MAP)