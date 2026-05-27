import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf

@st.cache_resource
def load_ml_artifacts():
    """Memuat model keras dan semua object pkl pendukung menggunakan caching resource"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "..", "models")
    
    artifacts = {}
    try:
        # Memuat Keras Model
        artifacts['model'] = tf.keras.models.load_model(os.path.join(model_dir, "edupredict_multioutput.keras"))
        
        # Memuat berkas pickle pendukung preprocessing
        with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
            artifacts['scaler'] = pickle.load(f)
        with open(os.path.join(model_dir, "label_encoders.pkl"), "rb") as f:
            artifacts['label_encoders'] = pickle.load(f)
        with open(os.path.join(model_dir, "feature_cols.pkl"), "rb") as f:
            artifacts['feature_cols'] = pickle.load(f)
        with open(os.path.join(model_dir, "risk_map.pkl"), "rb") as f:
            artifacts['risk_map'] = pickle.load(f)
            
        artifacts['status'] = "Success"
    except Exception as e:
        artifacts['status'] = f"Error: {str(e)}"
        
    return artifacts

def render_predict(color_map):
    st.title("🔮 Prediksi Risiko Akademik Siswa — Real-Time")
    st.markdown("Masukkan indikator performa harian siswa di bawah ini untuk mengestimasikan kategori risiko akademis harian secara instan menggunakan kecerdasan buatan.")

    # Ambil artifak model hasil latih
    artifacts = load_ml_artifacts()
    
    if "Error" in artifacts['status']:
        st.error("❌ Gagal memuat artifak model ML di folder 'models/'. Pastikan folder dan file model sudah Anda letakkan dengan benar.")
        st.info("Detail Eror: " + artifacts['status'])
        return
        
    # Ambil list referensi kolom fitur agar urutannya tidak tertukar saat scaling
    feature_cols = artifacts['feature_cols']
    label_encoders = artifacts['label_encoders']
    scaler = artifacts['scaler']
    model = artifacts['model']
    risk_map = artifacts['risk_map']
    
    # Pastikan kita memiliki pemetaan dua arah yang konsisten: label ke index dan index ke label
    if isinstance(list(risk_map.keys())[0], str):
        label_to_idx = risk_map
        idx_to_label = {v: k for k, v in risk_map.items()}
    else:
        idx_to_label = risk_map
        label_to_idx = {v: k for k, v in risk_map.items()}

    # Membuat Form Input untuk Guru/Dosen
    with st.form("prediction_form"):
        st.markdown("### 📝 Form Parameter Siswa")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### 🔢 Faktor Akademis & Kedisiplinan")
            hours_studied = st.slider("Jam Belajar Per Minggu (Hours Studied):", 1, 50, 15)
            attendance = st.slider("Persentase Kehadiran Kelas (Attendance %):", 0, 100, 85)
            previous_scores = st.slider("Nilai Ujian Sebelumnya (Previous Scores):", 0, 100, 75)
            tutoring_sessions = st.number_input("Jumlah Sesi Bimbingan (Tutoring Sessions):", min_value=0, max_value=20, value=1, step=1)
            sleep_hours = st.slider("Rata-rata Jam Tidur Harian (Sleep Hours):", 3, 12, 7)
            physical_activity = st.slider("Frekuensi Olahraga Per Minggu (Physical Activity):", 0, 7, 3)

        with c2:
            st.markdown("#### 🔠 Faktor Psikososial & Lingkungan")
            motivation_level = st.selectbox("Tingkat Motivasi Siswa:", options=list(label_encoders['Motivation_Level'].classes_), index=1)
            parental_involvement = st.selectbox("Keterlibatan Orang Tua:", options=list(label_encoders['Parental_Involvement'].classes_), index=1)
            access_resources = st.selectbox("Akses Fasilitas Belajar:", options=list(label_encoders['Access_to_Resources'].classes_), index=1)
            teacher_quality = st.selectbox("Kualitas Tenaga Pengajar:", options=list(label_encoders['Teacher_Quality'].classes_), index=1)
            family_income = st.selectbox("Tingkat Pendapatan Keluarga:", options=list(label_encoders['Family_Income'].classes_), index=1)
            parental_edu = st.selectbox("Tingkat Pendidikan Orang Tua:", options=list(label_encoders['Parental_Education_Level'].classes_), index=1)
            peer_influence = st.selectbox("Pengaruh Teman Sebaya (Peer Influence):", options=list(label_encoders['Peer_Influence'].classes_), index=1)
            internet_access = st.radio("Akses Jaringan Internet di Rumah:", options=list(label_encoders['Internet_Access'].classes_), index=1, horizontal=True)

        submit_btn = st.form_submit_button("🔮 Hitung Estimasi Risiko Akademik")

    if submit_btn:
        # 1. Tampung input ke dalam dictionary awal
        input_data = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Parental_Involvement': parental_involvement,
            'Access_to_Resources': access_resources,
            'Sleep_Hours': sleep_hours,
            'Previous_Scores': previous_scores,
            'Motivation_Level': motivation_level,
            'Internet_Access': internet_access,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': family_income,
            'Teacher_Quality': teacher_quality,
            'Peer_Influence': peer_influence,
            'Physical_Activity': physical_activity,
            'Parental_Education_Level': parental_edu
        }
        
        # 2. Konversi ke DataFrame satu baris
        input_df = pd.DataFrame([input_data])
        
        # 3. Urutkan kolom DataFrame dengan ketat sesuai susunan feature_cols.pkl agar tidak tertukar saat scaling
        input_df = input_df[feature_cols]
        
        # 4. Lakukan transformasi Label Encoding pada fitur kategorikal menggunakan encoder berkas pkl asli
        cat_cols_to_encode = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level']
        for col in cat_cols_to_encode:
            encoder = label_encoders[col]
            input_df[col] = encoder.transform(input_df[col])
            
        # 5. Lakukan standarisasi data numerik menggunakan objek Scaler bawaan pkl
        input_scaled = scaler.transform(input_df)
        
        # 6. Jalankan Prediksi menggunakan Neural Network Model (.keras)
        preds = model.predict(input_scaled)
        
        # Penanganan Model Multioutput: Model Keras multioutput mengembalikan list array output.
        # Kita deteksi array mana yang berisi 3 kelas probabilitas klasifikasi kategori risiko (Low, Med, High)
        if isinstance(preds, list):
            # Cari output array yang memiliki dimensi kolom = 3 kelompok klasifikasi kelas target
            risk_pred_array = [p for p in preds if p.shape[1] == 3][0]
        else:
            risk_pred_array = preds
            
        # Ambil indeks kelas probabilitas tertinggi menggunakan argmax
        pred_idx = np.argmax(risk_pred_array[0])
        probabilities = risk_pred_array[0]
        
        # Map indeks kembali menjadi String label nama kategori risiko (High / Medium / Low)
        final_risk_label = idx_to_label.get(pred_idx, "Unknown")
        
        # Tampilkan Hasil Visualisasi Prediksi yang Menarik
        st.markdown("---")
        st.markdown("### 🎯 Hasil Analisis Prediksi Kecerdasan Buatan")
        
        # Tampilkan box warna dinamis sesuai dengan tingkat bahaya kategori risikonya
        chosen_color = color_map.get(final_risk_label, "#7f8c8d")
        
        st.markdown(
            f"""
            <div style="background-color:{chosen_color}; padding:20px; border-radius:10px; text-align:center;">
                <h2 style="color:white; margin:0;">KATEGORI RISIKO: {final_risk_label.upper()} RISK</h2>
                <p style="color:white; margin:5px 0 0 0; font-size:16px;">Sistem mendeteksi tingkat kerentanan akademik siswa berada di kelas risiko {final_risk_label}.</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        # Tampilkan Distribusi Probabilitas Klasifikasi Kelas
        st.markdown("<br>", unsafe_allow_html=True)
        c_p1, c_p2, c_p3 = st.columns(3)
        # Menyesuaikan mapping probabilitas secara aman berdasarkan urutan alphabet standard encoder classes
        c_p1.metric("Probabilitas Low Risk", f"{probabilities[label_to_idx['Low']]*100:.2f}%")
        c_p2.metric("Probabilitas Medium Risk", f"{probabilities[label_to_idx['Medium']]*100:.2f}%")
        c_p3.metric("Probabilitas High Risk", f"{probabilities[label_to_idx['High']]*100:.2f}%")
        
        # Tampilkan Rekomendasi Aksi Intervensi Otomatis Berdasarkan Tingkat Risiko
        st.markdown("#### 💡 Rekomendasi Strategi Intervensi Pendidik:")
        if final_risk_label == 'High':
            st.error("🚨 **Intervensi Prioritas Utama:** Siswa memerlukan pendampingan belajar intensif empat mata harian. Segera adakan pertemuan segitiga bersama wali murid, optimalkan sesi bimbingan konseling akademik khusus, serta lakukan penyesuaian target capaian nilai harian.")
        elif final_risk_label == 'Medium':
            st.warning("⚠️ **Intervensi Preventif:** Tingkatkan frekuensi keterlibatan siswa dalam forum kelompok belajar, dorong siswa untuk menambah 2-3 jam sesi belajar mandiri terstruktur per minggu, serta lakukan pemantauan grafik absensi kehadiran di kelas secara berkala.")
        else:
            st.success("✅ **Strategi Preservasi Performa:** Siswa berada di jalur akademis yang aman dan stabil. Berikan apresiasi motivasi berkala agar siswa konsisten mempertahankan ritme kedisiplinan belajar dan tingkat kehadiran saat ini.")