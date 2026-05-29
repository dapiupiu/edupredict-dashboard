import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tensorflow as tf
import requests

def get_genai_insight(input_data, risk_category, confidence, predicted_score=0.0):
    """Fungsi untuk mengambil insight dari API Railway GenAI berdasarkan input data siswa"""
    RAILWAY_URL = "https://edupredictaimlproduction.up.railway.app/api/v1/analyze/recommendations"

    payload = {
        "student_id": "STU-Realtime",
        "features": input_data,
        "prediction": {
            "risk_category": risk_category,
            "confidence": float(confidence),
            "predicted_exam_score": float(predicted_score)
        }
    }

    try:
        #kirim data input ke API Railway
        response = requests.post(RAILWAY_URL, json=payload, timeout=15)
        if response.status_code == 200:
            result = response.json()
            recs = result.get('recommendations', [])
            if recs:
                return "\n".join([f"- {r.get('text', '')}" for r in recs])
            return "Tidak ada rekomendasi yang diberikan oleh AI."
        else:
            return f"Error API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Gagal terhubung ke Gen AI: {str(e)}"

@st.cache_resource
def load_ml_artifacts():
    """Memuat model keras dan semua object pkl pendukung menggunakan caching resource"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "..", "..", "models")
    
    artifacts = {}
    try:
        # load model Keras (.keras) dengan caching resource untuk efisiensi
        artifacts['model'] = tf.keras.models.load_model(os.path.join(model_dir, "edupredict_multioutput.keras"))
        
        # load semua objek pkl pendukung: scaler, label encoders, feature columns, dan risk map
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

# konfigurasi batas distribusi data pelatihan untuk input clamping dan OOD detection
TRAINING_BOUNDS = {
    'Attendance':      (60, 100),
    'Hours_Studied':   (4,  36),
    'Previous_Scores': (50, 100),
    'Sleep_Hours':     (4,  10),
    'Tutoring_Sessions': (0, 7),
    'Physical_Activity': (0, 6),
}

def clamp_input(input_dict):
    """Membatasi nilai input ke dalam range yang pernah dilihat model saat training (input clamping)"""
    clamped = input_dict.copy()
    for col, (lo, hi) in TRAINING_BOUNDS.items():
        if col in clamped:
            clamped[col] = max(lo, min(hi, clamped[col]))
    return clamped

def check_ood(input_dict):
    """Mendeteksi apakah input berada di luar distribusi data (Out-of-Distribution)"""
    warnings = []
    if input_dict.get('Attendance', 85) < 60:
        warnings.append(f"Kehadiran {input_dict['Attendance']}% di bawah data training (min 60%)")
    if input_dict.get('Hours_Studied', 15) < 4:
        warnings.append(f"Jam belajar {input_dict['Hours_Studied']} di bawah data training (min 4)")
    if input_dict.get('Hours_Studied', 15) > 36:
        warnings.append(f"Jam belajar {input_dict['Hours_Studied']} di atas data training (max 36)")
    if input_dict.get('Previous_Scores', 75) < 50:
        warnings.append(f"Nilai sebelumnya {input_dict['Previous_Scores']} di bawah data training (min 50)")
    if input_dict.get('Sleep_Hours', 7) < 4:
        warnings.append(f"Jam tidur {input_dict['Sleep_Hours']} di bawah data training (min 4)")
    if input_dict.get('Sleep_Hours', 7) > 10:
        warnings.append(f"Jam tidur {input_dict['Sleep_Hours']} di atas data training (max 10)")
    if input_dict.get('Tutoring_Sessions', 1) > 7:
        warnings.append(f"Sesi bimbingan {input_dict['Tutoring_Sessions']} di atas data training (max 7)")
    if input_dict.get('Physical_Activity', 3) > 6:
        warnings.append(f"Aktivitas fisik {input_dict['Physical_Activity']} di atas data training (max 6)")
    return warnings

def render_predict(color_map):
    st.title("Prediksi Risiko Akademik Siswa Secara Real-Time")
    st.markdown("Masukkan indikator performa harian siswa di bawah ini untuk mengestimasikan kategori risiko akademis harian secara instan menggunakan kecerdasan buatan.")

    # memuat model ML dan objek pendukung dengan penanganan error yang jelas dan informatif untuk pengguna
    artifacts = load_ml_artifacts()
    
    if "Error" in artifacts['status']:
        st.error("❌ Gagal memuat artifak model ML di folder 'models/'. Pastikan folder dan file model sudah Anda letakkan dengan benar.")
        st.info("Detail Eror: " + artifacts['status'])
        return
        
    # ekstrak objek-objek penting dari artifak yang dimuat untuk digunakan dalam proses prediksi
    feature_cols = artifacts['feature_cols']
    label_encoders = artifacts['label_encoders']
    scaler = artifacts['scaler']
    model = artifacts['model']
    risk_map = artifacts['risk_map']
    
    # membuat mapping label ke indeks dan sebaliknya untuk interpretasi hasil prediksi model multioutput
    if isinstance(list(risk_map.keys())[0], str):
        label_to_idx = risk_map
        idx_to_label = {v: k for k, v in risk_map.items()}
    else:
        idx_to_label = risk_map
        label_to_idx = {v: k for k, v in risk_map.items()}

    # form input parameter siswa dengan validasi dan penjelasan yang jelas untuk setiap fitur yang diminta
    with st.form("prediction_form"):
        st.markdown("### Formulir Input Parameter Siswa")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("#### Faktor Akademis & Kedisiplinan")
            hours_studied = st.slider("Jam Belajar Per Minggu (Hours Studied):", 1, 50, 15)
            attendance = st.slider("Persentase Kehadiran Kelas (Attendance %):", 0, 100, 85)
            previous_scores = st.slider("Nilai Ujian Sebelumnya (Previous Scores):", 0, 100, 75)
            tutoring_sessions = st.number_input("Jumlah Sesi Bimbingan (Tutoring Sessions):", min_value=0, max_value=20, value=1, step=1)
            sleep_hours = st.slider("Rata-rata Jam Tidur Harian (Sleep Hours):", 3, 12, 7)
            physical_activity = st.slider("Frekuensi Olahraga Per Minggu (Physical Activity):", 0, 7, 3)

        with c2:
            st.markdown("#### Faktor Psikososial & Lingkungan")
            motivation_level = st.selectbox("Tingkat Motivasi Siswa:", options=list(label_encoders['Motivation_Level'].classes_), index=1)
            parental_involvement = st.selectbox("Keterlibatan Orang Tua:", options=list(label_encoders['Parental_Involvement'].classes_), index=1)
            access_resources = st.selectbox("Akses Fasilitas Belajar:", options=list(label_encoders['Access_to_Resources'].classes_), index=1)
            teacher_quality = st.selectbox("Kualitas Tenaga Pengajar:", options=list(label_encoders['Teacher_Quality'].classes_), index=1)
            family_income = st.selectbox("Tingkat Pendapatan Keluarga:", options=list(label_encoders['Family_Income'].classes_), index=1)
            parental_edu = st.selectbox("Tingkat Pendidikan Orang Tua:", options=list(label_encoders['Parental_Education_Level'].classes_), index=1)
            peer_influence = st.selectbox("Pengaruh Teman Sebaya (Peer Influence):", options=list(label_encoders['Peer_Influence'].classes_), index=1)
            internet_access = st.radio("Akses Jaringan Internet di Rumah:", options=list(label_encoders['Internet_Access'].classes_), index=1, horizontal=True)

        submit_btn = st.form_submit_button("🚀 Hitung Estimasi Risiko Akademik")

    if submit_btn:
        # kumpulkan input ke dalam dictionary untuk memudahkan konversi ke DataFrame
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
        
        # deteksi Out-of-Distribution (OOD) dan berikan peringatan kepada pengguna
        ood_warns = check_ood(input_data)
        if ood_warns:
            st.warning("⚠️ **Input di luar distribusi data pelatihan:**\n" + 
                       "\n".join([f"- {w}" for w in ood_warns]) +
                       "\n\n*Sistem secara otomatis menyesuaikan (clamping) input ke batas distribusi data pelatihan agar prediksi tetap reliabel.*")
        
        # terapkan input clamping sebelum data diproses oleh model
        clamped_data = clamp_input(input_data)
        
        # simpan input yang telah di-clamp ke dalam DataFrame
        input_df = pd.DataFrame([clamped_data])
        
        # pastikan urutan kolom input_df sesuai dengan feature_cols.pkl
        input_df = input_df[feature_cols]
        
        # lakukan encoding untuk fitur kategorikal menggunakan label encoders yang sudah dimuat dari pkl
        cat_cols_to_encode = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level']
        for col in cat_cols_to_encode:
            encoder = label_encoders[col]
            input_df[col] = encoder.transform(input_df[col])
            
        # lakukan scaling untuk fitur numerik berdasarkan scaler dari .pkl
        input_scaled = scaler.transform(input_df)
        
        # jalankan prediksi menggunakan neural network keras
        preds = model.predict(input_scaled)
        
        # proses output prediksi untuk mendapatkan kategori risiko akhir dan probabilitas masing-masing kelas risiko (Low, Medium, High)
        # deteksi output array untuk klasifikasi 3 kelas
        if isinstance(preds, list):
            # jika model multioutput menghasilkan list, cari array yang memiliki dimensi kedua 3 (jumlah kelas risiko)
            risk_pred_array = [p for p in preds if p.shape[1] == 3][0]
        else:
            risk_pred_array = preds
            
        # pastikan risk_pred_array memiliki bentuk yang benar untuk klasifikasi 3 kelas
        pred_idx = np.argmax(risk_pred_array[0])
        probabilities = risk_pred_array[0]
        
        # mapping indeks prediksi ke label risiko menggunakan risk_map.pkl yang sudah dimuat
        final_risk_label = idx_to_label.get(pred_idx, "Unknown")

        with st.spinner("Sedang memproses prediksi..."):
            ai_insight = get_genai_insight(
                clamped_data,  # Gunakan data yang sudah di-clamp untuk konsistensi insight
                risk_category=final_risk_label,
                confidence=probabilities[pred_idx]
            )

        # tampilkan hasil prediksi dan rekomendasi intervensi
        st.markdown("---")
        st.markdown("### Hasil Analisis Prediksi Kecerdasan Buatan")
        
        # set background color berdasarkan kategori risiko
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
        
        # tampilkan probabilitas untuk kelas risiko
        st.markdown("<br>", unsafe_allow_html=True)
        c_p1, c_p2, c_p3 = st.columns(3)
        # gunakan label_to_idx untuk memastikan urutan probabilitas sesuai dengan kelas risiko yang benar
        c_p1.metric("Probabilitas Low Risk", f"{probabilities[label_to_idx['Low']]*100:.2f}%")
        c_p2.metric("Probabilitas Medium Risk", f"{probabilities[label_to_idx['Medium']]*100:.2f}%")
        c_p3.metric("Probabilitas High Risk", f"{probabilities[label_to_idx['High']]*100:.2f}%")
        
        # tampilkan insight tambahan dari GenAI untuk rekomendasi intervensi berbasis AI
        st.markdown("#### 💡Rekomendasi Intervensi AI:")
        st.info(ai_insight) # rekomendasi intervensi berbasis AI dari API Railway GenAI

        """
        # tampilkan rekomendasi intervensi berdasarkan risiko yang terdeteksi
        st.markdown("#### 💡 Rekomendasi Strategi Intervensi Pendidik:")
        if final_risk_label == 'High':
            st.error("🚨 **Intervensi Prioritas Utama:** Siswa memerlukan pendampingan belajar intensif empat mata harian. Segera adakan pertemuan segitiga bersama wali murid, optimalkan sesi bimbingan konseling akademik khusus, serta lakukan penyesuaian target capaian nilai harian.")
        elif final_risk_label == 'Medium':
            st.warning("⚠️ **Intervensi Preventif:** Tingkatkan frekuensi keterlibatan siswa dalam forum kelompok belajar, dorong siswa untuk menambah 2-3 jam sesi belajar mandiri terstruktur per minggu, serta lakukan pemantauan grafik absensi kehadiran di kelas secara berkala.")
        else:
            st.success("🌟 **Strategi Preservasi Performa:** Siswa berada di jalur akademis yang aman dan stabil. Berikan apresiasi motivasi berkala agar siswa konsisten mempertahankan ritme kedisiplinan belajar dan tingkat kehadiran saat ini.")
        """