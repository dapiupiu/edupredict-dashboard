import streamlit as st
import pandas as pd
import plotly.express as px

def render_bq(df, color_map):
    st.title("Business Questions Insights")
    
    bq_select = st.selectbox("Pilih Pertanyaan Bisnis (BQ):", options=[
        "BQ1: Variabel penentu utama performa nilai ujian akhir?",
        "BQ2: Bagaimana dampak Keterlibatan Orang Tua terhadap nilai ujian?",
        "BQ3: Apakah motivasi belajar linier dengan alokasi waktu dan kehadiran?",
        "BQ4: Bagaimana profil finansial dan akses internet pada siswa High Risk?",
        "BQ5: Bagaimana pengaruh Kualitas Guru terhadap proporsi risiko akademik?",
        "BQ6: Kombinasi faktor dominan apa yang mencirikan siswa High Risk?"
    ])
    
    NUMERICAL_COLS = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']

    if bq_select.startswith("BQ1"):
        st.info("**BQ1:** Fitur-fitur apa saja yang memiliki korelasi/asosisasi tertinggi terhadap capaian Exam Score siswa?")
        corrs = df[NUMERICAL_COLS].corrwith(df['Exam_Score']).reset_index(name='Korelasi').sort_values(by='Korelasi', ascending=False)
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(corrs, x='Korelasi', y='index', orientation='h', title="Korelasi Variabel Numerik vs Exam Score"), use_container_width=True)
        with c2:
            cat_mean = df.groupby('Motivation_Level')['Exam_Score'].mean().reset_index()
            st.plotly_chart(px.bar(cat_mean, x='Motivation_Level', y='Exam_Score', title="Rata-rata Exam Score Berdasarkan Motivasi"), use_container_width=True)
        st.success("**Insight:** Attendance dan Hours_Studied memiliki korelasi positif tertinggi terhadap Exam_Score.")

    elif bq_select.startswith("BQ2"):
        st.info("**BQ2:** Apakah tingkat Keterlibatan Orang Tua (Parental Involvement) berdampak signifikan terhadap sebaran nilai ujian?")
        stats_involve = df.groupby('Parental_Involvement')['Exam_Score'].agg(['mean', 'std']).reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(stats_involve, x='Parental_Involvement', y='mean', error_y='std', title="Rata-rata Nilai Ujian ± Std Deviasi"), use_container_width=True)
        with c2:
            st.plotly_chart(px.box(df, x='Parental_Involvement', y='Exam_Score', color='Parental_Involvement', title="Boxplot Distribusi Nilai"), use_container_width=True)
        st.success("**Insight:** Keterlibatan orang tua yang tinggi berkorelasi positif dengan kestabilan batas bawah nilai ujian siswa.")

    elif bq_select.startswith("BQ3"):
        st.info("**BQ3:** Apakah siswa dengan motivasi belajar rendah juga otomatis memiliki jam belajar dan tingkat kehadiran yang rendah?")
        m_hours = df.groupby('Motivation_Level')['Hours_Studied'].mean().reset_index()
        m_attend = df.groupby('Motivation_Level')['Attendance'].mean().reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(m_hours, x='Motivation_Level', y='Hours_Studied', title="Rata-rata Jam Belajar per Kategori Motivasi"), use_container_width=True)
        with c2:
            st.plotly_chart(px.bar(m_attend, x='Motivation_Level', y='Attendance', title="Rata-rata Kehadiran (%) per Kategori Motivasi"), use_container_width=True)
        st.success("**Insight:** Kondisi psikologis motivasi yang rendah sejalan dengan rendahnya durasi belajar mandiri dan tingkat kehadiran kelas harian.")

    elif bq_select.startswith("BQ4"):
        st.info("**BQ4:** Apakah faktor keterbatasan ekonomi (Family Income = Low) dan ketiadaan internet (Internet Access = No) mendominasi kelompok siswa High Risk?")
        hr_df = df[df['Risk_Category'] == 'High']
        if len(hr_df) == 0:
            st.warning("Tidak ada data siswa High Risk pada filter ini.")
        else:
            p_income = (hr_df['Family_Income'] == 'Low').mean() * 100
            p_internet = (hr_df['Internet_Access'] == 'No').mean() * 100
            bq4_data = pd.DataFrame({'Kondisi': ['Family Income = Low', 'Internet Access = No'], 'Persentase': [p_income, p_internet]})
            st.plotly_chart(px.bar(bq4_data, x='Persentase', y='Kondisi', orientation='h', text=bq4_data['Persentase'].apply(lambda x: f"{x:.1f}%"), title="Kondisi Lingkungan Kelompok High Risk", range_x=[0,100]), use_container_width=True)
            st.success("**Insight:** 54.4% siswa High Risk berasal dari ekonomi rendah, namun mayoritas tetap memiliki akses internet (85.3%). Internet bukan faktor dominan penentu.")

    elif bq_select.startswith("BQ5"):
        st.info("**BQ5:** Bagaimana distribusi proporsi Kategori Risiko akademik jika dihubungkan dengan tingkat kualitas pengajar (Teacher Quality)?")
        prop_df = df.groupby(['Teacher_Quality', 'Risk_Category']).size().reset_index(name='Count')
        prop_df['Persentase'] = (prop_df['Count'] / prop_df.groupby('Teacher_Quality')['Count'].transform('sum') * 100).round(2)
        st.plotly_chart(px.bar(prop_df, x='Teacher_Quality', y='Persentase', color='Risk_Category', color_discrete_map=color_map, title="Proporsi Risiko per Kualitas Guru (%)"), use_container_width=True)
        st.success("**Insight:** Peningkatan kualitas guru berbanding lurus dengan minimalisasi pembentukan kelompok siswa berisiko tinggi.")

    elif bq_select.startswith("BQ6"):
        st.info("**BQ6:** Bagaimana potret sebaran karakteristik psikososial khusus untuk siswa kategori High Risk?")
        hr_df = df[df['Risk_Category'] == 'High']
        if len(hr_df) == 0:
            st.warning("Tidak ada data siswa High Risk.")
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(px.bar(hr_df['Motivation_Level'].value_counts(normalize=True).reset_index(name='p').assign(p=lambda x:x['p']*100), x='p', y='Motivation_Level', orientation='h', title='Motivasi High Risk (%)'), use_container_width=True)
            with c2:
                st.plotly_chart(px.bar(hr_df['Parental_Involvement'].value_counts(normalize=True).reset_index(name='p').assign(p=lambda x:x['p']*100), x='p', y='Parental_Involvement', orientation='h', title='Keterlibatan Orang Tua (%)'), use_container_width=True)
            with c3:
                st.plotly_chart(px.bar(hr_df['Family_Income'].value_counts(normalize=True).reset_index(name='p').assign(p=lambda x:x['p']*100), x='p', y='Family_Income', orientation='h', title='Pendapatan Keluarga (%)'), use_container_width=True)
            st.success("**Insight:** Profil kombinasi karakteristik utama siswa High Risk didominasi oleh perpaduan tiga faktor: Motivasi Low (52.9%), Pendapatan Low (54.4%), dan Keterlibatan Orang Tua yang minim.")