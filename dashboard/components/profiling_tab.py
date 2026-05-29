import streamlit as st
import plotly.express as px

def render_profiling(df_raw, color_map):
    st.title("Profiling Khusus Siswa High Risk")
    
    # sidebar filter untuk fokus pada siswa high risk
    st.sidebar.header("Filter Profil High Risk")
    hr_gender = st.sidebar.selectbox("Gender:", ["All"] + list(df_raw['Gender'].unique()))
    hr_school = st.sidebar.selectbox("Tipe Sekolah:", ["All"] + list(df_raw['School_Type'].unique()))
    hr_income = st.sidebar.selectbox("Pendapatan Keluarga:", ["All"] + list(df_raw['Family_Income'].unique()))
    
    # filter data untuk siswa high risk berdasarkan input filter
    df_hr = df_raw[df_raw['Risk_Category'] == 'High']
    if hr_gender != "All":
        df_hr = df_hr[df_hr['Gender'] == hr_gender]
    if hr_school != "All":
        df_hr = df_hr[df_hr['School_Type'] == hr_school]
    if hr_income != "All":
        df_hr = df_hr[df_hr['Family_Income'] == hr_income]
        
    total_hr_all = len(df_raw[df_raw['Risk_Category'] == 'High'])
    total_hr_filtered = len(df_hr)
    
    c_m1, c_m2 = st.columns(2)
    c_m1.metric("Jumlah Siswa High Risk (Filtered)", f"{total_hr_filtered} siswa")
    c_m2.metric("Proporsi dari Total Kategori Sejenis", f"{(total_hr_filtered/max(1, total_hr_all))*100:.2f}%")
    
    if total_hr_filtered == 0:
        st.warning("⚠️ Tidak ada data yang sesuai filter.")
        return

    st.markdown("### Matriks Distribusi Fitur Khusus Siswa High Risk")
    feats = ['Motivation_Level', 'Parental_Involvement', 'Family_Income', 'Teacher_Quality', 'Internet_Access', 'Peer_Influence']
    cols_grid = st.columns(3) + st.columns(3)
    
    for idx, feat in enumerate(feats):
        with cols_grid[idx]:
            counts = df_hr[feat].value_counts().reset_index(name='Count')
            fig = px.bar(counts, x=feat, y='Count', title=f"Distribusi {feat}", color_discrete_sequence=['#e74c3c'])
            st.plotly_chart(fig, use_container_width=True)
            
    # tampilkan tabel detail siswa high risk yang memenuhi filter dengan highlight khusus
    st.markdown("### Daftar Detail Siswa Berisiko Tinggi")
    display_cols = ['Risk_Category', 'Hours_Studied', 'Attendance', 'Previous_Scores', 'Motivation_Level', 'Parental_Involvement', 'Family_Income', 'Gender', 'School_Type']
    st.dataframe(df_hr[display_cols], use_container_width=True)
    
    # download button untuk data siswa high risk yang sudah difilter
    csv_bytes = df_hr[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Unduh Data Siswa High Risk (CSV)", data=csv_bytes, file_name="Siswa_High_Risk_Prioritas.csv", mime="text/csv")