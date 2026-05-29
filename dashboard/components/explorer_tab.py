import streamlit as st

def render_explorer(df_raw):
    st.title("Data Explorer")
    st.markdown("Gunakan halaman ini untuk menyaring, menyortir, mendalami, dan mengunduh baris data spesifik.")
    
    # sidebar filter
    st.sidebar.header("Filter Eksplorasi")
    exp_risk = st.sidebar.multiselect("Risk Category:", options=['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
    exp_gender = st.sidebar.selectbox("Gender:", ["All"] + list(df_raw['Gender'].unique()))
    exp_school = st.sidebar.selectbox("School Type:", ["All"] + list(df_raw['School_Type'].unique()))
    
    # filter data berdasarkan input
    df_exp = df_raw[df_raw['Risk_Category'].isin(exp_risk)]
    if exp_gender != "All":
        df_exp = df_exp[df_exp['Gender'] == exp_gender]
    if exp_school != "All":
        df_exp = df_exp[df_exp['School_Type'] == exp_school]
        
    st.metric("Jumlah Baris Ditemukan (Sesudah Filter):", f"{len(df_exp)} baris")
    
    if len(df_exp) == 0:
        st.warning("⚠️ Tidak ada data yang sesuai dengan kombinasi kriteria pencarian.")
        return

    # highlight anomali (High Risk diberi warna latar merah muda)
    def highlight_high_risk(row):
        return ['background-color: #fadbd8' if row['Risk_Category'] == 'High' else '' for _ in row]
        
    st.markdown("> **Petunjuk:** Klik judul kolom untuk melakukan pengurutan data (*sorting*). Kolom `Exam_Score` diikutsertakan di sini murni untuk keperluan penelaahan riwayat historis.")
    st.dataframe(df_exp.style.apply(highlight_high_risk, axis=1), use_container_width=True, height=500)
    
    # download button
    csv_bytes = df_exp.to_csv(index=False).encode('utf-8')
    st.download_button(label="📥 Unduh Data Hasil Filter Ini (CSV)", data=csv_bytes, file_name="EduPredict_Filtered_Explorer.csv", mime="text/csv")