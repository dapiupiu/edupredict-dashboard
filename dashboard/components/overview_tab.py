import streamlit as st
import plotly.express as px

def render_overview(df, color_map):
    st.title("Overview (Ringkasan Eksekutif)")
    st.markdown("Sistem early warning detection berbasis data historis dan integrasi AI untuk deteksi dini risiko performa akademik siswa.")
    
    # kalkulasi metrik utama
    total_siswa = len(df)
    high_count = len(df[df['Risk_Category'] == 'High'])
    med_count = len(df[df['Risk_Category'] == 'Medium'])
    low_count = len(df[df['Risk_Category'] == 'Low'])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Siswa (Filtered)", f"{total_siswa:,}")
    col2.metric("Jumlah High Risk", f"{high_count:,}", delta=f"{(high_count/max(1, total_siswa))*100:.1f}%", delta_color="inverse")
    col3.metric("Jumlah Medium Risk", f"{med_count:,}", delta=f"{(med_count/max(1, total_siswa))*100:.1f}%", delta_color="off")
    col4.metric("Jumlah Low Risk", f"{low_count:,}", delta=f"{(low_count/max(1, total_siswa))*100:.1f}%", delta_color="normal")
    
    st.warning("⚠️ **Catatan Karakteristik Dataset:** Kelas **High Risk** sangat minoritas (hanya sekitar 1% pada data asli). Ini mencerminkan kondisi riil di mana siswa kritis berjumlah sedikit namun membutuhkan perhatian paling intensif.")
    
    st.markdown("### Distribusi Kategori Risiko")
    c1, c2 = st.columns(2)
    
    with c1:
        risk_dist = df['Risk_Category'].value_counts().reset_index()
        risk_dist.columns = ['Risk_Category', 'Count']
        fig_donut = px.pie(risk_dist, names='Risk_Category', values='Count', hole=0.4,
                           title='Persentase Distribusi Risk Category', color='Risk_Category', color_discrete_map=color_map)
        st.plotly_chart(fig_donut, use_container_width=True)
        
    with c2:
        risk_dist['Persentase'] = (risk_dist['Count'] / risk_dist['Count'].sum() * 100).round(2)
        fig_bar = px.bar(risk_dist, x='Risk_Category', y='Count', text=risk_dist.apply(lambda r: f"{r['Count']} ({r['Persentase']}%)", axis=1),
                         title='Jumlah Siswa per Kategori Risiko', color='Risk_Category', color_discrete_map=color_map)
        fig_bar.update_layout(yaxis_title="Jumlah Siswa", xaxis_title="Kategori Risiko")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("### Ringkasan Analisis")
    st.info(f"Berdasarkan filter aktif saat ini, terdapat **{high_count} siswa ({(high_count/max(1, total_siswa))*100:.1f}%)** yang tergolong dalam kategori **High Risk** dan membutuhkan intervensi segera.")