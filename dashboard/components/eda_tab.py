import streamlit as st
import plotly.express as px

def render_eda(df, color_map):
    st.title("🔍 Exploratory Data Analysis")
    st.markdown("Eksplorasi mendalam distribusi fitur numerik, kategorikal, dan pola hubungan variabel.")
    
    tab_num, tab_cat, tab_corr, tab_scatter = st.tabs(["🔢 Numerik", "🔠 Kategorikal", "🔗 Korelasi", "🎯 Scatterplot Interaktif"])
    
    NUMERICAL_COLS = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 'Tutoring_Sessions', 'Physical_Activity']
    CATEGORICAL_FEATURES = ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 'Peer_Influence', 'Parental_Education_Level', 'Gender', 'School_Type', 'Learning_Disabilities', 'Distance_from_Home', 'Extracurricular_Activities']

    with tab_num:
        selected_num = st.selectbox("Pilih Fitur Numerik:", NUMERICAL_COLS)
        fig_hist = px.histogram(df, x=selected_num, color="Risk_Category", marginal="box", opacity=0.7, barmode="overlay",
                                title=f"Distribusi dan Sebaran Data Fitur: {selected_num}", color_discrete_map=color_map)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with tab_cat:
        selected_cat = st.selectbox("Pilih Fitur Kategorikal:", CATEGORICAL_FEATURES)
        cat_counts = df.groupby([selected_cat, 'Risk_Category']).size().reset_index(name='Jumlah')
        fig_cat = px.bar(cat_counts, x=selected_cat, y='Jumlah', color='Risk_Category', barmode='group',
                         title=f"Frekuensi Distribusi Berdasarkan {selected_cat}", color_discrete_map=color_map)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    with tab_corr:
        corr_cols = NUMERICAL_COLS + ['Exam_Score']
        corr_matrix = df[corr_cols].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r', title='Matriks Korelasi Pearson (Analisis Asosiasi Nilai Ujian)')
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with tab_scatter:
        scat_x = st.selectbox("Pilih Sumbu X:", options=NUMERICAL_COLS, index=1)
        scat_y = st.selectbox("Pilih Sumbu Y:", options=NUMERICAL_COLS + ['Exam_Score'], index=0)
        fig_scat = px.scatter(df, x=scat_x, y=scat_y, color='Risk_Category', opacity=0.6,
                              title=f"Hubungan antara {scat_x} dan {scat_y}", color_discrete_map=color_map)
        st.plotly_chart(fig_scat, use_container_width=True)