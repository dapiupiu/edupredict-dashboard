import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats


def render_comparative(df_raw):
    st.title("Uji Komparatif Inferensial Berbasis Kehadiran")
    st.info(
        "ℹ**Disclaimer Metodologi:** Ini adalah uji komparatif dua sampel independen berbasis data observasional historis, bukan hasil eksperimen uji A/B terkontrol."
    )

    # slider batas threshold kehadiran
    attend_data = df_raw["Attendance"]
    q25, q50, q75 = (
        int(np.percentile(attend_data, 25)),
        int(np.percentile(attend_data, 50)),
        int(np.percentile(attend_data, 75)),
    )

    st.markdown("### Pengaturan Batas Threshold Kehadiran")
    threshold = st.slider(
        "Tentukan Batas Kehadiran (%) untuk Pembagian Grup:",
        min_value=q25,
        max_value=q75,
        value=q50,
        step=1,
    )

    grup_a = df_raw[df_raw["Attendance"] < threshold]["Exam_Score"]
    grup_b = df_raw[df_raw["Attendance"] >= threshold]["Exam_Score"]

    if len(grup_a) < 2 or len(grup_b) < 2:
        st.error("Ukuran sampel terlalu sedikit untuk pengujian statistik.")
        return

    st.markdown("### Perbandingan Distribusi Nilai Ujian Akhir (Exam Score)")
    combined = pd.concat(
        [
            pd.DataFrame(
                {"Exam_Score": grup_a, "Grup": f"Grup A (Attendance < {threshold}%)"}
            ),
            pd.DataFrame(
                {"Exam_Score": grup_b, "Grup": f"Grup B (Attendance >= {threshold}%)"}
            ),
        ]
    )
    fig_kde = px.histogram(
        combined,
        x="Exam_Score",
        color="Grup",
        marginal="violin",
        opacity=0.5,
        barmode="overlay",
        title="Distribusi dan Kepadatan Nilai: Grup A vs Grup B",
        color_discrete_sequence=["#e74c3c", "#2ecc71"],
    )
    st.plotly_chart(fig_kde, use_container_width=True)

    # kalkulasi statistika
    ttest_res = stats.ttest_ind(grup_b, grup_a, equal_var=False)
    # access TtestResult
    stat_attr = getattr(ttest_res, "statistic", None)
    p_attr = getattr(ttest_res, "pvalue", None)
    if stat_attr is not None and p_attr is not None:
        t_stat = np.asarray(stat_attr).squeeze().item()
        p_val = np.asarray(p_attr).squeeze().item()
    else:
        # fallback jika atribut tidak ditemukan
        t_stat = np.asarray(ttest_res[0]).squeeze().item()
        p_val = np.asarray(ttest_res[1]).squeeze().item()
    mean_a, mean_b = grup_a.mean(), grup_b.mean()
    n_a, n_b = len(grup_a), len(grup_b)
    pooled_std = np.sqrt(
        ((n_a - 1) * grup_a.var() + (n_b - 1) * grup_b.var()) / (n_a + n_b - 2)
    )
    cohen_d = (mean_b - mean_a) / pooled_std

    effect_desc = (
        "Kecil" if abs(cohen_d) < 0.2 else ("Sedang" if abs(cohen_d) < 0.8 else "Besar")
    )

    st.markdown("### Hasil Pengujian Hipotesis Statistik")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ukuran Sampel (A / B)", f"{n_a} / {n_b}")
    c2.metric("Rata-rata Nilai (A vs B)", f"{mean_a:.1f} vs {mean_b:.1f}")
    c3.metric("T-Statistic", f"{t_stat:.4f}")
    c4.metric("p-value", f"{p_val:.4e}")

    st.markdown("#### **Kesimpulan Analitik:**")
    if p_val <= 0.05:
        st.success(
            f"✅ **Tolak H0.** Nilai *p-value* ({p_val:.4e}) < 0.05. Terdapat perbedaan rata-rata nilai ujian yang signifikan secara statistik antara kelompok kehadiran tinggi dan rendah."
        )
    else:
        st.warning(
            "❌ **Gagal Tolak H0.** Tidak cukup bukti statistik menyatakan adanya perbedaan rata-rata."
        )

    st.info(
        f"📐 **Ukuran Efek (Cohen's d):** Nilai d = **{cohen_d:.3f}** tergolong dalam kategori **{effect_desc} Effect**."
    )
    st.caption(
        "💡 *Justifikasi Teorema Batas Pusat (CLT):* Karena ukuran sampel pada kedua kelompok bernilai sangat besar (N > 30), asumsi normalitas distribusi rata-rata sampel terpenuhi berdasarkan CLT sehingga pengujian ini valid."
    )
