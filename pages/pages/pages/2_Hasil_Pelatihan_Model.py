import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hasil Model", page_icon="🤖", layout="wide")

st.title("🤖 Hasil Pelatihan Model Machine Learning")
st.markdown("Di halaman ini, kita melihat performa dari dua model yang telah dilatih: **K-Nearest Neighbors (KNN)** dan **Gaussian Naive Bayes**.")
st.divider()

# Ambil hasil model dari session state
acc_knn = st.session_state.acc_knn
report_knn = st.session_state.report_knn
cm_knn = st.session_state.cm_knn
acc_nb = st.session_state.acc_nb
report_nb = st.session_state.report_nb
cm_nb = st.session_state.cm_nb

labels = ['Extrovert', 'Introvert']
col1, col2 = st.columns(2)
with col1:
    st.subheader("K-Nearest Neighbors (KNN)")
    st.metric(label="Akurasi Model", value=f"{acc_knn:.2%}")
    st.text("Laporan Klasifikasi:")
    st.code(report_knn)
    st.text("Confusion Matrix:")
    fig_cm_knn, ax_cm_knn = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax_cm_knn)
    ax_cm_knn.set_xlabel('Prediksi')
    ax_cm_knn.set_ylabel('Aktual')
    st.pyplot(fig_cm_knn)

with col2:
    st.subheader("Gaussian Naive Bayes")
    st.metric(label="Akurasi Model", value=f"{acc_nb:.2%}")
    st.text("Laporan Klasifikasi:")
    st.code(report_nb)
    st.text("Confusion Matrix:")
    fig_cm_nb, ax_cm_nb = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges', xticklabels=labels, yticklabels=labels, ax=ax_cm_nb)
    ax_cm_nb.set_xlabel('Prediksi')
    ax_cm_nb.set_ylabel('Aktual')
    st.pyplot(fig_cm_nb)
