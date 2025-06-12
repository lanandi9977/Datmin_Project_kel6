import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisis Data", page_icon="📊", layout="wide")

st.title("📊 Analisis Data Eksplorasi (EDA)")
st.markdown("Halaman ini menampilkan analisis awal dari dataset kepribadian. Anda dapat melihat data mentah, statistik deskriptif, dan berbagai visualisasi untuk memahami karakteristik data.")
st.divider()

# Ambil data yang sudah dimuat dari session state
df_raw = st.session_state.df_raw
df_processed = st.session_state.df_processed

st.subheader("Tabel Dataset")
st.dataframe(df_raw)

st.subheader("Statistik Deskriptif")
st.write(df_raw.describe())

st.divider()
st.subheader("Visualisasi Data")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Distribusi Kepribadian")
    fig_pie = px.pie(df_raw, names='Personality', title='Persentase Introvert vs. Extrovert', hole=0.3)
    st.plotly_chart(fig_pie, use_container_width=True)
with col2:
    st.markdown("#### Distribusi Rasa Takut Panggung")
    fig_bar = px.bar(df_raw['Stage_fear'].value_counts(), x=df_raw['Stage_fear'].value_counts().index, y=df_raw['Stage_fear'].value_counts().values, title='Jumlah Responden Berdasarkan Rasa Takut Panggung', labels={'x':'Rasa Takut Panggung', 'y':'Jumlah Orang'})
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("#### Hubungan Antara Lingkaran Pertemanan dan Kehadiran di Acara Sosial")
fig_scatter = px.scatter(df_raw, x='Friends_circle_size', y='Social_event_attendance', color='Personality', title='Ukuran Lingkaran Teman vs. Kehadiran di Acara Sosial', labels={'Friends_circle_size': 'Ukuran Lingkaran Teman', 'Social_event_attendance': 'Kehadiran di Acara Sosial'}, hover_data=['Time_spent_Alone'])
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()
st.subheader("Peta Korelasi Antar Fitur")
st.markdown("Heatmap ini menunjukkan bagaimana setiap fitur numerik berhubungan satu sama lain. Angka mendekati 1 (biru tua) atau -1 (merah tua) menunjukkan korelasi yang kuat.")
corr = df_processed.corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)
