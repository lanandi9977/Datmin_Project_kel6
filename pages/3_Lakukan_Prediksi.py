import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Prediksi", page_icon="🔮", layout="wide")

st.title("🔮 Formulir Prediksi Kepribadian")
st.markdown("Isi formulir di bawah ini dengan kebiasaan Anda, dan model akan mencoba memprediksi apakah Anda cenderung **Introvert** atau **Extrovert**.")
st.divider()

# Ambil semua yang dibutuhkan dari session state
model_knn = st.session_state.model_knn
model_nb = st.session_state.model_nb
df_processed = st.session_state.df_processed
introvert_avg = st.session_state.introvert_avg
extrovert_avg = st.session_state.extrovert_avg

with st.form("prediction_form"):
    st.subheader("Isi Data Anda:")
    col1, col2 = st.columns(2)
    with col1:
        time_spent_alone = st.slider("Waktu Dihabiskan Sendiri (Jam per hari)", 0, 11, 5)
        stage_fear = st.radio("Apakah Anda memiliki demam panggung?", ("Ya", "Tidak"))
        social_event_attendance = st.slider("Frekuensi menghadiri acara sosial (0-10)", 0, 10, 5)
        going_outside = st.slider("Frekuensi keluar rumah (kali per minggu)", 0, 7, 3)
    with col2:
        drained_after_socializing = st.radio("Apakah Anda merasa lelah setelah bersosialisasi?", ("Ya", "Tidak"))
        friends_circle_size = st.slider("Jumlah teman dekat (0-15)", 0, 15, 5)
        post_frequency = st.slider("Frekuensi posting di media sosial (0-10)", 0, 10, 4)
    
    submit_button = st.form_submit_button(label="✨ Lakukan Prediksi!")

if submit_button:
    stage_fear_num = 1 if stage_fear == "Ya" else 0
    drained_after_socializing_num = 1 if drained_after_socializing == "Ya" else 0
    input_data = pd.DataFrame([[time_spent_alone, stage_fear_num, social_event_attendance, going_outside, drained_after_socializing_num, friends_circle_size, post_frequency]], columns=df_processed.drop('Personality', axis=1).columns)

    prediction_knn = model_knn.predict(input_data)
    proba_knn = model_knn.predict_proba(input_data)
    confidence_knn = proba_knn[0][prediction_knn[0]]
    result_knn = "Introvert" if prediction_knn[0] == 1 else "Extrovert"

    prediction_nb = model_nb.predict(input_data)
    proba_nb = model_nb.predict_proba(input_data)
    confidence_nb = proba_nb[0][prediction_nb[0]]
    result_nb = "Introvert" if prediction_nb[0] == 1 else "Extrovert"

    st.divider()
    st.subheader("🎉 Hasil Prediksi Anda:")
    
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        st.info(f"**Prediksi Model KNN:** Anda cenderung seorang **{result_knn}** (kepercayaan: {confidence_knn:.0%})")
    with col_res2:
        st.success(f"**Prediksi Model Naive Bayes:** Anda cenderung seorang **{result_nb}** (kepercayaan: {confidence_nb:.0%})")

    with st.expander("Lihat Perbandingan Profil Anda (Radar Chart)"):
        feature_names = list(extrovert_avg.index)
        max_values = df_processed.drop('Personality', axis=1).max()
        user_normalized = input_data.iloc[0] / max_values
        introvert_normalized = introvert_avg / max_values
        extrovert_normalized = extrovert_avg / max_values

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=user_normalized, theta=feature_names, fill='toself', name='Profil Anda'))
        fig_radar.add_trace(go.Scatterpolar(r=introvert_normalized, theta=feature_names, fill='toself', name='Rata-rata Introvert', opacity=0.7))
        fig_radar.add_trace(go.Scatterpolar(r=extrovert_normalized, theta=feature_names, fill='toself', name='Rata-rata Extrovert', opacity=0.7))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Perbandingan Profil Anda dengan Profil Rata-Rata")
        st.plotly_chart(fig_radar, use_container_width=True)
