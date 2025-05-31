import streamlit as st
import pandas as pd
import os
import re
import time

from langchain_community.llms import Replicate
from dotenv import load_dotenv
from collections import Counter
from wordcloud import WordCloud 

import matplotlib.pyplot as plt 
import seaborn as sns 

load_dotenv()

st.set_page_config(page_title="Analisis Ulasan Game", layout="wide", initial_sidebar_state="expanded")

MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"
PROMPT_TEMPLATE_SENTIMENT_ASPECT = """
Analyze the following game review.
Determine the sentiment (Positive or Negative) and list up to 3 key aspects or topics discussed in the review related to gameplay, bugs, graphics, or performance.
If an aspect is not clearly discussed or the review is too short, aspects can be "N/A".

Review: "{review_text}"

Output in the following format:
Sentiment: [Positive/Negative]
Aspects: [Aspect1, Aspect2, Aspect3 or N/A]
"""
MODEL_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.2
}

@st.cache_resource
def init_llm_streamlit():
    replicate_api_token_env = os.environ.get('REPLICATE_API_TOKEN')

    if not replicate_api_token_env:
        st.error("Token API Replicate tidak ditemukan. Pastikan file .env sudah benar dan REPLICATE_API_TOKEN telah di-set.")
        return None

    try:
        llm_instance = Replicate(
            model=MODEL_ID,
            model_kwargs=MODEL_PARAMS
        )
        print(f"Model {MODEL_ID} berhasil diinisialisasi untuk Streamlit.")
        return llm_instance
    except Exception as e:
        st.error(f"Error saat inisialisasi model Replicate: {e}")
        return None

llm_streamlit = init_llm_streamlit()

# Fungsi untuk menganalisis ulasan dengan AI.
def analyze_review_with_ai_streamlit(review_text, prompt_template, current_llm):

    if not current_llm:
        return "Model LLM tidak siap atau gagal diinisialisasi."
    if not review_text or not review_text.strip():
        return "Teks ulasan kosong atau hanya berisi spasi."

    prompt = prompt_template.format(review_text=review_text)
    try:
        response = current_llm.invoke(prompt)
        return response.strip()
    except Exception as e:
        st.error(f"Error saat memanggil model AI: {e}")
        return None

# Fungsi untuk memparsing output AI yang diterima.
def parse_ai_output_streamlit(ai_output_text):

    predicted_sentiment = "N/A (Parsing Failed)"
    aspects_str = "N/A (Parsing Failed)"
    explanation_str = "N/A"

    if ai_output_text:
        try:
            lines = ai_output_text.split('\n')
            sentiment_found = False
            aspects_found = False
            explanation_lines = []
            temp_aspects = []

            for line_num, line_content in enumerate(lines):
                line_lower = line_content.lower().strip()
                if line_lower.startswith("sentiment:"):
                    predicted_sentiment = line_content.split(":", 1)[1].strip()
                    sentiment_found = True
                elif line_lower.startswith("aspects:"):
                    aspect_str_raw = line_content.split(":", 1)[1].strip()
                    if aspect_str_raw.lower() != "n/a" and aspect_str_raw:
                        temp_aspects = [aspect.strip() for aspect in re.split(r',|;', aspect_str_raw) if aspect.strip()]
                    else:
                        temp_aspects = ["N/A"]
                    aspects_found = True
                elif (sentiment_found and aspects_found and line_content.strip() and not line_lower.startswith("sentiment:")) or \
                     (not sentiment_found and not aspects_found and line_content.strip()):
                    explanation_lines.append(line_content.strip())
            
            if temp_aspects:
                aspects_str = ", ".join(temp_aspects)
            
            if explanation_lines:
                explanation_str = "\n".join(explanation_lines)
            elif not sentiment_found and not aspects_found and ai_output_text and not temp_aspects:
                explanation_str = ai_output_text
                if predicted_sentiment == "N/A (Parsing Failed)":
                    predicted_sentiment = "Format Tidak Dikenali"

        except Exception as e:
            st.warning(f"Error saat parsing output AI: {e}. Output asli:\n{ai_output_text}")
            explanation_str = f"Parsing Error: {e}. Output asli:\n{ai_output_text}"
    
    return predicted_sentiment, aspects_str, explanation_str

# Fungsi untuk memuat data dari file CSV
@st.cache_data 
def load_data_streamlit(csv_path):
    """Memuat data dari file CSV."""
    try:
        data = pd.read_csv(csv_path)
        return data
    except FileNotFoundError:
        st.error(f"File data '{csv_path}' tidak ditemukan. Pastikan file tersebut ada di direktori yang sama dengan app.py.")
        return pd.DataFrame() 

NAMA_FILE_SAMPEL_CSV = 'sampel_analisis.csv'
df_display_streamlit = load_data_streamlit(NAMA_FILE_SAMPEL_CSV)


st.title("Analisis Sentimen dan Aspek Ulasan Game")
st.markdown(f"Aplikasi ini menggunakan model AI **{MODEL_ID}** untuk menganalisis sentimen dan aspek dari ulasan game.")

st.sidebar.header("Informasi & Kontrol")
st.sidebar.info(f"**Model AI:**\n{MODEL_ID}")
st.sidebar.markdown("---")
st.sidebar.subheader("Parameter Default AI:")
for key, value in MODEL_PARAMS.items():
    st.sidebar.text(f"{key}: {value}")
st.sidebar.markdown("---")

# Dibuat dua tab utama: satu untuk analisis ulasan baru, dan satu untuk melihat data sampel.
tab_analisis_baru, tab_data_sampel = st.tabs(["Analisis Ulasan Baru", "Lihat Data Sampel"])

with tab_analisis_baru:
    st.header("Input Ulasan Baru")
    new_review_text_input = st.text_area(
        "Masukkan teks ulasan game di sini:",
        height=200,
        key="new_review_input_area",
        placeholder="Contoh: This game is amazing! The graphics are stunning and the gameplay is very engaging."
    )

    if st.button("Analisis Ulasan Ini!", key="analyze_button_main", type="primary"):
        if new_review_text_input and llm_streamlit:
            with st.spinner("Sedang menganalisis dengan AI... Ini mungkin memerlukan beberapa saat."):
                ai_output_new_review = analyze_review_with_ai_streamlit(new_review_text_input, PROMPT_TEMPLATE_SENTIMENT_ASPECT, llm_streamlit)
                
                if ai_output_new_review and "Model LLM tidak siap" not in ai_output_new_review and "Teks ulasan kosong" not in ai_output_new_review:
                    sentiment_new, aspects_new, explanation_new = parse_ai_output_streamlit(ai_output_new_review)
                    
                    st.subheader("Hasil Analisis AI:")
                    if sentiment_new and sentiment_new != "N/A (Parsing Failed)" and sentiment_new != "Format Tidak Dikenali":
                        if "positive" in sentiment_new.lower():
                            st.success(f"**Sentimen:** {sentiment_new}")
                        elif "negative" in sentiment_new.lower():
                            st.error(f"**Sentimen:** {sentiment_new}")
                        else: 
                            st.info(f"**Sentimen:** {sentiment_new}")
                    else:
                        st.warning(f"**Sentimen:** {sentiment_new}")

                    st.write(f"**Aspek yang Dibahas:** {aspects_new}")

                    if explanation_new and explanation_new != "N/A":
                        with st.expander("Lihat Penjelasan / Output Mentah dari AI"):
                            st.text(explanation_new)
                elif ai_output_new_review:
                     st.warning(ai_output_new_review)
                else:
                    st.error("Gagal mendapatkan respons dari model AI. Silakan periksa log error di terminal atau coba lagi.")
        elif not llm_streamlit:
            st.error("Model AI tidak siap atau gagal diinisialisasi. Periksa token API dan log error di terminal.")
        else:
            st.warning("Mohon masukkan teks ulasan terlebih dahulu.")

with tab_data_sampel:
    st.header("Contoh Hasil Analisis pada Data Sampel")
    if not df_display_streamlit.empty:
        st.info(f"Menampilkan data dari file: **{NAMA_FILE_SAMPEL_CSV}**")

        kolom_tampil_sampel = [col for col in ['id', 'content_cleaned', 'is_positive', 'ai_sentiment', 'ai_aspects'] if col in df_display_streamlit.columns]
        
        if not kolom_tampil_sampel:
            st.warning("Tidak ada kolom yang diharapkan untuk ditampilkan dari file CSV sampel.")
        else:
            st.dataframe(df_display_streamlit[kolom_tampil_sampel].head(50))

            kolom_sentimen_ai_chart = 'ai_sentiment' 
            if kolom_sentimen_ai_chart in df_display_streamlit.columns:
                st.subheader(f"Distribusi '{kolom_sentimen_ai_chart}' pada Data Sampel")
                ai_sentiment_counts_display = df_display_streamlit[kolom_sentimen_ai_chart].value_counts()
                if not ai_sentiment_counts_display.empty:
                    try:
                        fig, ax = plt.subplots(figsize=(8,5))
                        sns.barplot(x=ai_sentiment_counts_display.index, y=ai_sentiment_counts_display.values, ax=ax, palette="pastel")
                        ax.set_title(f"Distribusi '{kolom_sentimen_ai_chart}'")
                        ax.set_xlabel("Sentimen AI")
                        ax.set_ylabel("Jumlah Ulasan")
                        st.pyplot(fig) 
                    except Exception as e:
                        st.warning(f"Tidak bisa membuat chart distribusi sentimen: {e}. Mencoba st.bar_chart...")
                        st.bar_chart(ai_sentiment_counts_display)
                else:
                    st.write(f"Tidak ada data di kolom '{kolom_sentimen_ai_chart}' untuk ditampilkan dalam chart.")
            else:
                st.warning(f"Kolom '{kolom_sentimen_ai_chart}' tidak ditemukan untuk membuat chart distribusi.")
    else:
        st.warning(f"Tidak dapat memuat data sampel dari '{NAMA_FILE_SAMPEL_CSV}'. Pastikan file ada dan tidak kosong.")

st.markdown("---")
st.caption("Aplikasi Streamlit untuk Capstone Project Analisis Ulasan Game IBM x HACKTIV | Mohammad Hikam")