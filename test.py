# %%
"""
Nama: Mohammad Hikam 'Abdul Karim

Email: muhamadhikam94@gmail.com
"""

# %%
"""
## Data Preprocessing
"""

# %%
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from unidecode import unidecode
from collections import Counter
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import os
from dotenv import load_dotenv
import time


# %%
"""
Dataset: Steam Review & Games Dataset

Dataset link: https://www.kaggle.com/datasets/filipkin/steam-reviews?select=output.csv
"""

# %%
# Mengimport dataset dari file CSV.
df = pd.read_csv('steam_review.csv')
df.head()

# %%
# Deskripsi dataset.
df.describe()

# %%
# Informasi tentang dataset.
df.info()

# %%
# Shape dari dataset.
df.shape


# %%
# Cek apakah ada nilai yang duplikat.
df.duplicated().sum()

# %%
# Cek apakah ada nilai yang hilang
df.isna().sum()

# %%
# Drop/hapus kolom kosong
df = df.dropna(subset=['content'])
df.isna().sum()

# %%
# Preprocessing teks untuk membersihkan data.
def preprocess_text(text):
    
    if text is None:
        return ''
    
    text = text.lower()
    
    text = unidecode(text)
    
    text = text.replace('\r\n', ' ').replace('\n', ' ')
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'[^\w\s.!?]', '', text)
    
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    text = re.sub(r'[^\w\s.!?]', '', text)
    
    text = text.strip()
    
    text = re.sub(r'\s+', ' ', text)
    
    return text

# %%
# Aplikasi fungsi preprocessing ke kolom 'content' dan simpan hasilnya ke kolom baru 'content_cleaned'.
df['content_cleaned'] = df['content'].apply(preprocess_text)


# %%
# Filter ulasan yang sangat pendek (misal, hanya 1 karakter setelah tanda baca dihapus)..
min_char_length = 2
df = df[df['content_cleaned'].str.replace(r'[.!?]', '', regex=True).str.len() >= min_char_length]

# %%
# Mapping label sentimen dari teks ('Negative', 'Positive') ke numerik boolean(0, 1).
label_mapping = {'Negative': 0, 'Positive': 1}
df['sentiment_label'] = df['is_positive'].map(label_mapping)

# %%
#Tampilkan 100 baris pertama dari kolom 'content_cleaned'.
df['content_cleaned'].head(100)

# %%
# Menampilkan 10 baris pertama dari DataFrame.
df.head(10)

# %%
# Ekspor DataFrame yang telah dibersihkan ke file CSV.
to_csv_path = 'steam_review_cleaned.csv'
df.to_csv(to_csv_path, index=False)

# %%
"""
# Eksplorasi Data (EDA)

"""

# %%
"""
### I. Karakteristik Umum Dataset
"""

# %%
# Q1: Berapa jumlah total ulasan dalam dataset setelah semua proses pembersihan?
total_reviews_cleaned = df.shape[0]
print(f"Q1: Jumlah total ulasan setelah pembersihan: {total_reviews_cleaned}")

# %%
# Q2: Apakah masih ada nilai yang hilang (missing values) di kolom-kolom kunci?
missing_values_cleaned = df[['content_cleaned', 'is_positive', 'sentiment_label']].isnull().sum()
print(f"\nQ2: Nilai yang hilang per kolom penting setelah pembersihan:\n{missing_values_cleaned}")


# %%
# Q3: Berapa banyak ulasan unik (berdasarkan teksnya) dalam dataset?
unique_reviews_text_count = df['content_cleaned'].nunique()
print(f"\nQ3: Jumlah ulasan unik (berdasarkan teks 'content_cleaned'): {unique_reviews_text_count}")


# %%
"""
### II. Analisis Konten Teks Ulasan (`content_cleaned`)
"""

# %%
# Q4: Bagaimana distribusi panjang ulasan (jumlah karakter atau jumlah kata)?
df['char_length_cleaned'] = df['content_cleaned'].astype(str).str.len()
df['word_count_cleaned'] = df['content_cleaned'].astype(str).apply(lambda x: len(x.split()))

print(f"\nQ4: Statistik Deskriptif Panjang Ulasan (Karakter) setelah pembersihan:\n{df['char_length_cleaned'].describe()}")
print(f"\nQ4: Statistik Deskriptif Jumlah Kata Ulasan setelah pembersihan:\n{df['word_count_cleaned'].describe()}")


# %%
# Visualisasi distribusi panjang ulasan (jumlah kata)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(df['word_count_cleaned'], bins=50, kde=True) 
plt.title('Distribusi Jumlah Kata per Ulasan (Cleaned)')
plt.xlabel('Jumlah Kata')
plt.ylabel('Frekuensi')
plt.xlim(0, df['word_count_cleaned'].quantile(0.99))

plt.subplot(1, 2, 2)
sns.boxplot(y=df['word_count_cleaned'])
plt.title('Boxplot Jumlah Kata per Ulasan (Cleaned)')
plt.ylabel('Jumlah Kata')
plt.ylim(0, df['word_count_cleaned'].quantile(0.99))

plt.tight_layout()
plt.show()

# %%
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

stop_words_en = stopwords.words('english')

# %%
# Q5: Apa kata-kata (unigrams, bigrams) yang paling sering muncul secara keseluruhan?
corpus_all_cleaned = df['content_cleaned'].dropna().astype(str).str.cat(sep=' ')

words_all_cleaned = [word for word in corpus_all_cleaned.lower().split() if word.isalpha() and word not in stop_words_en]
word_counts_all_cleaned = Counter(words_all_cleaned)
most_common_words_cleaned = word_counts_all_cleaned.most_common(15) # Tampilkan 15 teratas
print(f"\nQ5: 15 Kata Paling Umum (Unigram) Keseluruhan (Cleaned & No Stopwords):\n{most_common_words_cleaned}")


# %%
# Visualisasi unigram umum
if most_common_words_cleaned:
    df_most_common_words_cleaned = pd.DataFrame(most_common_words_cleaned, columns=['Kata', 'Frekuensi'])
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Frekuensi', y='Kata', data=df_most_common_words_cleaned, palette='viridis')
    plt.title('15 Kata Paling Umum Keseluruhan (Cleaned)')
    plt.show()


# %%
# Word Cloud Keseluruhan
if corpus_all_cleaned.strip():
    wordcloud_all_cleaned = WordCloud(width=1000, height=500, background_color='white', stopwords=stop_words_en, max_words=100).generate(corpus_all_cleaned)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud_all_cleaned, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud Keseluruhan Ulasan (Cleaned)')
    plt.show()
else:
    print("Corpus 'content_cleaned' kosong, tidak bisa membuat word cloud keseluruhan.")


# %%
# Bigrams
valid_texts_for_bigram = df['content_cleaned'].dropna().astype(str)
if len(valid_texts_for_bigram) > 0:
    try:
        vectorizer_bigram_cleaned = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words_en, token_pattern=r'\b\w+\b', min_df=5) # min_df bisa disesuaikan
        X_bigram_cleaned = vectorizer_bigram_cleaned.fit_transform(valid_texts_for_bigram)
        bigram_counts_cleaned = X_bigram_cleaned.sum(axis=0)
        bigrams_freq_cleaned = [(word, bigram_counts_cleaned[0, idx]) for word, idx in vectorizer_bigram_cleaned.vocabulary_.items()]
        bigrams_freq_cleaned = sorted(bigrams_freq_cleaned, key = lambda x: x[1], reverse=True)
        print(f"\nQ5: 10 Bigram Paling Umum Keseluruhan (Cleaned & No Stopwords):\n{bigrams_freq_cleaned[:10]}")
    except ValueError as e:
        print(f"\nQ5: Tidak cukup data atau vocabulary kosong untuk membuat bigram: {e}")
else:
    print("\nQ5: Tidak ada data teks yang valid untuk analisis bigram.")

# %%
"""
### III. Analisis Label Sentimen (`is_positive` dan `sentiment_label`)
"""

# %%
# Q7: Bagaimana distribusi sentimen ulasan?
sentiment_distribution_cleaned = df['is_positive'].value_counts()
print(f"\nQ7: Distribusi Sentimen (dari 'is_positive'):\n{sentiment_distribution_cleaned}")


# %%
# Visualisasi menggunakan 'is_positive'
plt.figure(figsize=(7, 5))
sns.barplot(x=sentiment_distribution_cleaned.index, y=sentiment_distribution_cleaned.values, palette='pastel')
plt.title('Distribusi Sentimen Ulasan (Kolom "is_positive")')
plt.ylabel('Jumlah Ulasan')
plt.xlabel('Sentimen')
plt.show()

# %%
# Distribusi dari kolom 'sentiment_label' yang sudah di-map
sentiment_label_distribution = df['sentiment_label'].value_counts()
print(f"\nQ7: Distribusi Sentimen (dari 'sentiment_label' numerik):\n{sentiment_label_distribution}")

# %%
plt.figure(figsize=(7, 5))
sentiment_label_distribution.index = sentiment_label_distribution.index.map({v: k for k, v in label_mapping.items()})
sns.barplot(x=sentiment_label_distribution.index, y=sentiment_label_distribution.values, palette='pastel')
plt.title('Distribusi Sentimen Ulasan (Kolom "sentiment_label")')
plt.ylabel('Jumlah Ulasan')
plt.xlabel('Sentimen')
plt.show()

# %%
# Q8: Apakah ada label sentimen selain "Positive" dan "Negative" (atau 0 dan 1)?
unique_sentiments_text = df['is_positive'].unique()
unique_sentiments_numeric = df['sentiment_label'].unique()
print(f"\nQ8: Label Sentimen Unik (Teks 'is_positive'): {unique_sentiments_text}")
print(f"Q8: Label Sentimen Unik (Numerik 'sentiment_label'): {unique_sentiments_numeric}")



# %%
"""
### IV. Hubungan Antara Konten Teks dan Sentimen
"""

# %%
# Q9: Apakah ada perbedaan rata-rata panjang ulasan antara sentimen positif dan negatif?
mean_length_by_sentiment_cleaned = df.groupby('is_positive')['word_count_cleaned'].mean()
print(f"\nQ9: Rata-rata Jumlah Kata (Cleaned) per Sentimen:\n{mean_length_by_sentiment_cleaned}")


# %%
# Visualisasi perbandingan panjang ulasan
plt.figure(figsize=(8, 6))
sns.boxplot(x='is_positive', y='word_count_cleaned', data=df, palette='pastel')
plt.title('Perbandingan Jumlah Kata (Cleaned) berdasarkan Sentimen')
plt.xlabel('Sentimen')
plt.ylabel('Jumlah Kata (Cleaned)')
plt.ylim(0, df['word_count_cleaned'].quantile(0.95))
plt.show()

# %%
# Menganalisis kata-kata dan membuat word cloud.
def analyze_sentiment_words(df_sentiment, text_column_name, sentiment_value_text, stop_words_list, n_common_words=15, n_max_words_cloud=100):
    corpus_sentiment = df_sentiment[text_column_name].dropna().astype(str).str.cat(sep=' ')
    
    if not corpus_sentiment.strip():
        print(f"Tidak ada teks untuk dianalisis untuk sentimen '{sentiment_value_text}'.")
        return None

    words_sentiment = [word for word in corpus_sentiment.lower().split() if word.isalpha() and word not in stop_words_list]
    word_counts_sentiment = Counter(words_sentiment)
    most_common_sentiment = word_counts_sentiment.most_common(n_common_words)
    
    print(f"\n{n_common_words} Kata Paling Umum untuk Sentimen '{sentiment_value_text}':\n{most_common_sentiment}")

    # Visualisasi kataa
    if most_common_sentiment:
        df_most_common_sentiment = pd.DataFrame(most_common_sentiment, columns=['Kata', 'Frekuensi'])
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Frekuensi', y='Kata', data=df_most_common_sentiment, hue='Kata', palette='coolwarm', legend=False)
        plt.title(f'{n_common_words} Kata Paling Umum untuk Sentimen {sentiment_value_text}')
        plt.show()

    # Word Cloud
    if corpus_sentiment.strip():
        wordcloud_sentiment = WordCloud(width=1000, height=500, background_color='white', stopwords=stop_words_list, max_words=n_max_words_cloud).generate(corpus_sentiment)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud_sentiment, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud untuk Sentimen {sentiment_value_text}')
        plt.show()
    else:
        print(f"Corpus kosong setelah pembersihan, tidak bisa membuat Word Cloud untuk '{sentiment_value_text}'.")
        
    return most_common_sentiment

# %%
# Q10 & Q12: Kata-kata khas dan Word Cloud untuk ulasan Positif
print("\n--- Q10 & Q12: Analisis Ulasan Positif ---")
df_positive_cleaned = df[df['is_positive'] == 'Positive'].copy()

if not df_positive_cleaned.empty:
    analyze_sentiment_words(
        df_sentiment=df_positive_cleaned, 
        text_column_name='content_cleaned',
        sentiment_value_text='Positive', 
        stop_words_list=stop_words_en
    )
else:
    print("Tidak ada ulasan 'Positive' untuk dianalisis.")

# %%
# Q11 & Q12: Kata-kata khas dan Word Cloud untuk ulasan Negatif
print("\n--- Q11 & Q12: Analisis Ulasan Negatif ---")
df_negative_cleaned = df[df['is_positive'] == 'Negative'].copy()

if not df_negative_cleaned.empty:
    analyze_sentiment_words(
        df_sentiment=df_negative_cleaned, 
        text_column_name='content_cleaned',
        sentiment_value_text='Negative', 
        stop_words_list=stop_words_en
    )
else:
    print("Tidak ada ulasan 'Negative' untuk dianalisis.")

# %%
"""
## Penerapan Model AI (IBM Granite via Replicate)
"""

# %%
# Memuat token API dari file .env
load_dotenv()

api_token = os.environ.get('REPLICATE_API_TOKEN')

if api_token:
    os.environ["REPLICATE_API_TOKEN"] = api_token
    print("Token API Replicate berhasil dimuat dari .env.")
else:
    print("Variabel REPLICATE_API_TOKEN tidak ditemukan di file .env atau variabel lingkungan sistem.")

# %%
# Inisialisasi model Replicate jika token API tersedia.
if api_token:
    from langchain_community.llms import Replicate
    MODEL_ID = "ibm-granite/granite-3.3-8b-instruct"
    try:
        llm = Replicate(
            model=MODEL_ID,
        )
        print(f"Model {MODEL_ID} berhasil diinisialisasi.")
    except Exception as e:
        print(f"Error saat inisialisasi model Replicate: {e}")
        llm = None
else:
    llm = None
    print("Inisialisasi model dilewati karena API token tidak tersedia.")

# %%
prompt_template_sentiment_aspect = """
Analyze the following game review.
1. Determine the overall sentiment of the review. The sentiment must be strictly 'Positive' or 'Negative'.
2. List up to 3 key aspects or topics discussed in the review. Focus on aspects like gameplay, bugs, graphics, performance, or sound.
3. If an aspect is not clearly discussed or the review is too short, the aspects list can be 'N/A'.

Review: "{review_text}"

Provide the output STRICTLY in the following format, with each item on a new line:
Sentiment: [Positive/Negative]
Aspects: [Aspect1, Aspect2, Aspect3 or N/A]
"""

model_params = {
    "max_new_tokens": 150, 
    "temperature": 0.6,
    "top_p": 0.9,
    "repetition_penalty": 1.1 
}

print("Prompt template dan parameter model telah didefinisikan.")

# %%
def analyze_review_with_ai(review_text, prompt_template, model_parameters=None):

    if not llm: 
        print("Model LLM tidak terinisialisasi. Proses dibatalkan.")
        return None
    if not review_text or not review_text.strip():
        print("Teks ulasan kosong. Proses dibatalkan.")
        return None

    prompt = prompt_template.format(review_text=review_text)
    
    try:
        invoke_params = {}
        if model_parameters:
            for key, value in model_parameters.items():
                if key == 'max_tokens':
                    invoke_params['max_new_tokens'] = value
                elif key == 'temperature':
                    invoke_params['temperature'] = value
                elif key == 'top_k':
                    invoke_params['top_k'] = value
                elif key == 'top_p':
                    invoke_params['top_p'] = value
                elif key == 'repetition_penalty':
                    invoke_params['repetition_penalty'] = value

        if invoke_params:
            response = llm.invoke(prompt, **invoke_params)
        else:
            response = llm.invoke(prompt)
            
        return response.strip()
    except Exception as e:
        print(f"Error saat memanggil model AI untuk teks: '{review_text[:50]}...': {e}")
        return None

if 'df' in locals() and hasattr(df, 'empty') and not df.empty and 'llm' in locals() and llm:
    
    sample_size = min(50, len(df)) 
    sample_df = df.sample(n=sample_size, random_state=42).copy()

    ai_predicted_sentiments = []
    ai_extracted_aspects_list = []

    print(f"\nMemproses {len(sample_df)} sampel ulasan dengan AI (Model: {MODEL_ID})...")


    for index, row in sample_df.iterrows():
        review_text = str(row['content_cleaned']) 
        actual_label = row.get('is_positive', 'N/A') 

        print(f"\n----\nMenganalisis ulasan (ID Data Asli: {row.get('id', index)}, Label Asli: {actual_label}):")
        print(f"Teks Ulasan: {review_text[:150]}...") 

        ai_output = analyze_review_with_ai(review_text, prompt_template_sentiment_aspect, model_params)

        predicted_sentiment_from_ai = "Error/Parsing Failed"
        aspects_from_ai = "Error/Parsing Failed"

        if ai_output:
            print(f"Output Mentah AI: \n{ai_output}")
            try:
                lines = ai_output.split('\n')
                sentiment_found = False
                aspects_found = False
                temp_aspects = []

                for line in lines:
                    line_lower = line.lower().strip()
                    if line_lower.startswith("sentiment:"):
                        predicted_sentiment_from_ai = line.split(":", 1)[1].strip()
                        sentiment_found = True
                    elif line_lower.startswith("aspects:"):
                        aspect_str = line.split(":", 1)[1].strip()
                        if aspect_str.lower() != "n/a" and aspect_str:
                            temp_aspects = [aspect.strip() for aspect in re.split(r',|;', aspect_str) if aspect.strip()]
                        else:
                            temp_aspects = ["N/A"]
                        aspects_found = True
                
                if not sentiment_found:
                    print("Peringatan: Format 'Sentiment:' tidak ditemukan dalam output AI.")
                if not aspects_found:
                    print("Peringatan: Format 'Aspects:' tidak ditemukan dalam output AI.")
                
                aspects_from_ai = ", ".join(temp_aspects) if temp_aspects else "N/A"

            except Exception as e:
                print(f"Error saat parsing output AI: {e}")
        else:
            print("Tidak ada output dari AI atau terjadi error pada pemanggilan.")

        ai_predicted_sentiments.append(predicted_sentiment_from_ai)
        ai_extracted_aspects_list.append(aspects_from_ai)
        
        print(f"Hasil Parsing -> Sentimen AI: {predicted_sentiment_from_ai}, Aspek AI: {aspects_from_ai}")
        
        time.sleep(2)

    sample_df['ai_sentiment'] = ai_predicted_sentiments
    sample_df['ai_aspects'] = ai_extracted_aspects_list

    print("\n\n--- Hasil Analisis AI pada Sampel Data ---")
    print(sample_df[['id', 'content_cleaned', 'is_positive', 'ai_sentiment', 'ai_aspects']])
    
else:
    if not ('df' in locals() and hasattr(df, 'empty') and not df.empty):
        print("DataFrame 'df' tidak ditemukan atau kosong. Muat dan proses data Anda terlebih dahulu.")
    if not ('llm' in locals() and llm):
        print("Model LLM tidak diinisialisasi. Tidak dapat melanjutkan dengan analisis AI.")


# %%
print(sample_df[['id', 'content_cleaned', 'is_positive', 'ai_sentiment', 'ai_aspects']].head())

# %%
output_sampel = 'sampel_analisis.csv'
kolom_streamlit = ['id', 'content_cleaned', 'is_positive', 'ai_sentiment', 'ai_aspects']

for col in kolom_streamlit:
    if col not in sample_df.columns:
        sample_df[col] = "N/A"

try:
    sample_df_to_save = sample_df[kolom_streamlit]
    sample_df_to_save.to_csv(output_sampel, index=False)
    print(f"DataFrame sampel dengan hasil analisis AI telah disimpan ke '{output_sampel}'")
except KeyError as e:
    print(f"Error: Satu atau lebih kolom tidak ditemukan di sample_df: {e}")
    print(f"Kolom yang ada di sample_df: {sample_df.columns.tolist()}")
    print("Harap periksa kembali nama kolom di 'kolom_untuk_streamlit'.")
except Exception as e:
    print(f"Terjadi error saat menyimpan file CSV: {e}")