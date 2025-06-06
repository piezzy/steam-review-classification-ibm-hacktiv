# IBM x Hacktiv8: Klasifikasi dan Ekstraksi Aspek Data Ulasan Game Steam Menggunakan IBM Granite

## Ringkasan Proyek (Project Overview)

Proyek *capstone* ini merupakan bagian dari program IBM x Hacktiv8 yang bertujuan untuk menganalisis ulasan game dari platform Steam guna mendapatkan wawasan berharga. Fokus utama adalah pada klasifikasi sentimen (Positif/Negatif) dan ekstraksi aspek-aspek kunci (seperti *gameplay, bugs, graphics, performance*) yang sering dibicarakan oleh pemain. Dengan volume ulasan yang sangat besar, analisis manual menjadi tidak efisien. Oleh karena itu, proyek ini memanfaatkan kemampuan model AI Generatif IBM Granite (`granite-3.3-8b-instruct`) yang diakses melalui Replicate untuk mengotomatisasi proses analisis teks ini.

Pendekatan proyek meliputi beberapa tahapan utama:
1.  **Pengumpulan dan Pembersihan Data:** Dataset ulasan game Steam dari Kaggle dibersihkan dan diproses untuk menghilangkan *noise* dan menstandarisasi teks.
2.  **Analisis Data Eksploratif (EDA):** Memahami karakteristik data ulasan, distribusi sentimen awal, dan pola teks umum.
3.  **Implementasi AI:** Merancang *prompt* yang efektif dan menerapkan model IBM Granite untuk melakukan klasifikasi sentimen dan ekstraksi aspek pada ulasan.
4.  **Analisis Hasil AI:** Mengevaluasi output dari model AI dan menggali *insight*.
5.  **Pengembangan Aplikasi Demo:** Membuat aplikasi web interaktif menggunakan Streamlit untuk mendemonstrasikan kemampuan model AI dalam menganalisis ulasan baru dan menampilkan hasil analisis sampel.

Tujuan akhir dari proyek ini adalah untuk menunjukkan bagaimana AI dapat digunakan untuk mengubah data teks ulasan yang tidak terstruktur menjadi *insight* yang terstruktur dan dapat ditindaklanjuti, yang bermanfaat bagi pengembang game dalam meningkatkan produk mereka dan bagi pemain dalam membuat keputusan.

## Link Dataset Mentah (Raw Dataset Link)

* **Nama Dataset:** Steam Review & Games Dataset
* **Sumber:** Kaggle
* **Link:** [https://www.kaggle.com/datasets/filipkin/steam-reviews](https://www.kaggle.com/datasets/filipkin/steam-reviews) *

## *Insight* & Temuan Utama (Insight & Findings)


* **Distribusi Sentimen:**
    * Berdasarkan analisis AI pada sampel data, X% ulasan diklasifikasikan sebagai Positif dan Y% sebagai Negatif.
    * Terdapat perbedaan/kesamaan dengan distribusi sentimen pada label asli dataset.
* **Aspek Kunci yang Mempengaruhi Sentimen:**
    * Aspek seperti `[Contoh Aspek 1 dari hasil Anda, misal: "bugs"]` dan `[Contoh Aspek 2, misal: "performance issues"]` secara signifikan berkontribusi terhadap sentimen negatif.
    * Sebaliknya, `[Contoh Aspek 3, misal: "engaging gameplay"]` dan `[Contoh Aspek 4, misal: "stunning graphics"]` seringkali ditemukan pada ulasan dengan sentimen positif.
* **Tantangan Analisis AI:**
    * Model AI menunjukkan tantangan dalam menginterpretasikan ulasan yang sangat pendek, ambigu, atau menggunakan sarkasme.
    * Identifikasi ulasan yang tidak relevan (misalnya, ulasan bukan tentang game) oleh AI menjadi temuan menarik, menunjukkan pemahaman konteks model berdasarkan *prompt*.

## Penjelasan Dukungan AI (AI Support Explanation)

Proyek ini memanfaatkan kecerdasan buatan (AI) untuk melakukan analisis mendalam terhadap data teks ulasan game. Berikut adalah detail dukungan AI yang digunakan:

* **Model AI yang Digunakan:**
    * **IBM Granite (`ibm-granite/granite-3.3-8b-instruct`)**: Model bahasa besar (LLM) dari IBM yang dipilih karena kemampuannya dalam pemahaman bahasa alami, klasifikasi teks, dan ekstraksi informasi.
* **Platform Akses Model:**
    * Model IBM Granite diakses melalui **API dari Replicate**, yang menyediakan platform untuk menjalankan model *machine learning* di cloud.
* **Lingkungan Eksekusi:**
    * **Notebook:** Digunakan untuk seluruh proses analisis data, mulai dari *preprocessing*, EDA, hingga implementasi pemanggilan model AI dan analisis hasilnya.
    * **Python:** Bahasa pemrograman utama yang digunakan, dengan *library* seperti Pandas, NLTK, Scikit-learn, dan Langchain.
* **Cara Penggunaan AI:**
    1.  **Klasifikasi Sentimen:** Model AI diinstruksikan melalui *prompt* yang dirancang khusus untuk menentukan apakah sentimen dari sebuah ulasan game bersifat "Positif" atau "Negatif".
    2.  **Ekstraksi Aspek:** Selain sentimen, model juga diminta untuk mengidentifikasi dan mendaftar hingga 3 aspek atau topik kunci yang dibahas dalam ulasan, yang terkait dengan kategori seperti *gameplay, bugs, graphics,* atau *performance*.
    3.  ***Prompt Engineering*:** Proses perancangan dan iterasi *prompt* dilakukan untuk mendapatkan output yang paling akurat dan sesuai dengan format yang diinginkan. Contoh *prompt* final yang digunakan:
        ```
        Analyze the following game review.
        Determine the sentiment (Positive or Negative) and list up to 3 key aspects or topics discussed in the review related to gameplay, bugs, graphics, or performance.
        If an aspect is not clearly discussed or the review is too short, aspects can be "N/A".

        Review: "{review_text}"

        Output in the following format:
        Sentiment: [Positive/Negative]
        Aspects: [Aspect1, Aspect2, Aspect3 or N/A]
        ```
* **Relevansi dan Nilai Tambah AI:**
    * **Otomatisasi dan Skalabilitas:** AI memungkinkan analisis ribuan ulasan secara otomatis, sesuatu yang tidak praktis dilakukan manual.
    * **Ekstraksi Informasi Terstruktur:** Mengubah data teks ulasan yang tidak terstruktur menjadi informasi terstruktur (sentimen, daftar aspek) yang lebih mudah dianalisis dan dipahami.
    * **Objektivitas (dengan catatan):** AI dapat memberikan analisis yang lebih konsisten dibandingkan analis manusia yang berbeda-beda, meskipun bias dalam data training model tetap perlu dipertimbangkan.
    * **Penemuan *Insight*:** Membantu mengungkap pola dan tema yang mungkin tersembunyi dalam data ulasan.

Proyek ini menunjukkan bagaimana LLM seperti IBM Granite dapat menjadi alat yang ampuh untuk mendapatkan *insight* mendalam dari data tekstual dalam domain ulasan produk.

---

## Cara Menggunakan Repositori Ini

### 1. *Notebook* Utama

* *Notebook* utama proyek untuk analisis data dan pemodelan AI dapat ditemukan di: `steam_review_classification.ipynb`.

### 2. Menjalankan Aplikasi Demo Streamlit (`app.py`)

* **Instal Dependensi:**
    Pastikan semua *library* yang dibutuhkan sudah terinstal. Anda bisa menggunakan perintah berikut:
    ```bash
    pip install -r requirements.txt
    ```

* **Atur API Token:**
    1.  Buat file bernama `.env` di direktori utama repositori ini.
    2.  Tambahkan baris berikut ke dalam file `.env` dan ganti `MASUKAN_API_TOKEN_DI_SINI` dengan Replicate API Token Anda yang valid:
        ```env
        REPLICATE_API_TOKEN=MASUKAN_API_TOKEN_DI_SINI
        ```

* **Jalankan Aplikasi:**
    Buka terminal atau *command prompt*, arahkan ke direktori repositori, lalu jalankan:
    ```bash
    streamlit run app.py
    ```

### 3. Dataset

* **Dataset Mentah:**
    Dapat diakses melalui link yang disediakan di bagian "[Link Dataset Mentah](https://www.kaggle.com/datasets/filipkin/steam-reviews)" di atas.

* **Data Sampel Teranalisis:**
    File CSV hasil pembersihan dan analisis AI pada sampel data (`sampel_analisis.csv`) disertakan dalam repositori ini dan digunakan oleh aplikasi Streamlit untuk menampilkan contoh.
---