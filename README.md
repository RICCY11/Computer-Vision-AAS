# Computer-Vision-AAS

# 🚘 OCR Plat Nomor Kendaraan dengan Vision-Language Model (VLM)

Proyek ini bertujuan untuk mengeksplorasi kemampuan model **Vision-Language Model (VLM)** seperti **LLaVA** dan **Gemma 3-12B** dalam melakukan **Optical Character Recognition (OCR)** terhadap gambar plat nomor kendaraan Indonesia. Semua model dijalankan secara lokal menggunakan **LM Studio** sebagai backend inference API.

Selain proses inferensi, proyek ini juga mencakup evaluasi performa model dengan menggunakan metrik **Character Error Rate (CER)** untuk mengukur tingkat akurasi hasil prediksi.

---

## 🎯 Tujuan Proyek

- Menerapkan VLM (LLaVA & Gemma) untuk membaca teks dari gambar.
- Membangun pipeline inferensi OCR otomatis dengan Python.
- Mengevaluasi hasil prediksi model menggunakan CER.
- Membandingkan performa antara dua model multimodal.

---

## ⚙️ Teknologi dan Tools

- **Python 3.8+**
- **LM Studio** (lokal, port `1234`)
- **Model:**
  - `llava-v1.5-7b`
  - `google/gemma-3-12b` *(vision-enabled via LM Studio patch)*
- **Library:**
  - `pandas`
  - `tqdm`
  - `requests`
  - `Levenshtein`

---

## 📂 Struktur Proyek

- dataset/
- images/ # Gambar plat nomor
- label/labels.csv # Ground truth
- OCR_VLM.py # Script utama OCR
- Evaluasi_OCR.py # Script evaluasi CER
- Output_ocr.csv # Output prediksi model LLaVA
- Output_Gemma_ocr.csv # Output prediksi model Gemma
- README.md # Dokumentasi proyek


---

## 🚀 Cara Menjalankan

### 1. Jalankan LM Studio
Pastikan model vision telah diaktifkan, misalnya:
- `llava-v1.5-7b`
- `google/gemma-3-12b`

LM Studio harus berjalan di:
http://localhost:1234/v1/chat/completions


---

### 2. Jalankan Script OCR
python OCR_VLM.py
Catatan!:
Pastikan pada bagian MODEL_NAME = "llava" jika ingin menggunakan VLM llava dan ganti jadi "gemma" jika ingin menggunakan gemma

Script ini akan:
Membaca gambar dari folder images/test
Mengirim gambar sebagai base64 ke LM Studio
Menerima hasil prediksi dari model
Menyimpan ke CSV (Output_ocr.csv atau Output_Gemma_ocr.csv)


### 3. Evaluasi Hasil OCR
python Evaluasi_OCR.py
Script ini akan:
Catatan!:
Pastikan pada bagian csv_path = "Output_ocr.csv" jika ingin menggunakan hasil prediksi dari VLM llava dan ganti jadi "Output_Gemma_ocr.csv" jika ingin menggunakan hasil prediksi VLM gemma


Membaca file hasil prediksi
Menghitung:
Rata-rata CER
Median CER
Jumlah prediksi bagus (CER ≤ 0.10)
Jumlah prediksi gagal (CER > 0.10)
Menampilkan 10 contoh prediksi gagal

### 4. Contoh Hasil Evaluasi
📊 Evaluasi Hasil OCR
- Rata-rata CER     : 0.9485
- Median CER        : 0.5278
- Jumlah total data : 162
- Prediksi bagus    : 3 (1.85%)
- Prediksi gagal    : 148 (91.36%)

🛑 Contoh Prediksi Gagal (CER > 0.25):
        image     ground_truth     prediction     CER_score
  test001_2.jpg     B2407UZO            2471         0.625
  test003_1.jpg     B2634UZF          Bd2638         0.625
  test003_2.jpg     B1995JVK             BJK         0.625

