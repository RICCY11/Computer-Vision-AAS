import pandas as pd

# Ganti jika file CSV hasilmu pakai nama lain atau ada di folder berbeda
csv_path = "Output_ocr.csv"

# Load data
df = pd.read_csv(csv_path)

# Hitung metrik dasar
rata2 = df["CER_score"].mean()
median = df["CER_score"].median()

# Kriteria
bagus = df[df["CER_score"] <= 0.10]
gagal = df[df["CER_score"] > 0.10]

# Tampilkan hasil
print("📊 Evaluasi Hasil OCR")
print(f"- Rata-rata CER     : {rata2:.4f}")
print(f"- Median CER        : {median:.4f}")
print(f"- Jumlah total data : {len(df)}")
print(f"- Prediksi bagus    : {len(bagus)} ({len(bagus)/len(df)*100:.2f}%)")
print(f"- Prediksi gagal    : {len(gagal)} ({len(gagal)/len(df)*100:.2f}%)")

print("\n🛑 Contoh Prediksi Gagal (CER > 0.1):")
print(gagal[["image", "ground_truth", "prediction", "CER_score"]].head(10).to_string(index=False))
