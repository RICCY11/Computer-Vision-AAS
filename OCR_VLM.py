import os
import requests
import base64
import pandas as pd
from tqdm import tqdm
import Levenshtein

# GANTI BAGIAN INI SESUAI DENGAN MODEL DAN LOKASI SERVER LMSTUDIO
LMSTUDIO_API = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "llava"  # atau "bakllava"

# GANTI DENGAN PATH SEBENARNYA
IMAGE_FOLDER = r"C:\Users\mgala\OneDrive\Documents\Machine Vision\Riccy\Dataset\archive\Indonesian License Plate Recognition Dataset\images\test"
LABEL_CSV_PATH = r"C:\Users\mgala\OneDrive\Documents\Machine Vision\Riccy\Dataset\archive\Indonesian License Plate Recognition Dataset\labels\labels.csv"
OUTPUT_CSV = "Output_ocr.csv"

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def get_prediction(image_path):
    base64_image = encode_image(image_path)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "What is the license plate number shown in this image? Respond only with the plate number."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(LMSTUDIO_API, json=payload)
        response.raise_for_status()
        result = response.json()

        if "choices" not in result:
            print("⚠️ Unexpected response format:", result)
            return "ERROR"

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print(f"❌ Error saat memproses gambar {image_path}: {e}")
        return "ERROR"

def cer(gt, pred):
    if gt == "" or pred == "ERROR":
        return 1.0
    S = Levenshtein.distance(gt, pred)
    return S / len(gt) if len(gt) > 0 else 1.0

def process_dataset(img_dir, label_csv_path):
    # Baca file CSV label
    labels_df = pd.read_csv(label_csv_path)
    labels_df['image'] = labels_df['image'].astype(str).str.strip().str.lower()
    labels_df['ground_truth'] = labels_df['ground_truth'].astype(str).str.strip()

    label_dict = dict(zip(labels_df['image'], labels_df['ground_truth']))

    results = []
    for fname in tqdm(os.listdir(img_dir)):
        if not fname.lower().endswith(".jpg"):
            continue

        fname_clean = fname.strip().lower()

        if fname_clean not in label_dict:
            print(f"⚠️ {fname} tidak ditemukan di labels.csv, dilewati.")
            continue

        img_path = os.path.join(img_dir, fname)
        gt = label_dict[fname_clean]
        pred = get_prediction(img_path)
        cer_score = cer(gt, pred)

        results.append({
            "image": fname,
            "ground_truth": gt,
            "prediction": pred,
            "CER_score": cer_score
        })

    return results

if __name__ == "__main__":
    print("🔍 Memproses dataset menggunakan label dari CSV...")
    data = process_dataset(IMAGE_FOLDER, LABEL_CSV_PATH)
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Hasil disimpan di: {OUTPUT_CSV}")
