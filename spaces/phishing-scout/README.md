---
title: Phishing Scout
emoji: 🕵️‍♂️
colorFrom: rose
colorTo: stone
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
- phishing
- gradio
- text-classification
---

# 🕵️‍♂️ Phishing Scout — TF-IDF + LogReg (ID/EN)

Demo interaktif **CPU-only** untuk deteksi email phishing dwibahasa **Indonesia 🇮🇩 & English 🇺🇸**.

Dibuat oleh **Febriyansyah** sebagai showcase end-to-end Hugging Face: *dataset → model → Space* — semua di [huggingface.co/Febriyansyah](https://huggingface.co/Febriyansyah).

## Cara kerja

1. Tempel badan email (ID/EN) ke kotak teks.
2. Backend: `TF-IDF char_wb (2–4 gram)` → `LogisticRegression` (threshold **0.488**).
3. Output: `⛔ PHISHING` jika `prob ≥ 0.488`, else `✅ Aman`.
4. Inferensi <50ms di CPU — tanpa torch/transformers.

## Aset

`assets/model.joblib` berisi `{featurizer, learner, gateway, decode}` (diproduksi oleh `scripts/train_phishing_model.py`). Space juga membundel `hfhub/textfeat.py` agar unpickle konsisten. Jika aset hilang, UI menampilkan peringatan.

## Etika & Batasan

- **Synthetic & edu-only** — jangan dipakai untuk kampanye nyata.
- Akurasi 1.0 di hold-out sintetis ≠ performa real-world; validasi lanjutan wajib sebelum produksi.
- Bahasa terbatas ID/EN.

## Lihat juga

- 📦 Dataset: [`Febriyansyah/phishing-emails-multilingual`](https://huggingface.co/datasets/Febriyansyah/phishing-emails-multilingual)
- 🤖 Model: [`Febriyansyah/phishing-email-classifier`](https://huggingface.co/Febriyansyah/phishing-email-classifier)
