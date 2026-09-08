---
language:
- id
- en
tags:
- phishing
- spam
- text-classification
- security
- synthetic
license: mit
task_categories:
- text-classification
size_categories:
- n<1K
pretty_name: Phishing Emails Multilingual (ID/EN) — Synthetic
---

# Phishing Emails Multilingual (ID/EN) — Synthetic

Dataset **sintetis & edukatif** 600 email dwibahasa **Indonesia 🇮🇩 & English 🇺🇸** untuk riset **deteksi phishing** — oleh [Febriyansyah](https://huggingface.co/Febriyansyah).

> ⚠️ **Synthetic & edu-defense-only** — dibuat untuk pembelajaran defensive security, bukan untuk kampanye nyata. Jangan gunakan untuk aktivitas ilegal.

## Ringkasan

- **600 baris** — 300 phishing / 300 benign (seimbang), 321 EN / 279 ID
- **Kolom:** `id` (int), `language` (`id`/`en`), `text` (string, badan email), `label` (0=benign, 1=phishing)
- **Generator:** `scripts/build_phishing_dataset.py` (seed 2026) — pola linguistik phishing nyata (urgensi, spoofing institusi, URL mencurigakan, permintaan kredensial) vs email legitimasi rutin
- **Split rekomendasi:** 398 train / 202 validation (33.6% hold-out, seed 424242) — lihat `models/phishing_model/metadata.json` di repo model

## Contoh

```csv
id,language,text,label
379,id,"Subject: Newsletter Bulanan Investasi ...",0
73,en,"Hello Esteemed Client, Your account will be SUSPENDED ...",1
```

```python
from datasets import load_dataset
ds = load_dataset("Febriyansyah/phishing-emails-multilingual")
print(ds["train"][0])
# {'id': 379, 'language': 'id', 'text': '...', 'label': 0}
```

## Pembuatan

```bash
python scripts/build_phishing_dataset.py
# → data/phishing_email/phishing_email_multilingual.csv
```

Institusi spoof: BCA, Mandiri, BNI, PayPal, Netflix, Apple, Microsoft, dll. URL phishing disintesis (`*.verify-center.cc`, `*.updater-service.online`, dll) vs URL resmi (`klikbca.com`, `paypal.com`, dll).

## Penggunaan & Etika

- Cocok untuk: fine-tune classifier biner, demo TF-IDF/LogReg, evaluasi threshold, Space Gradio.
- **Batasan:** sintetis — tidak mencakup spear-phishing/BEC dunia nyata; bahasa terbatas ID/EN; butuh validasi di data real sebelum produksi.
- Lisensi: MIT. Guardrail: `edu-defense-only`.

## Terkait

- 🤖 Model: [`Febriyansyah/phishing-email-classifier`](https://huggingface.co/Febriyansyah/phishing-email-classifier) (TF-IDF char_wb + LogReg, thr 0.488)
- 🕵️ Space: [`Febriyansyah/phishing-scout`](https://huggingface.co/spaces/Febriyansyah/phishing-scout)

## Citation

```bibtex
@misc{febriyansyah2026phishingdata,
  author = {Febriyansyah},
  title = {Phishing Emails Multilingual (Synthetic, ID/EN)},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/datasets/Febriyansyah/phishing-emails-multilingual}
}
```
