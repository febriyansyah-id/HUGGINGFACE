---
language:
- id
- en
tags:
- phishing
- text-classification
- tf-idf
- logistic-regression
- multilingual
- security
library_name: sklearn
pipeline_tag: text-classification
metrics:
- accuracy
- f1
license: mit
datasets:
- Febriyansyah/phishing-emails-multilingual
widget:
- text: "URGENT!! Confirm your bank details NOW or lose access: secure-login.updates.network/login"
---

# Phishing Email Classifier — TF-IDF + Logistic Regression (ID/EN)

Model **ringan & CPU-friendly** untuk deteksi email phishing dwibahasa **Indonesia 🇮🇩 & English 🇺🇸** oleh [Febriyansyah](https://huggingface.co/Febriyansyah).

> ⚠️ **Synthetic & educational only** — jangan gunakan untuk aktivitas ilegal. Dataset & model dibuat untuk riset defensive security.

## Arsitektur

```
Text → TF-IDF char_wb (2–4 gram, 39k max, sublinear) → LogisticRegression (L2, liblinear, C=1.04)
```

- **Featurizer:** `hfhub.textfeat.Featurizer` — `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), max_features=39000, min_df=2, max_df=0.976, sublinear_tf=True)`
- **Classifier:** `LogisticRegression(C=1.04, solver='liblinear', max_iter=1700)`
- **Threshold operasional:** `0.488` (tuned, bukan 0.5) — `prob ≥ 0.488 ⇒ PHISHING`
- **Vocab aktual:** ~3.9k istilah (char_wb)
- **Payload:** `model.joblib` berisi `{featurizer, learner, gateway, decode, manifest}` — lihat `metadata.json` untuk audit trail.

Dirancang tanpa dependensi berat (tanpa torch/transformers) — inferensi <50ms di CPU, ideal untuk Spaces gratis & edge.

## Dataset Latih

[`Febriyansyah/phishing-emails-multilingual`](https://huggingface.co/datasets/Febriyansyah/phishing-emails-multilingual) — **600 email sintetis** (300 phishing / 300 benign), seimbang ID/EN, digenerate via `scripts/build_phishing_dataset.py` (seed 2026). Split: 398 train / 202 validation (33.6% hold-out, seed 424242).

## Evaluasi (hold-out 202 sampel)

| Metric | Value |
|--------|-------|
| Accuracy | **1.0000** |
| Macro-F1 | **1.0000** |
| Confusion Matrix | `[[93, 0], [0, 109]]` (TN, FP / FN, TP) |

> Hasil sempurna mencerminkan sifat sintetis & pola yang jelas — validasi di data real-world tetap diperlukan sebelum produksi.

## Penggunaan

### Python (lokal)

```bash
pip install scikit-learn numpy
python predict.py "URGENT: Verify your BCA account now at http://bca-secure.verify-center.cc/login"
# → [PHISHING] prob=0.97  (threshold 0.488)
```

```python
import pickle
from pathlib import Path

payload = pickle.load(open("model.joblib","rb"))
featurizer, clf, thr = payload["featurizer"], payload["learner"], payload["gateway"]
proba = clf.predict_proba(featurizer.transform(["Hello, your invoice is ready"]))[0,1]
print("phishing" if proba >= thr else "benign", proba)
```

### Hugging Face

```python
from huggingface_hub import hf_hub_download
import pickle
p = hf_hub_download("Febriyansyah/phishing-email-classifier", filename="model.joblib")
payload = pickle.load(open(p,"rb"))
```

## Keterbatasan & Etika

- Data sintetis — tidak mencakup semua variasi phishing dunia nyata (spear-phishing, BEC, dll).
- Bahasa terbatas ID/EN; performa di bahasa lain tidak terjamin.
- Guardrail: `edu-defense-only` — gunakan hanya untuk edukasi & pertahanan.

## Space Demo

Coba langsung di 🤗 Space: [`Febriyansyah/phishing-scout`](https://huggingface.co/spaces/Febriyansyah/phishing-scout) (Gradio, CPU).

## Reproduksi

```bash
python scripts/build_phishing_dataset.py
python scripts/train_phishing_model.py --csv data/phishing_email/phishing_email_multilingual.csv --out models/phishing_model
```

## Lisensi

MIT — lihat `LICENSE`.

## Citation

```bibtex
@misc{febriyansyah2026phishing,
  author = {Febriyansyah},
  title = {Phishing Email Classifier (TF-IDF + LogReg, ID/EN)},
  year = {2026},
  publisher = {Hugging Face},
  url = {https://huggingface.co/Febriyansyah/phishing-email-classifier}
}
```
