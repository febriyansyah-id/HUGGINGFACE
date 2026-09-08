"""
🌐 Phishing Scout — Live Demo Space
====================================
Classifier phishing-email dwibahasa (ID + EN) — TF-IDF char_wb + LogisticRegression.
Ringan, CPU-only, tanpa torch/transformers. Threshold 0.488.

Dipublikasikan dari workspace lokal `huggingface-workspace` milik Febriyansyah.
Lihat model & dataset pendamping di profil: https://huggingface.co/Febriyansyah
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import gradio as gr

# Pastikan Featurizer (hfhub.textfeat) bisa di-unpickle di Space
for cand in [Path(__file__).parent / "hfhub", Path(__file__).parent / "src" / "hfhub", Path(__file__).parent]:
    if cand.exists() and str(cand.parent) not in sys.path:
        sys.path.insert(0, str(cand.parent))

# --- lokasi artefak model (relatif terhadap root Space) -------------------
CANDIDATES = [
    Path(__file__).parent / "assets" / "model.joblib",
    Path(__file__).parent / "model.joblib",
    Path(__file__).parent / "models" / "phishing_model" / "model.joblib",
]
MODEL_JOBLIB = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])

_ASSETS_NOTICE = (
    "ℹ️ Asset model belum terpasang di Space ini. "
    "Pastikan `assets/model.joblib` ikut ter-upload (lihat README)."
)


def _load_payload():
    if not MODEL_JOBLIB.exists():
        raise FileNotFoundError(_ASSETS_NOTICE)
    with MODEL_JOBLIB.open("rb") as fh:
        return pickle.load(fh)


PAYLOAD = _load_payload()
FEATURIZER = PAYLOAD["featurizer"]
CLASSIFIER = PAYLOAD["learner"]
THRESHOLD = float(PAYLOAD.get("gateway", 0.488))
DECODE = PAYLOAD.get("decode", {0: "benign", 1: "phishing"})


def predict(email_text: str) -> tuple[str, float]:
    if not email_text or not email_text.strip():
        return "⚠️ Input kosong", 0.0
    vec = FEATURIZER.transform([email_text])
    proba_phish = float(CLASSIFIER.predict_proba(vec)[0, 1])
    verdict = "⛔ PHISHING" if proba_phish >= THRESHOLD else "✅ Aman / Legitimate"
    return verdict, round(proba_phish, 4)


BLOCK = '''
<div style="font-family:sans-serif;padding:1px;">
<h2 style="margin-bottom:2px;">🕵️‍♂️ Phishing Scout</h2>
<p><small>Deteksi email phising dwibahasa (ID&nbsp;&middot;&nbsp;EN) ·
<b>Febriyansyah Research Lab</b></small></p>
</div>
'''

interface = gr.Interface(
    fn=predict,
    title="Phishing Scout",
    description=BLOCK,
    theme=gr.themes.Base(primary_hue="rose", neutral_hue="stone"),
    inputs=gr.Textbox(lines=10, placeholder="Tempel badan email di sini… (Indonesia / English)"),
    outputs=[
        gr.Label(label="Putusan"),
        gr.Number(label="Probabilitas Phishing (0–1)"),
    ],
    examples=[
        ["Urgent: Klaim Rp 2.500.000 reward Anda sekarang! Kirim PIN ATM via tautan bit.ly/xyz"],
        ["Notification: monthly newsletter March edition is ready on our official blog."],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    interface.launch()
