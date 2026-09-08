#!/usr/bin/env python
"""
Inference untuk `Febriyansyah/phishing-email-classifier` — TF-IDF char_wb + LogReg.

Payload: model.joblib = {featurizer: hfhub.textfeat.Featurizer, learner: LogisticRegression, gateway: float}
Threshold operasional 0.488 (bukan 0.5).

Penggunaan:
    python predict.py "tempel-teks-email-di-sini"
    python predict.py --root ./models/phishing_model "Hello world"
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

# Pastikan hfhub.textfeat dapat di-import saat unpickle (lokal & Hub)
ROOT = Path(__file__).resolve().parents[1] if (Path(__file__).parent / "predict.py").exists() else Path.cwd()
for cand in [ROOT / "src", Path(__file__).resolve().parents[2] / "src", Path.cwd() / "src"]:
    if cand.exists() and str(cand) not in sys.path:
        sys.path.insert(0, str(cand))


def load_payload(model_root: Path):
    joblib = model_root / "model.joblib"
    meta = model_root / "metadata.json"
    if not joblib.exists():
        raise SystemExit(f"❌ '{joblib}' tidak ditemukan. Jalankan train_phishing_model.py dulu.")
    with joblib.open("rb") as fh:
        pl = pickle.load(fh)
    cfg = {}
    if meta.exists():
        import json
        cfg = json.loads(meta.read_text())
    return pl, cfg


def classify_one(root: Path, text: str) -> tuple[int, float, float]:
    """Return (label, proba_phishing, threshold)."""
    pl, _ = load_payload(root)
    featurizer = pl["featurizer"]
    clf = pl["learner"]
    thr = float(pl.get("gateway", 0.488))
    vec = featurizer.transform([text])
    proba = float(clf.predict_proba(vec)[0, 1])
    label = int(proba >= thr)
    return label, proba, thr


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict phishing probability")
    ap.add_argument("text", nargs="?", help="badan email")
    ap.add_argument("--root", default=str(Path(__file__).parent.parent /
                                          ".." / "models" / "phishing_model"))
    args = ap.parse_args(argv)

    txt = args.text or ""
    if not txt.strip():
        txt = ("URGENT!! Your account will be suspended today. Verify now: "
               "http://secure-login.bank-update.info/panel")

    lbl, pr, thr = classify_one(Path(args.root), txt)
    tag = "PHISHING" if lbl == 1 else "AMAN"
    print(f"\n[{tag}] prob-phishing={pr:.4f} (thr={thr:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
