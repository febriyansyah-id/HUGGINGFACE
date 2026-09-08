"""
Trainer pengklasifikasi PHISHING EMAIL dwibahasa (ID+EN) — TF-IDF murni.

ARSITEKTUR TERMODULAR:
  • Definisi `Featurizer` tinggal di `src/hfhub/textfeat.py` (shared module)
    sehingga objek yang di-pickle di dalam `model.joblib` dapat di-restore
    konsisten dari mana saja (lokal, Space HF, CI/CD).
  • Pelatihan: TF-IDF char_wb (2..4 gram) -> LogisticRegression(L2, liblinear).

OUTPUT:
  models/phishing_model/model.joblib   (payload: {featurizer, classifier, ..})
  models/phishing_model/metadata.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hfhub.textfeat import Featurizer  # noqa: E402

DATA_CSV = ROOT / "data" / "phishing_email" / "phishing_email_multilingual.csv"
ARTIFACT_DIR = ROOT / "models" / "phishing_model"
HELD_OUT_PORTION = 0.336
REPRO_SEED = 424242
OPERATIONAL_THRESHOLD = 0.488
LIMITATION_STAMP = "edu-defense-only"


def indices_partition(count: int, portion: float, seed: int):
    generator = np.random.default_rng(seed)
    ordering = generator.permutation(count)
    divider = int(round(count * portion))
    return ordering[:divider], ordering[divider:]   # (validation, training)


def main() -> None:
    cmd = argparse.ArgumentParser(description="Latih detector phishing (TF-IDF)")
    cmd.add_argument("--csv", default=str(DATA_CSV))
    cmd.add_argument("--out", default=str(ARTIFACT_DIR))
    opt = cmd.parse_args()

    table = pd.read_csv(opt.csv)
    body = table["text"].fillna("").astype(str).tolist()
    gold = table["label"].round().clip(upper=1).astype(int).values

    engine = Featurizer().fit(body)
    encoded = engine.transform(body)

    val_keys, train_keys = indices_partition(gold.size, HELD_OUT_PORTION, REPRO_SEED)
    xtrn, xval = encoded[train_keys], encoded[val_keys]
    ytrn, yval = gold[train_keys], gold[val_keys]

    learner = LogisticRegression(C=1.04, solver="liblinear", max_iter=1700,
                                 random_state=REPRO_SEED)
    learner.fit(xtrn, ytrn)

    risk = learner.predict_proba(xval)[:, 1]
    guess = (risk >= OPERATIONAL_THRESHOLD).astype(int)

    score_acc = accuracy_score(yval, guess)
    score_f1 = f1_score(yval, guess, average="macro")
    grid_err = confusion_matrix(yval, guess)

    outlet = Path(opt.out)
    outlet.mkdir(parents=True, exist_ok=True)

    envelope = {
        "manifest": "phishing-tfidf-v2-shared-featurizer",
        "featurizer_module": "hfhub.textfeat::Featurizer",
        "featurizer": engine,
        "learner": learner,
        "order": [0, 1],
        "decode": {0: "benign", 1: "phishing"},
        "gateway": OPERATIONAL_THRESHOLD,
    }
    with (outlet / "model.joblib").open("wb") as io_stream:
        pickle.dump(envelope, io_stream)

    ledger = {
        "blueprint": "TF-IDF(char_wb,(2,4)) + LogisticRegression(L2)",
        "origin": str(Path(opt.csv).resolve()),
        "records": int(table.shape[0]),
        "flagged_true": int(gold.sum()),
        "flag_free": int((gold == 0).sum()),
        "keepaside": {"rate": HELD_OUT_PORTION, "salt": REPRO_SEED},
        "partitions": {"learning": int(train_keys.size), "auditing": int(val_keys.size)},
        "benchmarks": {
            "precision_recall_area": None,
            "correct_rate": round(float(score_acc), 4),
            "harmonic_f1": round(float(score_f1), 4),
        },
        "mistake_grid": grid_err.tolist(),
        "go_noGo_at": OPERATIONAL_THRESHOLD,
        "guardrail": LIMITATION_STAMP,
        "credit": "Febriyansyah",
    }
    (outlet / "metadata.json").write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    print("\n======================= REKAP LATIH =======================")
    print(f"Asal korpus : {Path(opt.csv).resolve()}")
    print(f"Leksikon    : {engine.vocabulary_size():,} istilah")
    print(f"Irisan      : belajar={train_keys.size}  audiens={val_keys.size}")
    print(f"Akurasi     : {score_acc:.4f}")
    print(f"Makro-F1    : {score_f1:.4f}")
    print("Grid eror (audience):")
    print(grid_err)
    print(f"=> artefak @ {(outlet / 'model.joblib').resolve()}")


if __name__ == "__main__":
    main()
