"""
Modul fitur tekstual bersama untuk model phishing.

Berisi kelas `Featurizer` (wrapper TF-IDF char_word-boundaries) yang
digunakan baik saat training maupun inference. Menempatkannya di modul
level-package memastikan objek yang di-pickle di dalam ``model.joblib``
dapat di-restore konsisten dari mana saja (lokal, Space HF, CI/CD).
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


class Featurizer:
    """Pembungkus TF-IDF analisis huruf berdasar kata (char_wb).

    Kelas modul top-level sehingga aman untuk pickle lintas-environment.
    Parameter dijaga immutable lewat atribut ``params`` untuk audit trail.
    """

    DEFAULT_KWARGS = dict(
        ngrams=(2, 4),
        max_features=39000,
        min_df=2,
        max_df=0.976,
        sublinear=True,
    )

    def __init__(self, **override):
        merged = {**self.DEFAULT_KWARGS, **override}
        self.params = {
            "ngrams": tuple(merged.pop("ngrams")),
            "max_features": int(merged.pop("max_features")),
            "min_df": int(merged.pop("min_df")),
            "max_df": float(merged.pop("max_df")),
            "sublinear": bool(merged.pop("sublinear")),
        }
        ng = self.params["ngrams"]
        self.core = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=ng,
            lowercase=True,
            strip_accents="unicode",
            max_df=self.params["max_df"],
            min_df=self.params["min_df"],
            norm="l2",
            sublinear_tf=self.params["sublinear"],
            max_features=self.params["max_features"],
        )

    def fit(self, corpus):
        self.core.fit(corpus)
        return self

    def transform(self, corpus):
        return self.core.transform(corpus)

    def vocabulary_size(self) -> int:
        vocab = getattr(self.core, "vocabulary_", None)
        return len(vocab) if vocab else 0

    def describe(self) -> dict:
        return {"family": "tfidf_charWB", **self.params,
                "actual_vocab": self.vocabulary_size()}
