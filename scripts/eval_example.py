"""
Smoke-test inference against a freshly-trained (or externally hosted) model.

Examples:
    python scripts/eval_example.py                          # uses ./exports/…
    python scripts/eval_example.py --model cardiffnlp/tweet-topic-21-multi
"""

from __future__ import annotations

import argparse
from pathlib import Path

LOCAL_EXPORT = Path("./exports/finetuned_distilbert_imdb")

DEMO_TEXT = [
    "Absolutely adored every second of this movie – stellar performances!",
    "Painfully boring dialogue and forgettable cast. Hard pass.",
    "Nice visuals although the finale dragged quite badly.",
    "Hands-down among the weakest films released this year.",
    "Quirky independent gem that rewards patient viewers.",
]

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--model", default=str(LOCAL_EXPORT), help="Either a local dir or a Hub repo-id.")


def main(arguments: argparse.Namespace) -> None:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TextClassificationPipeline,
    )

    tokeniser = AutoTokenizer.from_pretrained(arguments.model)
    backbone = AutoModelForSequenceClassification.from_pretrained(arguments.model)
    predictor = TextClassificationPipeline(
        task="text-classification",
        model=backbone,
        tokenizer=tokeniser,
        top_k=None,
    )

    scored_results = predictor(DEMO_TEXT, truncation=True, max_length=320)
    for phrase, ranked in zip(DEMO_TEXT, scored_results):
        winner = max(ranked, key=lambda kv: kv["score"])
        sign = "+" if "POSITIVE" in winner["label"].upper() else "\u2212"
        print(f"{sign} ({winner['score']:.1%})\t\u00ab{phrase[:56]}\u2026\u00bb  \u2794 {winner['label'].lower()}")


if __name__ == "__main__":
    main(parser.parse_args())
