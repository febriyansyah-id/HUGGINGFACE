"""
End-to-end demonstration: fine-tune a compact Transformer classifier on IMDb
using 🤗 Transformers + Accelerate, then (optionally) push the result to the Hub.

Quick start (GPU recommended):

    python scripts/train_example.py                     # basic run
    python scripts/train_example.py --epochs 3 --fp16   # tuned hyperparams
    python scripts/train_example.py \\
        --push-final-adapter USERNAME/imdb-distilbert   # ship to the Hub

Everything sits behind sane defaults so you can experiment freely.
"""

from __future__ import annotations

import argparse
from pathlib import Path

BASE_MODEL = "distilbert/distilbert-base-uncased"
DATA_LOCAL_DIR = "./data/imdb_mini"
CHECKPOINTS_OUT = Path("./models/")
LOGGING_INTERVAL = 90

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--learning-rate", type=float, default=2e-5)
parser.add_argument("--batch-size", type=int, default=28)
parser.add_argument("--sequence-length", type=int, default=192)
parser.add_argument("--subset-count", type=int, default=-1, help="-1 keeps the FULL train split")
parser.add_argument("--mixed-precision", choices=[None, "fp16", "bf16"], default=None)
parser.add_argument("--push-final-adapter", metavar="OWNER/NAME", default=None,
                    help="Namespace/id to publish the finetuned adapter to.")
ARGS = parser.parse_args()


def main(_argv=None):
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    SEED = 139
    NUM_LABELS = 2

    # ---------------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------------
    splits = load_dataset("imdb", split={"train": "train", "validation": "test"})

    if ARGS.subset_count > 0:
        splits["train"] = splits["train"].shuffle(seed=SEED).select(range(ARGS.subset_count))
        splits["validation"] = splits["validation"].shuffle(seed=SEED).select(range(ARGS.subset_count // 4))

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def encode_rows(group):
        encoded = tokenizer(
            group["text"],
            truncation=True,
            max_length=ARGS.sequence_length,
            padding="max_length",
        )
        encoded["label"] = group["label"]
        return encoded

    processed = splits.map(encode_rows, batched=True, remove_columns=["text"])

    # ---------------------------------------------------------------
    # 2. Instantiate model
    # ---------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=NUM_LABELS,
        hidden_dropout_prob=0.045,
        attention_probs_dropout_prob=0.035,
    )

    # ---------------------------------------------------------------
    # 3. Configure training
    # ---------------------------------------------------------------
    mp = ARGS.mixed_precision or ("bf16" if torch.cuda.is_bf16_supported() else None)

    conf = TrainingArguments(
        output_dir=str(CHECKPOINTS_OUT),
        overwrite_output_dir=True,
        num_train_epochs=ARGS.epochs,
        per_device_train_batch_size=ARGS.batch_size,
        gradient_accumulation_steps=2,
        weight_decay=0.015,
        learning_rate=ARGS.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.065,
        optim="adamw_torch",
        logging_first_step=True,
        logging_steps=LOGGING_INTERVAL,
        eval_strategy="steps",
        eval_steps=180,
        save_strategy="best",
        greater_is_better=True,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True,
        bf16=(mp == "bf16"),
        fp16=(mp == "fp16"),
        tf32=True,
        report_to=[],
        seed=SEED,
    )

    def compute_metrics(outcome):
        probs, truth = outcome.predictions, outcome.label_ids
        hits = (truth.flatten() == probs.argmax(axis=-1)).sum()
        return {"accuracy": float(hits / max(len(truth), 1))}

    stopper = EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.002)

    trainer = Trainer(
        model=model,
        args=conf,
        train_dataset=processed["train"],
        eval_dataset=processed["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[stopper],
    )

    # ---------------------------------------------------------------
    # 4. Train!
    # ---------------------------------------------------------------
    trainer.train()

    EXPORT_DIR = Path("./exports/finetuned_distilbert_imdb")
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))
    print(f"\n\u2713 Final model bundle saved to {EXPORT_DIR.resolve()}")

    # ---------------------------------------------------------------
    # 5. Ship to the Hub?
    # ---------------------------------------------------------------
    if ARGS.push_final_adapter:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=ARGS.push_final_adapter,
            folder_path=str(EXPORT_DIR),
            commit_title="DistilBERT fine-tuned on IMDb (personal demo)",
        )
        print(f"\U0001F680 Published \u2192 https://huggingface.co/{ARGS.push_final_adapter}")


if __name__ == "__main__":
    main()
