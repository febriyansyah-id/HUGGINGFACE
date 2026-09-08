"""
Download a model/dataset/spaces asset from the Hugging Face Hub into ./downloads/.

Example:
    python scripts/download_from_hub.py --kind model --repo bert-base-uncased
"""

from __future__ import annotations

import argparse
from pathlib import Path

OUT_ROOT = Path("./downloads")

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--kind", choices=["model", "dataset", "space"], required=True)
parser.add_argument("--repo", required=True, help="Remote repo-id, eg. google-bert/bert-base-uncased")
parser.add_argument("--filename", default=None, help="Optionally grab a single file.")
parser.add_argument("--revision", default="main", help="Branch/tag to checkout.")

ARGS = parser.parse_args()


def main(args: argparse.Namespace) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.kind == "model":
        from huggingface_hub import snapshot_download

        downloaded_at = snapshot_download(
            repo_id=args.repo,
            revision=args.revision,
            local_dir=OUT_ROOT / args.repo.replace("/", "__"),
            resume_download=True,
        )
        print(f"Saved model snapshots to {downloaded_at}")

    elif args.kind == "dataset":
        from datasets import load_dataset

        ds = load_dataset(args.repo, revision=args.revision)
        print(ds)
        print(f"Cached under {OUT_ROOT}/datasets_cache")

    elif args.kind == "space":
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=args.repo,
            repo_type="space",
            revision=args.revision,
            local_dir=OUT_ROOT / ("space-" + args.repo.replace("/", "__")),
        )
        print("Space cloned!")


if __name__ == "__main__":
    main(ARGS)
