"""
Push a trained artefact (folder) to the Hugging Face Hub.

Usage:
    python scripts/push_to_hub.py \
      --kind model \
      --input-path ./experiments/runs/v1 \
      --repo febriyansyah/text-classifier-v1 \
      --public
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_KIND_TO_TYPE = {"model": "model", "dataset": "dataset", "space": "space"}

parser = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--kind", choices=list(REPO_KIND_TO_TYPE.keys()), required=True)
parser.add_argument("--input-path", type=Path, required=True, help="Local folder to upload.")
parser.add_argument("--repo", required=True, help="Destination repo-id, eg. myuser/myproj:v1")
parser.add_argument("--public", action="store_true", help="Default behaviour creates PRIVATE repos.")

ARGS = parser.parse_args()


def main(args: argparse.Namespace) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    repo_type = REPO_KIND_TO_TYPE[args.kind]

    # Split eventual ':tag'
    tag = None
    repo_full = args.repo
    if ":" in args.repo:
        repo_full, tag = args.repo.split(":", 1)

    api.create_repo(
        repo_id=repo_full,
        repo_type=repo_type,
        private=(not args.public),
        exist_ok=True,
    )

    uploaded = []
    for candidate in sorted((args.input_path).iterdir()) if args.input_path.exists() else []:
        if candidate.is_file():
            resp = api.upload_file(
                path_or_fileobj=candidate.as_posix(),
                path_in_repo=candidate.name,
                repo_id=repo_full,
                repo_type=repo_type,
            )
            uploaded.append(resp.commit_hash)
        elif candidate.is_dir():
            resp = api.upload_folder(
                repo_id=repo_full,
                repo_type=repo_type,
                folder_path=candidate.as_posix(),
            )
            uploaded.append(resp.commit_hash)

    latest = uploaded[-1] if uploaded else None
    if tag and latest:
        api.tag_version(
            repo_id=repo_full,
            repo_type=repo_type,
            tag=tag,
            target_commitish=latest,
        )
    print(f"DONE uploading {len(uploaded)} commits to {repo_full}/{repo_type}:{tag or ''}")


if __name__ == "__main__":
    main(ARGS)
