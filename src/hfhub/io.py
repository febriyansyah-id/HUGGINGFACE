"""Helpers for shipping artefacts to / fetching resources from the Hub."""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

RIGHT_ARROW = "\u2192"


def utc_now_compact() -> str:
    """Compact UTC timestamp usable in filenames (safe chars only)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S.%fZ")


def save_json(content: Mapping[str, Any], path: str | Path, indent: int = 2) -> None:
    """Persist a mapping to UTF-8 JSON, atomically-ish."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handler:
        json.dump(content, handler, indent=indent, sort_keys=True, ensure_ascii=False)


def save_locally_or_push_to_hub(
    export_content: Mapping[str, Any],
    destination: str,
    commit_message: str = "",
    quiet: bool = False,
) -> None:
    """
    Write ``export_content`` somewhere useful.

    * If ``destination`` resembles a Hub repo-id (contains '/'), push it as a
      small JSON blob attached to that repo.
    * Otherwise treat it as a local file stem and emit ``<stem>.json`` beside it.
    """
    if "/" in destination:
        from huggingface_hub import HfApi

        api = HfApi()
        staging = Path.home() / f".hf-stage-{utc_now_compact()}"
        staging.mkdir(parents=True, exist_ok=True)
        leaf = destination.rsplit("/", 1)[-1].rsplit(":", 1)[0]
        save_json(export_content, staging / f"{leaf}_metrics.json")
        try:
            api.upload_folder(
                repo_id=destination,
                folder_path=str(staging),
                commit_message=commit_message or f"Auto-committed metrics {utc_now_compact()}",
            )
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        if not quiet:
            print(f"{RIGHT_ARROW} Metrics committed to https://huggingface.co/{destination}")
    else:
        resolved = Path(destination).with_suffix(".json")
        save_json(export_content, resolved)
        if not quiet:
            print(f"Metrics written to {resolved.resolve()}")


def push_local_folder_as_dataset(local_root: str, ds_id: str, private: bool = True) -> None:
    """Turn a local directory into a (typically private) Hub dataset."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=ds_id, repo_type="dataset", private=private, exist_ok=True)
    workers = max(1, min(shutil.os.cpu_count() or 1, 96))
    api.upload_large_folder(
        repo_id=ds_id,
        repo_type="dataset",
        folder_path=local_root,
        num_processes=workers,
    )


def push_local_folder_as_space(local_root: str, sp_id: str, sdk: str = "gradio") -> None:
    """Ship a local Gradio/Streamlit app up as a living Space."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=sp_id, repo_type="space", space_sdk=sdk, exist_ok=True)
    api.upload_folder(repo_id=sp_id, repo_type="space", folder_path=local_root)


_RESERVED_ARGNAMES = frozenset({"num_labels", "hidden_dropout_prob", "attention_probs_dropout_prob"})


def pull_model(
    model_id: str,
    cls: type | None = None,
    *,
    dtype: Any = None,
    device_map: Any = None,
    trust_remote_code: bool = True,
    flash_attention: bool | None = None,
    **constructor_overrides: Any,
) -> Any:
    """
    Sensible-default loader for a sequence-classification model from the Hub.

    Handles BF16 autocasting, CUDA placement, FlashAttention probing and lets
    callers inject constructor extras minus a couple of guarded internals.
    """
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    klass = cls or AutoModelForSequenceClassification
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    filtered_extras = {k: v for k, v in constructor_overrides.items() if k not in _RESERVED_ARGNAMES}

    use_flash = flash_attention if flash_attention is not None else torch.cuda.is_available()
    implementation = "flash_attention_2" if use_flash else None

    return klass.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype or torch.bfloat16,
        device_map=device_map or ("auto" if torch.cuda.is_available() else None),
        trust_remote_code=trust_remote_code,
        attn_implementation=implementation,
        low_cpu_mem_usage=True,
        **filtered_extras,
    )


def pull_dataset(name: str, streaming: bool = False, **loader_options: Any) -> Any:
    """Thin passthrough to ``datasets.load_dataset`` keeping signatures tidy."""
    from datasets import load_dataset

    return load_dataset(path=name, streaming=streaming, **loader_options)
