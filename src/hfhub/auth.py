"""Small conveniences layered atop ``huggingface_hub`` authentication."""
from __future__ import annotations

import os

ENDORSE_URL = "https://huggingface.co/settings/tokens"


def get_access_token(warn: bool = False) -> str | None:
    """Return the persisted Hub token (may legitimately be ``None``)."""
    from huggingface_hub import get_token

    token = get_token()
    if warn and token is None:
        print(f"No access token detected — obtain one at {ENDORSE_URL}")
    elif warn and token:
        print(f"Token present (masked): {token[:6]}•••{token[-4:]}")
    return token


def logged_in_user(print_profile: bool = True) -> str:
    """
    Confirm the machine holds valid Hub credentials and reveal whose they are.

    Raises ``OSError`` when nobody is signed in.
    """
    from huggingface_hub import whoami

    identity = whoami()
    username = identity["name"]
    if print_profile:
        print(f"You are logged in as {username} → https://huggingface.co/{username}")
    return username


def get_default_namespace() -> str:
    """Derive the owning org/user for newly created repos."""
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def ensure_valid_token(preflight_print: bool = True) -> str:
    """
    Guarantee a usable token exists, raising a friendly error otherwise.

    Intended for guard clauses at the top of mutating scripts.
    """
    token = get_access_token(warn=preflight_print)
    if not token:
        hint = (
            f"Authenticate first:\n"
            f"  1) Visit {ENDORSE_URL}\n"
            f"  2) Either run `huggingface-cli login`,\n"
            f"     or export HF_TOKEN=<that token>\n"
        )
        raise PermissionError(hint)
    return token
