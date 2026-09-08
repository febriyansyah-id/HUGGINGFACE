#!/usr/bin/env python3
"""Simpan HF token aman ke ~/.cache/huggingface/token (tidak ke repo)."""
import getpass
from huggingface_hub import HfFolder, whoami

token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
if not token.startswith("hf_"):
    raise SystemExit("Token harus diawali hf_")
HfFolder.save_token(token)
print("Saved to ~/.cache/huggingface/token")
print(whoami())
