# 🚀 Hugging Face Workspace

Sandbox for experimenting with the **[🤗 Hugging Face]** ecosystem — loading,
fine‑tuning, evaluating and publishing models, datasets and Spaces — wired up
so you spend your energy on ML, not plumbing.

[Hugging Face]: https://huggingface.co

---

## ✨ Features

- 🔐 Drop‑dead‑easy **authentication** helpers (`whoami`, token lookup).
- 📦 Scripts to **push** local experiments ↔ **pull** upstream assets.
- ⏱️ Minimal **Trainer** recipe (IMDb sentiment) demonstrating the happy path.
- 🌍 Ready‑made **Gradio** starter for turning a model into a Shareable link.
- 💬 Tiny **LLM chat** harness for interrogating instruct models instantly.
- ✅ Offline structural tests guarding the scaffold.

## 📁 Layout

```
├── configs/           # reusable knob templates (json / yaml)
├── data/              # local corpora (gitignored apart from placeholders)
├── examples/          # runnable showcase scripts
│   ├── chat_with_model.py
│   └── minigradio_app.py
├── models/            # intermediate checkpoints (usually gitignored)
├── notebooks/         # throwaway explorations
├── results/           # metrics / plots emitted by jobs
├── scripts/           # operational entrypoints
│   ├── train_example.py
│   ├── eval_example.py
│   ├── push_to_hub.py
│   └── download_from_hub.py
└── src/hfhub/         # reusable Python package backing the scripts
```

## 🧰 Quick Start

```bash
# 1️⃣ Bootstrap a dedicated virtualenv + install deps
make init                 # shorthand for: make env && make install

# 2️⃣ Authenticate with the Hub (paste a WRITE-capable token)
make login                # alias: huggingface-cli login

# 3️⃣ Sanity-check connectivity
python - <<'EOF'
from hfhub import logged_in_user
print(logged_in_user())
EOF
```

## ▶️ Common Workflows

### Chat with an instruct model

```bash
python examples/chat_with_model.py --model mistralai/Ministral-8B-Instruct-2410
```

### Launch a shareable summariser UI

```bash
python examples/minigradio_app.py --model sshleifer/distilbart-cnn-12-6 --share-link
```

### Fine‑tune the bundled demo

```bash
python scripts/train_example.py --epochs 2 --subset-count 1600
```

### Evaluate a local / Hosted checkpoint

```bash
python scripts/eval_example.py                              # local export
python scripts/eval_example.py --model cardiffnlp/tweet-topic-21-multi
```

### Publish artefacts to the Hub

```bash
# Repositories are PRIVATE by default; opt-in with --public
python scripts/push_to_hub.py --kind model   --input-path ./exports/finetuned_distilbert_imdb --repo YOU/imdb-demo
python scripts/push_to_hub.py --kind dataset --input-path ./some/local/folder                   --repo YOU/my-data --public
```

Prefer the packaged CLI?

```bash
python -m hfhub.cli whoami
python -m hfhub.cli push-data ./corpus YOU/public-corpus --public
python -m hfhub.cli deploy  ./apps/summer-ui YOU/summ-space
```

### Grab stuff FROM the Hub

```bash
python scripts/download_from_hub.py --kind model   --repo google-bert/bert-base-uncased
python scripts/download_from_hub.py --kind dataset --repo ag_news
```

## 🔑 Authentication Details

Wherever `huggingface_hub` hunts for creds, ours plug in identically:

- **Interactive:** `make login` persists the token to `~/.cache/huggingface/token`.
- **Headless / CI:** populate a `.env` (copy `.env.example`) with
  `HF_TOKEN=<your_write_token>`, or export it in your shell.

> Mint tokens at <https://huggingface.co/settings/tokens>, grant the narrowest
> role that suffices (**WRITE** covers pushes), and rotate periodically.

## 🛠 Developer Loop

```bash
make activate                         # reminder to join the venv
ruff check src scripts examples       # static hygiene (needs ruff)
python -m pytest tests -q             # green-light the scaffold
```

Happy hacking! 🎉
