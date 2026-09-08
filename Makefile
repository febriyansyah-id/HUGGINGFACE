PYTHON ?= python
VENV_DIR := .venv

# Resolve the Python interpreter lazily:
#   • honour an overriding PYTHON,
#   • else use a local .venv if one exists,
#   • else fall back to the neighbouring collab-workspace venv (known-good).
ifeq ($(wildcard $(VENV_DIR)/bin/python),)
VPY := ../collab-workspace/.venv/bin/python
else
VPY := ./$(VENV_DIR)/bin/python
endif

.PHONY: init env install login logout whoami push-model push-dataset pull-model pull-dataset train eval clean

# Source .env (when present) so network-dependent targets inherit HF_TOKEN et al.
ifneq ($(wildcard .env),)
-include .env
export
endif

## Show which interpreter the targets will use
doctor:
	@echo "Interpreter: $(VPY)"
	@"$(VPY)" -c "import sys,huggingface_hub;print('huggingface_hub',huggingface_hub.__version__,'·','python',sys.version.split()[0])"

## Who owns the current Hub session?
whoami:
	@"$(VPY)" -c "from huggingface_hub import whoami; u=whoami(); print('Signed in as', u['name'], '\u2192 https://huggingface.co/'+u['name'])"

## Forget the stored Hub credentials
logout:
	@if [ -x "$$(dirname "$(VPY)")/hf" ]; then "$$(dirname "$(VPY)")/hf" auth logout; else echo "Removing stored token…"; rm -f ~/.cache/huggingface/stored_tokens; fi

## Initialize everything
init: doctor install

## Create virtual environment (requires python3.X-venv/system pip)
env:
	$(PYTHON) -m venv $(VENV_DIR)

## Activate venv helper note
activate:
	@echo "Source the venv: source $(VENV_DIR)/bin/activate"

## Install dependencies into the resolved interpreter
install:
	"$(VPY)" -m pip install --upgrade pip
	"$(VPY)" -m pip install -r requirements.txt
	"$(VPY)" -m pip install -e .

## Login to Hugging Face hub (via the modern `hf` CLI when available)
define LOGIN_SH =
	if [ -x "$$(dirname "$(VPY)")/hf" ]; then \
	  echo "[+] Found modern 'hf' CLI — launching interactive login…"; \
	  "$$(dirname "$(VPY)")/hf" auth login; \
	else \
	  echo "[!] Neither 'hf' nor 'huggingface-cli' is usable."; \
	  echo "    Paste a WRITE token into .env as HF_TOKEN=..., or run:"; \
	  echo "    curl -LsSf https://hf.co/cli/install.sh | bash"; \
	  exit 1; \
	fi
endef
login:
	$(LOGIN_SH)

## Push a local dir/model to HF Hub
push-model:
	HF_HUB_ENABLE_HF_TRANSFER=1 "$(VPY)" scripts/push_to_hub.py --kind model

push-dataset:
	HF_HUB_ENABLE_HF_TRANSFER=1 "$(VPY)" scripts/push_to_hub.py --kind dataset

pull-model:
	"$(VPY)" scripts/download_from_hub.py --kind model

pull-dataset:
	"$(VPY)" scripts/download_from_hub.py --kind dataset

train:
	"$(VPY)" scripts/train_example.py

eval:
	"$(VPY)" scripts/eval_example.py

clean:
	rm -rf __pycache__ *.egg-info dist build
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
