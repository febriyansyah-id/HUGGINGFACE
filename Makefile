PYTHON ?= python
VENV_DIR := .venv

.PHONY: init env install login upload download train eval clean

## Initialize everything
init: env install

## Create virtual environment
env:
	$(PYTHON) -m venv $(VENV_DIR)

## Activate venv helper note
activate:
	@echo "Source the venv: source $(VENV_DIR)/bin/activate"

## Install dependencies
install:
	./$(VENV_DIR)/bin/pip install --upgrade pip
	./$(VENV_DIR)/bin/pip install -r requirements.txt
	./$(VENV_DIR)/bin/pip install -e .

## Login to Hugging Face hub
login:
	huggingface-cli login

## Push a local dir/model to HF Hub
push-model:
	HF_HUB_ENABLE_HF_TRANSFER=1 ./$(VENV_DIR)/bin/python scripts/push_to_hub.py --kind model

push-dataset:
	HF_HUB_ENABLE_HF_TRANSFER=1 ./$(VENV_DIR)/bin/python scripts/push_to_hub.py --kind dataset

pull-model:
	./$(VENV_DIR)/bin/python scripts/download_from_hub.py --kind model

pull-dataset:
	./$(VENV_DIR)/bin/python scripts/download_from_hub.py --kind dataset

train:
	./$(VENV_DIR)/bin/python scripts/train_example.py

eval:
	./$(VENV_DIR)/bin/python scripts/eval_example.py

clean:
	rm -rf __pycache__ *.egg-info dist build
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
