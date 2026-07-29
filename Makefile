PYTHON=python3
VENV=venv
PIP=$(VENV)/bin/pip
PY=$(VENV)/bin/python

# ---- setup ----
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -e ".[train]"

# ---- dev setup ----
setup-dev:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install torch --index-url https://download.pytorch.org/whl/cu121
	$(PIP) install -e ".[train,dev]"

# ---- run inference ----
run:
	$(PY) -m vjeanette.run --in_file ./data/test.fastq

# ---- train ----
train:
	$(PY) -m vjeanette.train
test:
	$(VENV)/bin/pytest
# ---- clean ----
clean:
	rm -rf $(VENV)
	find . -type d -name "__pycache__" -exec rm -r {} +

# ---- reinstall ----
reinstall: clean setup