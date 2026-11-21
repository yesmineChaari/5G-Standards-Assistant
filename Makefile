# Usage: make <target>. Targets not implemented yet print the step that adds them.

ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python
  BOOTSTRAP := py -3.11
else
  PY := .venv/bin/python
  BOOTSTRAP := python3.11
endif

.PHONY: install download ingest index app eval eval-retrieval eval-generation test all

install:
	$(BOOTSTRAP) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.ingest.download

ingest:
	@echo Not implemented yet - steps 3-4.

index:
	@echo Not implemented yet - steps 5-6.

app:
	@echo Not implemented yet - step 11.

eval-retrieval:
	@echo Not implemented yet - step 13.

eval-generation:
	@echo Not implemented yet - step 14.

eval: eval-retrieval eval-generation

test:
	$(PY) -m pytest -q tests

all:
	@echo Not implemented yet - step 16.
