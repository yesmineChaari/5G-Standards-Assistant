# Usage: make <target>. `make install` once, then `make all` runs the whole pipeline:
# download -> ingest -> index -> test -> eval (retrieval, generation, tables).

ifeq ($(OS),Windows_NT)
  PY := .venv/Scripts/python
  BOOTSTRAP := py -3.11
else
  PY := .venv/bin/python
  BOOTSTRAP := python3.11
endif

.PHONY: install download ingest index app eval eval-retrieval eval-generation tables test all bonus-models

install:
	$(BOOTSTRAP) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

download:
	$(PY) -m src.ingest.download

ingest:
	$(PY) -m src.ingest.chunker
	$(PY) scripts/inspect_chunks.py
	$(PY) -m src.retrieve.acronyms build

index:
	$(PY) -m src.index.vector_store build
	$(PY) -m src.index.bm25_index build

app:
	$(PY) -m streamlit run src/app/streamlit_app.py --server.fileWatcherType none --browser.gatherUsageStats false

eval-retrieval:
	$(PY) -m eval.run_retrieval_eval

eval-generation:
	$(PY) -m eval.run_generation_eval

tables:
	$(PY) -m eval.make_tables

eval: eval-retrieval eval-generation tables

test:
	$(PY) -m pytest -q tests

all: download ingest index test eval

# Bonus: model comparison (replays recorded calls; new generators need Ollama, new embedding models build their own index)
bonus-models:
	$(PY) -m eval.run_model_comparison llm
	$(PY) -m eval.run_model_comparison embeddings
