# 5G RAG Chatbot: 3GPP TS 23.501 & TS 24.501 (Release 18)

Retrieval-Augmented Generation chatbot that answers questions in English about 5G architecture,
procedures, NAS messages and mechanisms, grounded exclusively in:

- **TS 23.501 V18.12.0**: System architecture for the 5G System (Stage 2)
- **TS 24.501 V18.14.0**: NAS protocol for 5GS (Stage 3)

Every answer cites spec and section, e.g. `(TS 24.501, §5.5.1.2.2)`. When the information is not in
the corpus, the answer is exactly: `I could not find this information in the documents.`

> Mini-projet de contrôle continu, Futurs Réseaux Mobiles (RT5, INSAT 2026/2027).

## Status

Work in progress. See `PROGRESS.md`.

## Quick start

Requirements: Python 3.11.

```bash
make install          # or: py -3.11 -m venv .venv && .venv\Scripts\python -m pip install -r requirements.txt
cp .env.example .env  # then fill in GROQ_API_KEY
```

Full installation, execution, architecture and results sections will be completed in the final step.
