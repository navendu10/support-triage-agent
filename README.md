# support-triage-agent

A portfolio project that triages customer messages for a fictional online
store, built with **LangChain**, **LangGraph** and **LangSmith** in Python.

## Goal
Given a customer message, the agent works out what it is about (category and
intent, e.g. `REFUND / get_refund`), looks up relevant knowledge-base articles,
and drafts a reply or routes it to a human. LangSmith is used to trace each run
and to evaluate triage accuracy.

## Data
This project uses the **Bitext Customer Support LLM Chatbot Training Dataset**
by [Bitext](https://www.bitext.com/), available on Hugging Face at
[bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset).
It is licensed under the
[Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)](https://cdla.dev/sharing-1-0/).
Thanks to Bitext for making it publicly available.

The dataset has 26,872 rows with columns `flags`, `instruction`, `category`,
`intent` and `response`, across 11 categories (ACCOUNT, CANCEL, CONTACT,
DELIVERY, FEEDBACK, INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION).
It is downloaded to `data/raw/`, which is not committed.

## Models
| Role | Model | Why |
|---|---|---|
| Chat / reasoning | Gemini `gemini-3.5-flash-lite` (free tier) | Free, fast, and the most generous free quota of Gemini's Flash models. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local) | Small (~90 MB), fast on CPU, widely used; runs locally so there are no costs or rate limits. |

All models are created in [src/triage/config.py](src/triage/config.py). The
chat provider can be switched (e.g. to Groq, Ollama or Hugging Face Inference
Providers) by changing `CHAT_MODEL` in `.env`. Calls are rate-limited to stay
under free-tier limits and retried with backoff.

## Setup (Windows, Python 3.11+)
```powershell
git clone https://github.com/navendu10/support-triage-agent.git
cd support-triage-agent
py -3.12 -m venv .venv

# CPU-only PyTorch first (much smaller than the default CUDA build)
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip install -e ".[dev]"

copy .env.example .env      # then fill in your keys
.venv\Scripts\python scripts/check_setup.py
.venv\Scripts\python -m pytest
```
On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python`.

Keys needed: `GOOGLE_API_KEY` ([AI Studio](https://aistudio.google.com/apikey)),
`LANGSMITH_API_KEY` ([LangSmith](https://smith.langchain.com)), and optionally
`HF_TOKEN` (the dataset is public, but a token avoids anonymous rate limits).

## Layout
```
src/triage/   application code (config.py creates all models)
scripts/      one-off scripts (check_setup.py)
data/raw/     downloaded dataset (gitignored)
data/kb/      knowledge-base documents (committed)
evals/        LangSmith evaluations
tests/        pytest tests (model calls are mocked)
docs/         notes and design docs
```

## Licence
Code: MIT (see [LICENSE](LICENSE)). Data: CDLA-Sharing-1.0, © Bitext.
