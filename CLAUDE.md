# CLAUDE.md

## Project
A LangChain + LangGraph + LangSmith support-triage agent for a fictional online
store. It triages customer messages using the Bitext customer support dataset
(`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on Hugging Face,
licence CDLA-Sharing-1.0; it must be credited in the README).

The owner is a learner: keep code simple and add short comments explaining *why*.

## Models
- Create models ONLY through `src/triage/config.py` (`get_chat_model()`, `get_embeddings()`).
  Never instantiate a chat model or embedding model anywhere else.
- Chat: Gemini free tier via `init_chat_model`. Embeddings: local sentence-transformers.

## Gemini free-tier rate limits are tight
- Keep model calls to a minimum; never call a model in a loop without need.
- Never run evaluations with high concurrency (use `max_concurrency=1`).
- Always test with small samples first (e.g. 5-10 rows) before larger runs.

## APIs
- Use current LangChain 1.x and LangGraph 1.x APIs. Never use legacy chains
  (`LLMChain`, `ConversationChain`, `RetrievalQA`, etc.).
- Chroma via `langchain-chroma`; embeddings via `langchain-huggingface`.

## Workflow
- One git branch and one pull request per phase (e.g. `phase-0-setup`).
- Run `pytest` before every commit. Tests must mock model calls; they never hit an API.
- Never commit secrets (`.env`) or `data/raw/`.

## Commands (Windows)
- Tests: `.venv\Scripts\python -m pytest`
- Setup check: `.venv\Scripts\python scripts/check_setup.py`
