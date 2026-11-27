"""Model factory: the ONLY place in this project where models are created.

Keeping model creation in one file means switching provider or model is a
one-line .env change, and rate limiting is applied everywhere automatically.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_huggingface import HuggingFaceEmbeddings

# Read .env once when this module is imported, so every caller sees the keys.
load_dotenv()

DEFAULT_CHAT_MODEL = "google_genai:gemini-3.5-flash-lite"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# One shared limiter for the whole process. The Gemini free tier allows only
# a handful of requests per minute, so we stay well below it:
# 0.1 requests/second = at most 6 requests per minute.
# Note: this limits one Python process only, not several running at once.
_rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,
    check_every_n_seconds=0.5,  # how often a waiting call checks for a free slot
    max_bucket_size=1,          # no bursts: calls are spaced out evenly
)


@lru_cache(maxsize=1)
def get_chat_model():
    """Return the chat model named by CHAT_MODEL in .env.

    CHAT_MODEL uses LangChain's "provider:model" format, so init_chat_model
    knows which integration to load. To switch provider, install its package
    and change one line in .env, for example:

      Hugging Face Inference Providers (pip install langchain-huggingface, set HF_TOKEN):
          CHAT_MODEL=huggingface:meta-llama/Llama-3.1-8B-Instruct
      Groq (pip install langchain-groq, set GROQ_API_KEY):
          CHAT_MODEL=groq:llama-3.1-8b-instant
      Ollama, fully local (pip install langchain-ollama, run `ollama pull llama3.2`):
          CHAT_MODEL=ollama:llama3.2

    The rate limiter and retries below work the same for every provider.
    """
    model_name = os.getenv("CHAT_MODEL") or DEFAULT_CHAT_MODEL
    return init_chat_model(
        model_name,
        temperature=0,              # deterministic answers suit classification
        rate_limiter=_rate_limiter, # wait before sending if we're going too fast
        max_retries=3,              # retry transient errors (e.g. 429) with exponential backoff
        timeout=60,                 # seconds; don't hang forever on a stuck request
    )


@lru_cache(maxsize=1)
def get_embeddings():
    """Return a local sentence-transformers embedding model.

    Embeddings run on your own CPU, so they are free and have no rate limit.
    The model is downloaded from Hugging Face on first use, then cached.
    """
    model_name = os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        # Unit-length vectors make cosine similarity equal to a dot product.
        encode_kwargs={"normalize_embeddings": True},
    )
