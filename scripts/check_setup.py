"""Check that keys, models and the dataset all work.

Run from the project root:  .venv\\Scripts\\python scripts/check_setup.py
It makes exactly ONE chat model call, to save free-tier quota.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Make the src/ package importable when running this file directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

load_dotenv(ROOT / ".env")

from triage.config import get_chat_model, get_embeddings  # noqa: E402

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
RAW_DIR = ROOT / "data" / "raw"


def check_keys() -> None:
    """Report which settings are present, never their values."""
    print("== Environment ==")
    for name in ["GOOGLE_API_KEY", "HF_TOKEN", "LANGSMITH_API_KEY",
                 "LANGSMITH_TRACING", "LANGSMITH_PROJECT", "CHAT_MODEL", "EMBEDDING_MODEL"]:
        status = "set" if os.getenv(name) else "MISSING"
        print(f"  {name:<18} {status}")


def check_chat_model() -> None:
    """One short call. With LANGSMITH_TRACING=true it shows up as a trace."""
    print("\n== Chat model ==")
    if not os.getenv("GOOGLE_API_KEY") and "google" in (os.getenv("CHAT_MODEL") or "google"):
        print("  skipped: GOOGLE_API_KEY is not set")
        return
    reply = get_chat_model().invoke("Reply with exactly one word: ready")
    # Newer models (e.g. Gemini 3.x) return content as a list of blocks
    # (text plus a hidden "thought signature"). `.text` joins just the text parts.
    print(f"  model replied: {reply.text!r}")
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true":
        print(f"  check the trace in LangSmith project '{os.getenv('LANGSMITH_PROJECT')}'")


def check_embeddings() -> None:
    print("\n== Embeddings ==")
    vector = get_embeddings().embed_query("Where is my order?")
    print(f"  vector length: {len(vector)}")


def check_dataset() -> None:
    """Download Bitext once and save it as CSV in data/raw/ (gitignored)."""
    print("\n== Dataset ==")
    from datasets import load_dataset  # imported here: it is slow to import

    ds = load_dataset(DATASET_ID, split="train")
    df = ds.to_pandas()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "bitext.csv"
    df.to_csv(out_path, index=False)

    print(f"  saved to: {out_path.relative_to(ROOT)}")
    print(f"  rows: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    print(f"  categories: {sorted(df['category'].unique())}")


if __name__ == "__main__":
    check_keys()
    # Each check runs independently, so one failure doesn't hide the others.
    for check in (check_chat_model, check_embeddings, check_dataset):
        try:
            check()
        except Exception as exc:  # show a friendly error and keep going
            print(f"  FAILED: {type(exc).__name__}: {exc}")
