"""Settings for the routing + multi-source RAG chat app.

Every value can be overridden with an environment variable (or a line in .env).
If NVIDIA_API_KEY is missing, the app runs in MOCK mode: offline stand-ins for the
LLMs, embeddings and reranker, so you can test the whole pipeline without keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")


def _env(name: str, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return type(default)(value) if not isinstance(default, bool) else value.lower() in ("1", "true", "yes")


class Config(BaseModel):
    # --- API keys ---------------------------------------------------------
    nvidia_api_key: str | None = os.getenv("NVIDIA_API_KEY") or None
    perplexity_api_key: str | None = os.getenv("PERPLEXITY_API_KEY") or None
    # Point this at a self-hosted NIM (e.g. http://localhost:8000/v1) to leave the hosted API.
    # As the blog says: that's the only code change needed.
    nim_base_url: str | None = os.getenv("NIM_BASE_URL") or None

    # --- Models (small router, large answerer) ----------------------------
    routing_model: str = _env("ROUTING_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    chat_model: str = _env("CHAT_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    embedding_model: str = _env("EMBEDDING_MODEL", "nvidia/nemotron-3.5-embed-1b")
    rerank_model: str = _env("RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
    perplexity_model: str = _env("PERPLEXITY_MODEL", "sonar")

    # --- Retrieval ---------------------------------------------------------
    similarity_top_k: int = _env("SIMILARITY_TOP_K", 10)  # pulled from Milvus
    rerank_top_n: int = _env("RERANK_TOP_N", 4)  # kept after reranking
    use_reranker: bool = _env("USE_RERANKER", False)
    chunk_size: int = _env("CHUNK_SIZE", 512)
    chunk_overlap: int = _env("CHUNK_OVERLAP", 64)

    # --- Generation / context ---------------------------------------------
    n_messages_in_history: int = _env("N_MESSAGES_IN_HISTORY", 6)
    max_tokens_generated: int = _env("MAX_TOKENS", 1024)
    context_window: int = _env("CONTEXT_WINDOW", 128_000)

    # --- Timeouts (seconds) -----------------------------------------------
    perplexity_timeout: float = _env("PERPLEXITY_TIMEOUT", 15.0)
    llm_timeout: float = _env("LLM_TIMEOUT", 60.0)
    workflow_timeout: float = _env("WORKFLOW_TIMEOUT", 120.0)

    # --- Paths -------------------------------------------------------------
    data_dir: str = str(ROOT / "data")
    milvus_path: str = str(ROOT / "db" / "milvus_lite.db")
    index_info_path: str = str(ROOT / "db" / "index_info.json")
    collection_name: str = "docs"

    @property
    def mock(self) -> bool:
        return not self.nvidia_api_key or _env("FORCE_MOCK", False)

    @property
    def active_embedding_model(self) -> str:
        return "mock-hash-embedding" if self.mock else self.embedding_model


config = Config()
