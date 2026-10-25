"""Builds the models. Real mode uses NVIDIA NIM (+ optional Perplexity);
mock mode uses small offline stand-ins so the pipeline can run without API keys."""

import hashlib
import logging
import math
import re
from typing import Any, Sequence

from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import (
    CompletionResponse,
    CompletionResponseGen,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.schema import NodeWithScore

from config import config

logger = logging.getLogger(__name__)
_WORD = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------
# Mock components
# --------------------------------------------------------------------------
class HashEmbedding(BaseEmbedding):
    """Bag-of-words hashing embedding. Crude, but keyword overlap makes retrieval meaningful."""

    dim: int = 512

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 20) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


class EchoLLM(CustomLLM):
    """Pretends to be an LLM: rewrites queries by stripping format words and
    'answers' by quoting the best-matching source lines with citations."""

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(model_name="mock-echo-llm", context_window=128_000, num_output=512)

    def _answer(self, prompt: str) -> str:
        if prompt.rstrip().endswith("Rewritten:"):
            raw = prompt.rsplit("Raw text:", 1)[1].rsplit("Rewritten:", 1)[0]
            drop = {"write", "make", "table", "email", "summary", "summarize", "please", "a", "an", "the", "me", "it", "that"}
            words = [w for w in raw.split() if w.lower().strip(".,?") not in drop]
            earlier = prompt.split("Earlier question(s):", 1)[1].split("Raw text:", 1)[0]
            earlier = earlier.replace("user:", "").replace("(none)", "").strip()
            if earlier and re.search(r"\b(it|that|its|this|they)\b", raw, re.I):
                words += [w for w in earlier.split() if len(w) > 3 and w.lower() not in drop]  # crude reference resolution
            return " ".join(words).strip()

        query = prompt.rsplit("User:", 1)[1].rsplit("Assistant:", 1)[0].strip()
        sources = re.findall(r"\[(\d+)\] \(([^)]*)\)\n(.*?)(?=\n\[\d+\] \(|\nConversation so far:)", prompt, re.S)
        if not sources:
            return f"(mock mode, no retrieval) You said: {query!r}. Add NVIDIA_API_KEY to .env for real answers."
        q = set(_WORD.findall(query.lower()))
        lines = []
        for num, _label, text in sources:
            for line in re.split(r"(?<=[.!?])\s+|\n", text):
                score = len(q & set(_WORD.findall(line.lower())))
                if score and len(line) > 20:
                    lines.append((score, line.strip()[:220], num))
        lines.sort(key=lambda t: -t[0])
        best = lines[:3] or [(0, sources[0][2].strip()[:220], sources[0][0])]
        body = "\n".join(f"- {text} [{num}]" for _, text, num in best)
        return f"(mock mode: quoting the most relevant source lines)\n{body}"

    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        return CompletionResponse(text=self._answer(prompt))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        text, so_far = self._answer(prompt), ""
        for tok in re.findall(r"\S+\s*|\s+", text):
            so_far += tok
            yield CompletionResponse(text=so_far, delta=tok)


SMALL_TALK = re.compile(
    r"^(hi|hello|hey|thanks|thank you|what are you|who are you|count to|write a (haiku|poem)|"
    r"(fix|check|edit|proofread|rewrite).*(email|typo|grammar)|translate|debug|help me debug)",
    re.I,
)


def mock_route(query: str) -> bool:
    """Keyword stand-in for the LLM router. True = use retrieval."""
    return not SMALL_TALK.search(query.strip())


class KeywordRerank:
    """Stand-in for the NVIDIA reranker: sorts by query-word overlap."""

    def __init__(self, top_n: int):
        self.top_n = top_n

    def postprocess_nodes(self, nodes: Sequence[NodeWithScore], query_str: str) -> list[NodeWithScore]:
        q = set(_WORD.findall(query_str.lower()))
        scored = []
        for n in nodes:
            overlap = len(q & set(_WORD.findall(n.node.get_content().lower())))
            scored.append(NodeWithScore(node=n.node, score=float(overlap)))
        scored.sort(key=lambda n: -n.score)
        return scored[: self.top_n]


# --------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------
def _nim_kwargs() -> dict:
    kw = {"api_key": config.nvidia_api_key}
    if config.nim_base_url:
        kw["base_url"] = config.nim_base_url
    return kw


def build_embed_model() -> BaseEmbedding:
    if config.mock:
        return HashEmbedding()
    from llama_index.embeddings.nvidia import NVIDIAEmbedding

    return NVIDIAEmbedding(model=config.embedding_model, truncate="END", **_nim_kwargs())


def build_llms():
    """Returns (routing_llm, chat_llm)."""
    if config.mock:
        llm = EchoLLM()
        return llm, llm
    # NIM speaks the OpenAI API. OpenAILike with is_chat_model=True always hits
    # /chat/completions, even for new models the NVIDIA connector doesn't know yet.
    from llama_index.llms.openai_like import OpenAILike

    def make(model, temperature, max_tokens):
        return OpenAILike(
            model=model,
            api_base=config.nim_base_url or "https://integrate.api.nvidia.com/v1",
            api_key=config.nvidia_api_key,
            is_chat_model=True,
            context_window=config.context_window,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=config.llm_timeout,
        )

    return make(config.routing_model, 0.0, 64), make(config.chat_model, 0.2, config.max_tokens_generated)


def build_reranker():
    if not config.use_reranker:
        return None
    if config.mock:
        return KeywordRerank(config.rerank_top_n)
    from llama_index.postprocessor.nvidia_rerank import NVIDIARerank

    return NVIDIARerank(model=config.rerank_model, top_n=config.rerank_top_n, truncate="END", **_nim_kwargs())


def build_perplexity():
    if not config.perplexity_api_key:
        return None
    from llama_index.llms.perplexity import Perplexity

    return Perplexity(model=config.perplexity_model, api_key=config.perplexity_api_key, temperature=0.2)
