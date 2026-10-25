"""The RAG logic as a LlamaIndex Workflow (event-driven steps).

    StartEvent
      └─ route ──────────────► ShortcutEvent ───────────────────────────┐   (no retrieval needed)
          └─ RawQueryEvent                                               │
               └─ rewrite_query ─► TransformedQueryEvent                 │
                     ├─ embed_query ─► EmbeddedQueryEvent                │
                     │     └─ milvus_retrieve ─► MilvusQueryEvent ─┐     │
                     └─ pplx_retrieve ─────────► PerplexityQueryEvent ┤  │   (runs in parallel)
                                     collect_nodes (+ rerank) ◄─────┘    │
                                           └─ NodeCollectEvent ─► synthesize_response ─► StopEvent

Based on NVIDIA's routing-multisource-rag example, updated for current LlamaIndex,
with a reranker, numbered citations, and timeouts added.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from time import time

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step

import prompts
from config import config
from providers import build_embed_model, build_llms, build_perplexity, build_reranker, mock_route

logger = logging.getLogger(__name__)
logging.getLogger("grpc._server").setLevel(logging.CRITICAL)  # harmless Milvus Lite noise

routing_llm, chat_llm = build_llms()
embed_model = build_embed_model()
reranker = build_reranker()
perplexity_llm = build_perplexity()
_vector_store = None


def get_vector_store():
    """Opened lazily so the app still starts (and answers non-RAG questions) before ingest.py has run."""
    global _vector_store
    if _vector_store is None:
        if not os.path.exists(config.milvus_path):
            raise FileNotFoundError("No vector index yet. Run `python ingest.py` first.")
        if os.path.exists(config.index_info_path):
            info = json.load(open(config.index_info_path))
            if info.get("embedding_model") != config.active_embedding_model:
                raise RuntimeError(
                    f"Index was built with {info.get('embedding_model')!r} but the app is using "
                    f"{config.active_embedding_model!r}. Re-run `python ingest.py`."
                )
        from llama_index.vector_stores.milvus import MilvusVectorStore

        _vector_store = MilvusVectorStore(
            uri=config.milvus_path, collection_name=config.collection_name, use_async_client=False
        )
        _vector_store.client.load_collection(config.collection_name)  # Milvus Lite needs an explicit load
    return _vector_store


def approx_tokens(text: str) -> int:
    return len(text) // 4 + 1


# --------------------------------------------------------------------------
# Tracing: a Chainlit Step in the UI, a log line in the terminal
# --------------------------------------------------------------------------
class _LogStep:
    def __init__(self, name):
        self.name, self.input, self.output = name, None, None


@asynccontextmanager
async def trace(name: str):
    try:
        import chainlit as cl
        from chainlit.context import context

        context.session  # raises outside a Chainlit session
    except Exception:
        s, t0 = _LogStep(name), time()
        yield s
        logger.info("[%s] %.2fs output=%s", name, time() - t0, str(s.output)[:200])
        return
    async with cl.Step(name=name, type="tool") as s:
        yield s


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
class ShortcutEvent(Event):
    pass


class RawQueryEvent(Event):
    raw_query: str


class TransformedQueryEvent(Event):
    transformed_query: str


class EmbeddedQueryEvent(Event):
    embedding: list[float]


class MilvusQueryEvent(Event):
    nodes: list[NodeWithScore]


class PerplexityQueryEvent(Event):
    nodes: list[NodeWithScore]


class NodeCollectEvent(Event):
    nodes: list[NodeWithScore]


def history_string(messages: list[dict], n: int) -> str:
    return "".join(f"{m['role']}: {m['content']}\n" for m in messages[-n:]) if n > 0 else ""


def source_label(node) -> str:
    md = node.metadata
    if md.get("source") == "perplexity":
        return "Web search (Perplexity)"
    label = md.get("file_name", "document")
    if md.get("page_label"):
        label += f", p. {md['page_label']}"
    return label


# --------------------------------------------------------------------------
# Workflow
# --------------------------------------------------------------------------
class QueryFlow(Workflow):
    @step
    async def route(self, ctx: Context, ev: StartEvent) -> ShortcutEvent | RawQueryEvent:
        await ctx.store.set("raw_query", ev.query)
        await ctx.store.set("chat_messages", ev.get("chat_messages", []))
        await ctx.store.set("sources", [])

        async with trace("query router") as s:
            t0 = time()
            s.input = ev.query
            if config.mock:
                use_search = mock_route(ev.query)
            else:
                try:
                    prompt = prompts.ROUTING_PROMPT.format(query=ev.query) + "\nAnswer with exactly one word, true or false.\nuse_search:"
                    out = str(await routing_llm.acomplete(prompt)).strip().lower()
                    use_search = "false" not in out[:12]
                except Exception as e:  # fail open: retrieval is the safe default
                    logger.warning("Router failed (%s); defaulting to retrieval.", e)
                    use_search = True
            s.output = {"use_search": use_search, "seconds": round(time() - t0, 3)}

        await ctx.store.set("route", "rag" if use_search else "direct")
        return RawQueryEvent(raw_query=ev.query) if use_search else ShortcutEvent()

    @step
    async def rewrite_query(self, ctx: Context, ev: RawQueryEvent) -> TransformedQueryEvent:
        async with trace("query rewrite") as s:
            t0 = time()
            msgs = await ctx.store.get("chat_messages")
            prev_user = [m for m in msgs if m["role"] == "user"][-1:]
            prompt = prompts.QUERY_REWRITE_PROMPT.format(
                query=ev.raw_query, history_str=history_string(prev_user, 1) or "(none)"
            )
            try:
                rewritten = str(await chat_llm.acomplete(prompt)).strip().strip('"') or ev.raw_query
            except Exception as e:
                logger.warning("Rewrite failed (%s); using raw query.", e)
                rewritten = ev.raw_query
            s.input = ev.raw_query
            s.output = {"transformed_query": rewritten, "seconds": round(time() - t0, 3)}
        await ctx.store.set("transformed_query", rewritten)
        return TransformedQueryEvent(transformed_query=rewritten)

    @step
    async def embed_query(self, ev: TransformedQueryEvent) -> EmbeddedQueryEvent:
        async with trace("vector embedding") as s:
            t0 = time()
            emb = await embed_model.aget_query_embedding(ev.transformed_query)
            s.output = {"dim": len(emb), "preview": [round(x, 4) for x in emb[:5]], "seconds": round(time() - t0, 3)}
        return EmbeddedQueryEvent(embedding=emb)

    @step
    async def milvus_retrieve(self, ev: EmbeddedQueryEvent) -> MilvusQueryEvent:
        async with trace("Milvus search") as s:
            t0 = time()
            try:
                store = get_vector_store()
                res = await asyncio.to_thread(
                    store.query, VectorStoreQuery(query_embedding=ev.embedding, similarity_top_k=config.similarity_top_k)
                )
                sims = res.similarities or [None] * len(res.nodes)
                nodes = [NodeWithScore(node=n, score=sc) for n, sc in zip(res.nodes, sims)]
            except Exception as e:
                logger.error("Milvus retrieval failed: %s", e)
                nodes = []
            s.output = {
                "hits": [f"{source_label(n.node)} ({(n.score or 0):.3f})" for n in nodes],
                "seconds": round(time() - t0, 3),
            }
        return MilvusQueryEvent(nodes=nodes)

    @step
    async def pplx_retrieve(self, ev: TransformedQueryEvent) -> PerplexityQueryEvent:
        async with trace("Perplexity web search") as s:
            t0 = time()
            nodes: list[NodeWithScore] = []
            if perplexity_llm is None:
                s.output = "skipped (no PERPLEXITY_API_KEY)"
                return PerplexityQueryEvent(nodes=[])
            try:
                resp = await asyncio.wait_for(
                    perplexity_llm.achat(
                        [
                            ChatMessage(role="system", content="Be precise and concise."),
                            ChatMessage(role="user", content=ev.transformed_query),
                        ]
                    ),
                    timeout=config.perplexity_timeout,
                )
                raw = resp.raw if isinstance(resp.raw, dict) else {}
                urls = raw.get("citations") or [r.get("url") for r in raw.get("search_results", []) if r.get("url")]
                text = resp.message.content
                if urls:
                    text += "\nWeb sources: " + ", ".join(urls[:5])
                nodes = [NodeWithScore(node=TextNode(text=text, metadata={"source": "perplexity", "urls": urls[:5]}), score=1.0)]
                s.output = {"answer": text[:500], "seconds": round(time() - t0, 3)}
            except asyncio.TimeoutError:
                s.output = f"timed out after {config.perplexity_timeout}s, continuing without web results"
            except Exception as e:
                s.output = f"error: {e}"
        return PerplexityQueryEvent(nodes=nodes)

    @step
    async def collect_nodes(self, ctx: Context, ev: MilvusQueryEvent | PerplexityQueryEvent) -> NodeCollectEvent | None:
        ready = ctx.collect_events(ev, [MilvusQueryEvent, PerplexityQueryEvent])
        if ready is None:
            return None  # wait for the other retriever
        milvus_nodes = next(e.nodes for e in ready if isinstance(e, MilvusQueryEvent))
        web_nodes = next(e.nodes for e in ready if isinstance(e, PerplexityQueryEvent))

        if reranker and milvus_nodes:
            async with trace("rerank") as s:
                t0 = time()
                query = await ctx.store.get("transformed_query")
                try:
                    milvus_nodes = await asyncio.to_thread(reranker.postprocess_nodes, milvus_nodes, query_str=query)
                except Exception as e:
                    logger.warning("Rerank failed (%s); keeping vector order.", e)
                    milvus_nodes = milvus_nodes[: config.rerank_top_n]
                s.output = {"kept": [source_label(n.node) for n in milvus_nodes], "seconds": round(time() - t0, 3)}
        else:
            milvus_nodes = milvus_nodes[: config.rerank_top_n]
        return NodeCollectEvent(nodes=milvus_nodes + web_nodes)

    @step
    async def synthesize_response(self, ctx: Context, ev: NodeCollectEvent | ShortcutEvent) -> StopEvent:
        query = await ctx.store.get("raw_query")
        hist = history_string(await ctx.store.get("chat_messages"), config.n_messages_in_history)
        nodes = getattr(ev, "nodes", [])

        if isinstance(ev, ShortcutEvent):
            prompt = prompts.NO_RAG_PROMPT_TEMPLATE.format(history_str=hist, query=query)
            sources = []
        else:
            # Pack numbered sources until the context budget runs out.
            budget = config.context_window - config.max_tokens_generated - approx_tokens(hist + query) - 600
            blocks, sources = [], []
            for n in nodes:
                block = f"[{len(sources) + 1}] ({source_label(n.node)})\n{n.node.get_content().strip()}\n"
                if approx_tokens(block) > budget:
                    break
                budget -= approx_tokens(block)
                blocks.append(block)
                sources.append({"n": len(sources) + 1, "label": source_label(n.node), "urls": n.node.metadata.get("urls", [])})
            context_str = "\n".join(blocks) or "(no relevant sources found)"
            prompt = prompts.SYSTEM_PROMPT_TEMPLATE.format(context_str=context_str, history_str=hist, query=query)

        async with trace("response synthesis") as s:
            s.input = prompt
            s.output = {"prompt_tokens_approx": approx_tokens(prompt), "sources": len(sources)}
            stream = await chat_llm.astream_complete(prompt)

        return StopEvent(
            result={
                "stream": stream,
                "sources": sources,
                "route": await ctx.store.get("route"),
                "transformed_query": await ctx.store.get("transformed_query", default=None),
            }
        )


def format_sources(sources: list[dict], answer: str) -> str:
    """Reference list for the sources the answer actually cited (falls back to all)."""
    if not sources:
        return ""
    cited = [s for s in sources if f"[{s['n']}]" in answer] or sources
    lines = []
    for s in cited:
        line = f"[{s['n']}] {s['label']}"
        if s["urls"]:
            line += " — " + ", ".join(s["urls"][:3])
        lines.append(line)
    return "\n\n**Sources**\n" + "\n".join(f"- {l}" for l in lines)
