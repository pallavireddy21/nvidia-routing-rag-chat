# Routing + Multi-Source RAG Chat

A working rebuild of the app from NVIDIA's blog post
[Creating RAG-Based Question-and-Answer LLM Workflows at NVIDIA](https://developer.nvidia.com/blog/creating-rag-based-question-and-answer-llm-workflows-at-nvidia/)
(Krapu & Rodge, Oct 2024). It's updated for current LlamaIndex and Chainlit, and adds the
"extra features" the post suggests: **reranking, citations, and timeouts**.

```
user query
  └─ router (small LLM): does this need retrieval?
       ├─ no  → answer directly (edit an email, write code, small talk)
       └─ yes → rewrite query (strip "table", "email", expand acronyms)
                  ├─ embed → Milvus Lite vector search ─┐   in parallel
                  └─ Perplexity web search (timeout) ───┤
                                    collect + rerank ◄──┘
                                      └─ answer (large LLM) with [1][2] citations
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # paste your NVIDIA_API_KEY (and optionally PERPLEXITY_API_KEY)
python ingest.py              # index everything in data/
python run_ui.py              # opens http://localhost:8000
```

Prefer the terminal? `python cli.py -v "What is the A100's memory bandwidth?"` (`-v` prints each step).

**No key yet?** Everything still runs in **mock mode**: real PDF parsing, chunking, Milvus search and
routing, with offline stand-ins for the models (answers are quoted source lines). Once you add
`NVIDIA_API_KEY`, re-run `python ingest.py` because the embeddings change.

Get a free NVIDIA key at [build.nvidia.com](https://build.nvidia.com). Perplexity is optional; without
it the web-search step is skipped.

## Files

| File | What it does |
|---|---|
| `workflow.py` | The LlamaIndex `Workflow`: route → rewrite → (Milvus ‖ Perplexity) → collect + rerank → synthesize |
| `ingest.py` | Loads `data/` (PDF, DOCX, PPTX, MD, TXT…), chunks it, and embeds it into Milvus Lite |
| `app.py` | Chainlit UI: each workflow step appears as an expandable trace, answers stream |
| `cli.py` | Same workflow from the terminal |
| `providers.py` | Builds NIM / Perplexity clients, plus the offline mock versions |
| `config.py` | All settings; any of them can be overridden in `.env` |
| `prompts.py` | Router, query-rewrite and answer prompts |

## Using your own documents

Drop files into `data/` (sub-folders are fine) and re-run `python ingest.py`. The three NVIDIA
datasheets in there come from the original example repo.

## Design choices (and how they map to the blog)

- **Router first.** A small, fast model decides whether retrieval is needed, so "fix this email"
  doesn't waste tokens and latency on a search. If the router errors, it falls back to retrieval.
- **Small model routes, large model answers.** Defaults are `llama-3.1-8b-instruct` and
  `llama-3.3-70b-instruct`. You can swap either in `.env`.
- **Query rewrite** removes format words and resolves "it" from the previous question.
- **Two sources in parallel**, joined with `ctx.collect_events`. Perplexity has a hard timeout
  (15s by default), so a slow web search never blocks the answer.
- **Rerank** (new): pulls the top 10 chunks from Milvus and keeps the best 4.
- **Citations** (new): sources are numbered in the prompt, and the reply lists only the ones cited.
- **Stateless workflow**: chat history lives in the UI session, so the backend is easy to put
  behind a REST API later.
- **Self-hosted NIM**: set `NIM_BASE_URL`. No other code changes are needed.

## Notes

- Model IDs on build.nvidia.com change over time. If you get a "model not found" error, choose a
  current one from the catalog and set it in `.env`.
- Milvus Lite is a single local file and is fine for prototyping. For large collections, point
  `milvus_path` at a Milvus server.
- Text is only pulled from PDFs; tables and images aren't. The blog's next step is to use
  vision-language models for multimodal ingestion.

Adapted from NVIDIA's `GenerativeAIExamples/community/routing-multisource-rag` (Apache-2.0).
