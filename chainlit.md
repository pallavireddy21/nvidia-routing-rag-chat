# Routing + multi-source RAG chat

A rebuild of NVIDIA's *Creating RAG-Based Question-and-Answer LLM Workflows* example.
Each answer shows its steps: router → query rewrite → Milvus search + Perplexity web search (in parallel) → rerank → answer with numbered citations.
Small talk and text tasks (editing an email, writing code) skip retrieval entirely.
