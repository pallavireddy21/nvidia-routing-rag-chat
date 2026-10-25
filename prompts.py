"""Prompts, adapted from NVIDIA's routing-multisource-rag example (Apache-2.0)."""

ROUTING_PROMPT = """Below is a user query or statement. You can use a document database to help answer it.

The database connects to:
- Internal documents (NVIDIA product documentation and datasheets)
- Perplexity, a general-purpose web search engine

If the user is making small talk, or asking for help with a task that needs no outside
information (editing an email, writing or debugging code, translating, summarizing text they
pasted), do NOT use the database.

Examples:
User: What is the capital of France?            -> use_search: true
User: How do I install the latest NVIDIA driver? -> use_search: true
User: Hello!                                     -> use_search: false
User: What are you?                              -> use_search: false
User: Count to 3.                                -> use_search: false
User: When is the B200 available?                -> use_search: true
User: What is the difference between the A100 and the A30? -> use_search: true
User: Check this email for typos "To whom it may concern, ..." -> use_search: false
User: Help me debug this Python code: "import pandas; ..."      -> use_search: false

Classify this query:
User: {query}
"""

QUERY_REWRITE_PROMPT = """You are a research assistant. Rewrite the text below into the most informative
search query for a keyword/vector search engine over NVIDIA documentation and the web.
Remove instructions about format (table, email, summary, bullet points). Expand acronyms.
Use the earlier question only to resolve references like "it" or "that one".
Return ONLY the rewritten query, with no commentary or quotes.

Examples:
Raw text: Make a table showing customer wins using gpus for bio
Rewritten: Examples of customers using graphics processing units (GPUs) in biology and bioinformatics

Raw text: Write an email to Walmart about how they benefit from DGX servers
Rewritten: Walmart usage of NVIDIA DGX servers

Earlier question(s):
{history_str}

Raw text: {query}
Rewritten:"""

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant. Follow these rules every time:
- Be brief and to the point. No disclaimers, no "As an AI language model".
- If the sources below don't support an answer, say you don't know. Never make facts up.
- When you use a source, cite it inline with its number in square brackets, e.g. [1] or [2][3].
  Only cite numbers that appear below. Do not invent a reference list; one is added automatically.
- Only create a table if the user asks for one; keep cells to 1-2 sentences and valid Markdown.
- Prefer newer information when dates are available.

Sources:
{context_str}

Conversation so far:
{history_str}

User: {query}
Assistant:"""

NO_RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Be brief and useful. No disclaimers.

Conversation so far:
{history_str}

User: {query}
Assistant:"""
