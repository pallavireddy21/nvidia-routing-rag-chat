"""Chat with the workflow in the terminal (no UI). Handy for testing.

    python cli.py                      # interactive
    python cli.py "What is the A100's memory bandwidth?"   # one-shot
"""

import asyncio
import logging
import sys
import time

from config import config
from workflow import QueryFlow, format_sources


async def ask(flow: QueryFlow, query: str, history: list[dict]) -> str:
    t0 = time.time()
    result = await flow.run(query=query, chat_messages=history)
    print(f"\n  route={result['route']}", end="")
    if result["transformed_query"]:
        print(f"  rewritten={result['transformed_query']!r}", end="")
    print("\n")
    answer = ""
    async for chunk in result["stream"]:
        answer += chunk.delta or ""
        print(chunk.delta or "", end="", flush=True)
    refs = format_sources(result["sources"], answer)
    print(refs)
    print(f"\n  ({time.time() - t0:.1f}s)\n")
    return answer + refs


async def main() -> None:
    logging.basicConfig(level=logging.INFO if "-v" in sys.argv else logging.WARNING, format="%(message)s")
    args = [a for a in sys.argv[1:] if a != "-v"]
    print(f"Mode: {'MOCK (no NVIDIA_API_KEY)' if config.mock else 'NVIDIA NIM'} | web search: {'on' if config.perplexity_api_key else 'off'}")
    flow = QueryFlow(timeout=config.workflow_timeout)
    history: list[dict] = []
    queries = args or None
    while True:
        query = queries.pop(0) if queries is not None else input("you> ").strip()
        if query:
            print(f"you> {query}" if queries is not None else "", end="")
            answer = await ask(flow, query, history)
            history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
        if queries is not None and not queries:
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, EOFError):
        pass
