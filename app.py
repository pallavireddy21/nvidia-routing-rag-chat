"""Chainlit UI.   Run:  chainlit run app.py -w"""

import logging
import time

import chainlit as cl

from config import config
from workflow import QueryFlow, format_sources

logging.basicConfig(level=logging.INFO)


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(label="A100 memory bandwidth", message="What is the GPU memory bandwidth of the A100 80GB?"),
        cl.Starter(label="Compare T400 vs RTX 6000", message="Make a table comparing the T400 and RTX 6000 on memory, power, and display outputs."),
        cl.Starter(label="Fix my email (no RAG)", message='Fix the typos in this email: "Hi team, plese find teh attached report for reveiw."'),
        cl.Starter(label="Haiku about GPUs (no RAG)", message="Write a haiku about GPUs."),
    ]


@cl.on_chat_start
async def on_chat_start():
    # History is kept here, client-side, not inside a stateful LlamaIndex engine,
    # which makes it easy to move behind a REST API later.
    cl.user_session.set("history", [])
    cl.user_session.set("flow", QueryFlow(timeout=config.workflow_timeout))
    if config.mock:
        await cl.Message(
            content="⚠️ Running in **mock mode** (no `NVIDIA_API_KEY`). Retrieval is real, but answers are "
            "quoted source lines rather than LLM-written. Add your key to `.env` and re-run `python ingest.py`."
        ).send()


@cl.on_message
async def on_message(message: cl.Message):
    t0 = time.time()
    history = cl.user_session.get("history")
    flow: QueryFlow = cl.user_session.get("flow")
    reply = cl.Message(content="")

    try:
        result = await flow.run(query=message.content, chat_messages=history)
        n_tokens = 0
        async for chunk in result["stream"]:
            if chunk.delta:
                n_tokens += 1
                await reply.stream_token(chunk.delta)
        refs = format_sources(result["sources"], reply.content)
        if refs:
            await reply.stream_token(refs)
        route = "retrieval" if result["route"] == "rag" else "direct answer (no retrieval)"
        await reply.stream_token(f"\n\n*{route} · {n_tokens} streamed chunks · {time.time() - t0:.1f}s*")
    except Exception as e:
        logging.exception("workflow failed")
        reply.content = f"Sorry, something went wrong: `{e}`"

    await reply.send()
    history += [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": reply.content},
    ]
    cl.user_session.set("history", history)
