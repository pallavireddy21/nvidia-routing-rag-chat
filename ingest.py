"""Load documents from data/ into a Milvus Lite vector store.

Usage:  python ingest.py            (re-builds the index from scratch)
Supports PDF, DOCX, PPTX, MD, TXT, HTML, CSV (via LlamaIndex SimpleDirectoryReader).
Re-run this whenever you add files or switch between mock and real embeddings.
"""

import json
import logging
import time

from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore

from config import config
from providers import build_embed_model

logging.basicConfig(level=logging.WARNING)
logging.getLogger("grpc._server").setLevel(logging.CRITICAL)  # harmless Milvus Lite noise


def main() -> None:
    t0 = time.time()
    mode = "MOCK (offline hash embeddings)" if config.mock else f"NVIDIA NIM ({config.embedding_model})"
    print(f"Embedding mode: {mode}")

    print(f"Loading documents from {config.data_dir} ...")
    documents = SimpleDirectoryReader(input_dir=config.data_dir, recursive=True).load_data()
    # PDFs load one Document per page, with file_name + page_label metadata used for citations.
    print(f"Loaded {len(documents)} pages/documents.")

    embed_model = build_embed_model()
    dim = len(embed_model.get_text_embedding("dimension probe"))

    vector_store = MilvusVectorStore(
        uri=config.milvus_path,
        collection_name=config.collection_name,
        dim=dim,
        overwrite=True,
        use_async_client=False,
    )
    splitter = SentenceSplitter(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    VectorStoreIndex.from_documents(
        documents,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    with open(config.index_info_path, "w") as f:
        json.dump({"embedding_model": config.active_embedding_model, "dim": dim, "built_at": time.ctime()}, f)

    n = vector_store.client.get_collection_stats(config.collection_name)["row_count"]
    print(f"Indexed {n} chunks (dim={dim}) into {config.milvus_path} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
