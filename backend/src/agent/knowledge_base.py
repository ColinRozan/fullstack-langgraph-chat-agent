"""Knowledge base module for RAG retrieval using Chroma vector store.

.. note::
    **Enterprise Vector DB Migration Path**

    Chroma (embedded / file-based) is retained for simplicity in single-node
    deployments.  When you need horizontal scaling, high availability, or
    multi-tenant isolation, migrate to one of the following alternatives:

    * **Qdrant** (recommended for mid-scale) – Rust-based, supports hybrid
      search (vector + payload filtering), distributed mode, and has a
      first-class LangChain integration (``langchain-qdrant``).
      Migration effort: low – swap ``Chroma`` → ``Qdrant`` in
      ``get_vectorstore()``.

    * **Milvus / Zilliz Cloud** (recommended for enterprise / SaaS) –
      GPU-accelerated indexing, RBAC, multi-tenancy, billion-scale vectors.
      Migration effort: medium – requires collection design + embedding
      dimension alignment.

    * **Pinecone** (managed, serverless) – Zero-ops, metadata filtering,
      automatic scaling.  Migration effort: low – fully managed but vendor
      lock-in.

    * **Weaviate** – GraphQL interface, modular AI integrations, hybrid
      search out of the box.

    Whichever target you choose, the BM25 + RRF + cross-encoder rerank
    pipeline in ``hybrid_retriever.py`` can be preserved; only the
    ``vectorstore`` source needs to change.
"""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agent.observability import get_logger
from agent import persistence

logger = get_logger(__name__)

# Default paths relative to backend/
DEFAULT_DOCS_DIR = Path(__file__).parent.parent.parent / "data" / "docs"
DEFAULT_CHROMA_DIR = Path(__file__).parent.parent.parent / "data" / "chroma"


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible wrapper around fastembed (ONNX-based) embedding.

    Uses BAAI/bge-small-zh-v1.5 for high-quality Chinese semantic retrieval.
    Falls back to all-MiniLM-L6-v2 if the Chinese model fails to load.
    """

    def __init__(self) -> None:
        from fastembed import TextEmbedding
        try:
            self._model = TextEmbedding(model_name="BAAI/bge-small-zh-v1.5")
            logger.info("embedding_model_loaded", model="BAAI/bge-small-zh-v1.5")
        except Exception as e:
            logger.warning("embedding_model_fallback", error=str(e))
            from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
            self._model = None
            self._fallback = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self._model is None:
            return self._fallback(texts)
        results = list(self._model.embed(texts))
        # fastembed returns numpy arrays; convert to plain Python lists
        return [r.tolist() if hasattr(r, "tolist") else list(r) for r in results]

    def embed_query(self, text: str) -> List[float]:
        if self._model is None:
            return self._fallback([text])[0]
        result = list(self._model.embed([text]))[0]
        return result.tolist() if hasattr(result, "tolist") else list(result)


def _load_documents(docs_dir: Path) -> List[Document]:
    """Load all supported documents from the docs directory."""
    documents: List[Document] = []
    if not docs_dir.exists():
        logger.warning("docs_dir_not_found", path=str(docs_dir))
        return documents

    for file_path in docs_dir.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".pdf":
                from langchain_community.document_loaders import PyPDFLoader

                loader = PyPDFLoader(str(file_path))
                docs = loader.load()
            elif suffix in (".txt", ".md", ".markdown"):
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
            else:
                continue

            # Normalize metadata source to relative path
            for doc in docs:
                doc.metadata["source"] = str(file_path.relative_to(docs_dir))
            documents.extend(docs)
            logger.info("document_loaded", file=file_path.name, pages=len(docs))
        except Exception as e:
            logger.warning("document_load_failed", file=str(file_path), error=str(e))

    return documents


def _build_vectorstore(
    docs_dir: Path,
    chroma_dir: Path,
    embeddings: Embeddings,
) -> Chroma:
    """Build a new Chroma vector store from documents."""
    logger.info("building_vectorstore")
    raw_docs = _load_documents(docs_dir)
    if not raw_docs:
        logger.warning("no_documents_found")
        # Create empty store so retrieval doesn't crash
        return Chroma.from_documents(
            documents=[Document(page_content="", metadata={"source": "placeholder"})],
            embedding=embeddings,
            persist_directory=str(chroma_dir),
            collection_name="knowledge_base",
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(raw_docs)
    logger.info("documents_split", chunk_count=len(splits))

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name="knowledge_base",
    )
    logger.info("vectorstore_built", path=str(chroma_dir))

    # Track document index metadata
    try:
        for doc in raw_docs:
            source = doc.metadata.get("source", "unknown")
            persistence.upsert_document_index(
                filename=source,
                chunk_count=len(splits),
                total_chars=sum(len(d.page_content) for d in splits),
            )
    except Exception as exc:
        logger.warning("document_index_tracking_failed", error=str(exc))

    return vectorstore


def get_vectorstore(
    docs_dir: Path | None = None,
    chroma_dir: Path | None = None,
) -> Chroma:
    """Get or create the Chroma vector store.

    If a persisted store exists and is non-empty, load it.
    Otherwise, index documents from the docs directory.
    """
    docs_dir = docs_dir or DEFAULT_DOCS_DIR
    chroma_dir = chroma_dir or DEFAULT_CHROMA_DIR
    embeddings = FastEmbedEmbeddings()

    if chroma_dir.exists() and any(chroma_dir.iterdir()):
        try:
            vectorstore = Chroma(
                persist_directory=str(chroma_dir),
                embedding_function=embeddings,
                collection_name="knowledge_base",
            )
            count = vectorstore._collection.count()
            logger.info("vectorstore_loaded", document_count=count)
            return vectorstore
        except Exception as e:
            logger.warning("vectorstore_load_failed", error=str(e))

    return _build_vectorstore(docs_dir, chroma_dir, embeddings)


def retrieve_documents(
    query: str,
    top_k: int = 5,
    use_hybrid: bool = False,
    enable_bm25: bool = True,
    enable_rerank: bool = True,
    hybrid_top_k: int = 10,
    llm=None,
) -> List[Document]:
    """Retrieve the top-k most relevant documents for the given query.

    Args:
        query: The search query.
        top_k: Number of final documents to return.
        use_hybrid: Whether to use hybrid search (BM25 + vector + rerank).
        enable_bm25: Whether to enable BM25 in hybrid mode.
        enable_rerank: Whether to enable reranking in hybrid mode.
        hybrid_top_k: Number of candidates to retrieve per modality before fusion.
        llm: Optional LLM for LLM-based rerank fallback.
    """
    if use_hybrid:
        from agent.hybrid_retriever import get_hybrid_retriever

        retriever = get_hybrid_retriever(
            top_k=hybrid_top_k,
            rerank_top_k=top_k,
            enable_bm25=enable_bm25,
            enable_rerank=enable_rerank,
            llm=llm,
        )
        return retriever.retrieve(query)

    vectorstore = get_vectorstore()
    try:
        return vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        logger.warning("retrieval_failed", error=str(e))
        return []
