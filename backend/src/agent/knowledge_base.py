"""Knowledge base module for RAG retrieval using Chroma vector store."""

from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


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
            print("[Knowledge Base] Using BAAI/bge-small-zh-v1.5 for embeddings")
        except Exception as e:
            print(f"[Knowledge Base] Failed to load bge-small-zh-v1.5: {e}, falling back to all-MiniLM-L6-v2")
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
        print(f"[Knowledge Base] Docs directory not found: {docs_dir}")
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
            print(f"[Knowledge Base] Loaded {len(docs)} pages from {file_path.name}")
        except Exception as e:
            print(f"[Knowledge Base] Failed to load {file_path}: {e}")

    return documents


def _build_vectorstore(
    docs_dir: Path,
    chroma_dir: Path,
    embeddings: Embeddings,
) -> Chroma:
    """Build a new Chroma vector store from documents."""
    print("[Knowledge Base] Building vector store from documents...")
    raw_docs = _load_documents(docs_dir)
    if not raw_docs:
        print("[Knowledge Base] No documents found. Vector store will be empty.")
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
    print(f"[Knowledge Base] Split into {len(splits)} chunks")

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=str(chroma_dir),
        collection_name="knowledge_base",
    )
    print(f"[Knowledge Base] Vector store built at {chroma_dir}")
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
            print(f"[Knowledge Base] Loaded existing vector store with {count} documents")
            return vectorstore
        except Exception as e:
            print(f"[Knowledge Base] Failed to load existing store: {e}. Rebuilding...")

    return _build_vectorstore(docs_dir, chroma_dir, embeddings)


def retrieve_documents(query: str, top_k: int = 5) -> List[Document]:
    """Retrieve the top-k most relevant documents for the given query."""
    vectorstore = get_vectorstore()
    try:
        return vectorstore.similarity_search(query, k=top_k)
    except Exception as e:
        print(f"[Knowledge Base] Retrieval failed: {e}")
        return []
