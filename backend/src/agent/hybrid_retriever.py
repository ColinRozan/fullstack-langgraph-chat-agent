"""Hybrid retriever combining vector search, BM25, and reranking.

Retrieval pipeline:
    1. Vector search (Chroma)  → top candidates
    2. BM25 keyword search      → top candidates
    3. RRF fusion               → merged candidate list
    4. Cross-encoder rerank     → final ranked list
"""

import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel

from agent.knowledge_base import get_vectorstore

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
try:
    from rank_bm25 import BM25Okapi

    _HAS_BM25 = True
except Exception:
    BM25Okapi = None  # type: ignore[misc,assignment]
    _HAS_BM25 = False

# ---------------------------------------------------------------------------
# Reranker: fastembed cross-encoder (preferred) → LLM fallback
# ---------------------------------------------------------------------------
_RERANKER = None
_RERANKER_TYPE: Optional[str] = None


def _init_reranker() -> Optional[Any]:
    """Lazy-initialize the reranker."""
    global _RERANKER, _RERANKER_TYPE
    if _RERANKER is not None:
        return _RERANKER

    # Try fastembed cross-encoder first
    try:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        model_name = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
        _RERANKER = TextCrossEncoder(model_name=model_name)
        _RERANKER_TYPE = "fastembed"
        print(f"[HybridRetriever] Loaded fastembed reranker: {model_name}")
        return _RERANKER
    except Exception as e:
        print(f"[HybridRetriever] fastembed reranker not available: {e}")

    # If fastembed unavailable, mark for LLM fallback later
    _RERANKER_TYPE = "llm_fallback"
    print("[HybridRetriever] Will use LLM fallback for reranking")
    return None


# ---------------------------------------------------------------------------
# Tokenization (lightweight, no extra deps)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Tokenize text for BM25.

    Extracts English words / numbers and individual CJK characters.
    """
    text = text.lower()
    # English words and numbers
    tokens = re.findall(r"[a-z0-9]+", text)
    # Individual CJK characters (common Chinese range)
    tokens.extend(re.findall(r"[一-鿿]", text))
    return tokens


# ---------------------------------------------------------------------------
# LLM-based rerank fallback
# ---------------------------------------------------------------------------

_RERANK_PROMPT_TEMPLATE = """\
You are a relevance scoring assistant. Given a user query and a document, rate how relevant the document is to the query on a scale of 0 to 10.

Query: {query}

Document:
{doc}

Respond with a JSON object: {{"score": 7}}
"""


def _llm_rerank(query: str, docs: List[Document], llm: BaseChatModel) -> List[Tuple[Document, float]]:
    """Rerank documents using an LLM to score query-doc relevance."""
    scored: List[Tuple[Document, float]] = []
    for doc in docs:
        prompt = _RERANK_PROMPT_TEMPLATE.format(query=query, doc=doc.page_content[:800])
        try:
            response = llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            # Extract JSON score
            import json as _json

            m = re.search(r'(\{.*\})', raw_text, re.DOTALL)
            if m:
                parsed = _json.loads(m.group(1))
                score = float(parsed.get("score", 5))
            else:
                score = 5.0
        except Exception:
            score = 5.0
        scored.append((doc, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Combine vector search, BM25, and optional cross-encoder reranking."""

    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        chroma_dir: Optional[Path] = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        llm: Optional[BaseChatModel] = None,
        enable_bm25: bool = True,
        enable_rerank: bool = True,
    ):
        self.vectorstore = get_vectorstore(docs_dir, chroma_dir)
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.llm = llm
        self.enable_bm25 = enable_bm25 and _HAS_BM25
        self.enable_rerank = enable_rerank

        self.bm25_index: Optional[Any] = None
        self.doc_list: List[Document] = []
        self._build_bm25_index()

        self.reranker = _init_reranker() if self.enable_rerank else None

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _build_bm25_index(self) -> None:
        """Build or load a BM25 index backed by the current Chroma collection."""
        if not self.enable_bm25:
            print("[HybridRetriever] BM25 disabled")
            return

        # Determine cache path based on chroma persist dir
        chroma_dir = getattr(self.vectorstore, "_persist_directory", None)
        if chroma_dir is None:
            from agent.knowledge_base import DEFAULT_CHROMA_DIR

            chroma_dir = DEFAULT_CHROMA_DIR
        cache_dir = Path(chroma_dir).parent / "bm25"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "bm25_index.pkl"
        meta_path = cache_dir / "bm25_meta.json"

        # Count current documents in Chroma
        try:
            collection = self.vectorstore._collection
            chroma_count = collection.count()
        except Exception as e:
            print(f"[HybridRetriever] Failed to get Chroma count: {e}")
            chroma_count = -1

        # Try loading cached index if document counts match
        if cache_path.exists() and meta_path.exists() and chroma_count >= 0:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("chroma_count") == chroma_count:
                    with open(cache_path, "rb") as f:
                        data = pickle.load(f)
                    self.bm25_index = data["bm25"]
                    self.doc_list = data["docs"]
                    print(f"[HybridRetriever] Loaded cached BM25 index ({len(self.doc_list)} docs)")
                    return
            except Exception as e:
                print(f"[HybridRetriever] Failed to load cached BM25: {e}. Rebuilding...")

        # Fetch all documents from Chroma
        try:
            collection = self.vectorstore._collection
            result = collection.get(include=["documents", "metadatas"])
            documents = result.get("documents", [])
            metadatas = result.get("metadatas", [])
        except Exception as e:
            print(f"[HybridRetriever] Failed to fetch docs from Chroma: {e}")
            return

        if not documents:
            print("[HybridRetriever] No documents in Chroma; BM25 index empty")
            return

        self.doc_list = []
        for text, meta in zip(documents, metadatas):
            self.doc_list.append(Document(page_content=text, metadata=meta or {}))

        tokenized_docs = [_tokenize(doc.page_content) for doc in self.doc_list]
        self.bm25_index = BM25Okapi(tokenized_docs)

        # Persist cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({"bm25": self.bm25_index, "docs": self.doc_list}, f)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"chroma_count": chroma_count}, f)
            print(f"[HybridRetriever] Built and cached BM25 index ({len(self.doc_list)} docs)")
        except Exception as e:
            print(f"[HybridRetriever] Failed to cache BM25 index: {e}")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> List[Document]:
        """Execute the full hybrid retrieval pipeline."""
        # 1. Vector search
        vector_docs = self._vector_search(query)

        # 2. BM25 search
        bm25_docs: List[Document] = []
        if self.enable_bm25 and self.bm25_index is not None:
            bm25_docs = self._bm25_search(query)

        # 3. RRF fusion
        fused = self._rrf_fuse(vector_docs, bm25_docs, k=60)

        # 4. Rerank
        if self.enable_rerank and len(fused) > self.rerank_top_k:
            fused = self._rerank(query, fused)

        return fused[: self.rerank_top_k]

    def _vector_search(self, query: str) -> List[Document]:
        """Run vector similarity search."""
        try:
            return self.vectorstore.similarity_search(query, k=self.top_k * 2)
        except Exception as e:
            print(f"[HybridRetriever] Vector search failed: {e}")
            return []

    def _bm25_search(self, query: str) -> List[Document]:
        """Run BM25 keyword search."""
        if self.bm25_index is None or not self.doc_list:
            return []
        try:
            tokenized_query = _tokenize(query)
            scores = self.bm25_index.get_scores(tokenized_query)
            # Get top indices
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[: self.top_k * 2]
            return [self.doc_list[i] for i in top_indices]
        except Exception as e:
            print(f"[HybridRetriever] BM25 search failed: {e}")
            return []

    def _rrf_fuse(
        self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60
    ) -> List[Document]:
        """Fuse two ranked lists using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Document] = {}

        def _doc_key(doc: Document) -> str:
            # Use content + source as unique key
            source = doc.metadata.get("source", "") if hasattr(doc, "metadata") else ""
            return f"{source}::{doc.page_content[:200]}"

        for rank, doc in enumerate(vector_docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)

        for rank, doc in enumerate(bm25_docs, start=1):
            key = _doc_key(doc)
            doc_map[key] = doc
            scores[key] = scores.get(key, 0) + 1.0 / (k + rank)

        # Sort by fused score descending
        sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [doc_map[key] for key in sorted_keys]

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank documents using cross-encoder or LLM fallback."""
        # Fastembed reranker path
        if _RERANKER_TYPE == "fastembed" and self.reranker is not None:
            try:
                return self._fastembed_rerank(query, docs)
            except Exception as e:
                print(f"[HybridRetriever] Fastembed rerank failed: {e}")

        # LLM fallback path
        if self.llm is not None:
            try:
                scored = _llm_rerank(query, docs, self.llm)
                return [doc for doc, _ in scored]
            except Exception as e:
                print(f"[HybridRetriever] LLM rerank failed: {e}")

        # If everything fails, return as-is
        return docs

    def _fastembed_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        """Rerank using fastembed TextCrossEncoder."""
        texts = [doc.page_content for doc in docs]
        # fastembed API: rerank(query, documents) returns iterable of (query, doc, score)
        results = list(self.reranker.rerank(query, texts))
        # results is typically list of tuples; sort by score descending
        scored = []
        for item in results:
            # Handle different return shapes gracefully
            if isinstance(item, tuple) and len(item) >= 3:
                _, text, score = item
            elif isinstance(item, dict):
                text = item.get("document", "")
                score = item.get("score", 0)
            else:
                continue
            scored.append((text, float(score)))

        # Map back to Document objects by content
        text_to_doc = {doc.page_content: doc for doc in docs}
        scored.sort(key=lambda x: x[1], reverse=True)
        reranked = []
        for text, _ in scored:
            if text in text_to_doc:
                reranked.append(text_to_doc[text])
        # If mapping lost any docs, append remaining in original order
        seen = set(d.page_content for d in reranked)
        for doc in docs:
            if doc.page_content not in seen:
                reranked.append(doc)
        return reranked


# ---------------------------------------------------------------------------
# Convenience singleton / factory
# ---------------------------------------------------------------------------

_hybrid_retriever_singleton: Optional[HybridRetriever] = None


def get_hybrid_retriever(
    docs_dir: Optional[Path] = None,
    chroma_dir: Optional[Path] = None,
    top_k: int = 10,
    rerank_top_k: int = 5,
    llm: Optional[BaseChatModel] = None,
    enable_bm25: bool = True,
    enable_rerank: bool = True,
) -> HybridRetriever:
    """Return a HybridRetriever instance (singleton)."""
    global _hybrid_retriever_singleton
    if _hybrid_retriever_singleton is None:
        _hybrid_retriever_singleton = HybridRetriever(
            docs_dir=docs_dir,
            chroma_dir=chroma_dir,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            llm=llm,
            enable_bm25=enable_bm25,
            enable_rerank=enable_rerank,
        )
    return _hybrid_retriever_singleton
