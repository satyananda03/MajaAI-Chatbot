import logging
import pickle
from typing import List, Any, Optional, Set 
from collections import defaultdict
from pydantic import PrivateAttr
from langchain_core.documents import Document
from langchain_core.runnables import RunnableSerializable, RunnableConfig
from app.modules.rag.vectorstore import VectorStore
from app.modules.rag.docstore import DocStore
from app.services.embbedding import get_embeddings
from app.utils.text_preprocessing import load_stopwords, preprocess_text
from app.core.config import settings

logger = logging.getLogger(__name__)

def weighted_rrf(doc_lists: List[List[Document]], weights: List[float], c: int) -> List[Document]:
    rrf_score_map = defaultdict(float)
    doc_map = {}
    for doc_list, weight in zip(doc_lists, weights):
        for rank, doc in enumerate(doc_list):
            doc_id = str(doc.metadata.get("doc_id") or hash(doc.page_content))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            rrf_score_map[doc_id] += weight * (1 / (c + rank))
    sorted_doc_ids = sorted(rrf_score_map.keys(), key=lambda x: rrf_score_map[x], reverse=True)
    final_docs = []
    for did in sorted_doc_ids:
        doc = doc_map[did]
        doc.metadata["rrf_score"] = rrf_score_map[did]
        final_docs.append(doc)
    return final_docs

class ParentChildRetriever(RunnableSerializable):
    _vector_store: Any = PrivateAttr()
    _doc_store: Any = PrivateAttr()
    _bm25_retriever: Any = PrivateAttr(default=None)
    _stopwords: Set[str] = PrivateAttr(default_factory=set)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._vector_store = VectorStore.get_vector_store(
            embedding_model = get_embeddings(),
            collection_name = settings.COLLECTION_NAME
        )
        self._doc_store = DocStore.get_doc_store(collection_name=settings.COLLECTION_NAME)
        self._stopwords = load_stopwords(settings.STOPWORDS_PATH)

    def _text_preprocessing(self, text: str):
        return preprocess_text(text, self._stopwords)

    def _load_bm25_retriever(self):
        if self._bm25_retriever:
            return self._bm25_retriever
        try:
            with open(settings.BM25_INDEX_PATH, "rb") as f:
                retriever = pickle.load(f)
            retriever.preprocess_func = self._text_preprocessing
            retriever.k = settings.BM25_SEARCH_K * 3
            self._bm25_retriever = retriever
            return retriever
        except Exception as e:
            logger.error(f"Gagal load BM25 Index: {e}")
            return None

    def _search_children_vector(self, query: str) -> List[Document]:
        results = self._vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=settings.VECTOR_SEARCH_K,
            score_threshold=settings.SCORE_THRESHOLD,
            filter={"type": "child"}
        )
        vector_results = []
        for doc, score in results:
            if doc.metadata.get("type") == "child":
                vector_results.append(doc)
        return vector_results

    def _search_children_hybrid(self, query: str) -> List[Document]:
        # A. Vector Search
        vector_docs = self._search_children_vector(query)
        # B. BM25 Search
        bm25_retriever = self._load_bm25_retriever()
        bm25_docs = []
        if bm25_retriever:
            raw_docs = bm25_retriever.invoke(query)
            filtered_docs = [d for d in raw_docs if d.metadata.get("type") == "child"]
            bm25_docs = filtered_docs[:settings.BM25_SEARCH_K]
        # C. RRF Scoring
        fused_docs = weighted_rrf(
            doc_lists=[vector_docs, bm25_docs],
            weights=[settings.VECTOR_WEIGHT, settings.BM25_WEIGHT], 
            c=settings.RRF_CONSTANT
        )
        return fused_docs

    def _fetch_parents(self, child_docs: List[Document]) -> List[Document]:
        unique_parent_map = {}
        for child in child_docs:
            pid = child.metadata.get("parent_id")
            if pid and pid not in unique_parent_map:
                unique_parent_map[pid] = child
        parent_ids = list(unique_parent_map.keys())
        if not parent_ids:
            return []
        try:
            parent_docs = self._doc_store.mget(parent_ids)
        except Exception as e:
            logger.error(f"Redis Fetch Error: {e}")
            return []
            
        final_results = []
        for pid, p_doc in zip(parent_ids, parent_docs):
            if p_doc:
                child_ref = unique_parent_map[pid]
                rrf_score = child_ref.metadata.get("rrf_score", 0)
                p_doc.metadata["retrieval_score"] = rrf_score
                p_doc.metadata["matched_child_content"] = child_ref.page_content
                final_results.append(p_doc)
        final_results.sort(key=lambda x: x.metadata.get("retrieval_score", 0), reverse=True)
        return final_results[:settings.MAX_RESULTS]

    def invoke(self, query: str, config: Optional[RunnableConfig] = None, **kwargs) -> List[Document]:
        if settings.RETRIEVAL_STRATEGY == "hybrid":
            child_docs = self._search_children_hybrid(query)
        elif settings.RETRIEVAL_STRATEGY == "vector":
            child_docs = self._search_children_vector(query)
        else:
            raise ValueError(f"Strategy tidak didukung")       
        if not child_docs:
            return [] 
        parent_docs = self._fetch_parents(child_docs)
        return parent_docs