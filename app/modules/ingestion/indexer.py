import uuid
import logging
import pickle
from langchain_community.retrievers import BM25Retriever
from typing import List, Dict
from functools import partial
from langchain_core.documents import Document
from app.services.embbedding import get_embeddings
from app.modules.rag.vectorstore import VectorStore
from app.modules.rag.docstore import DocStore
from app.utils.text_preprocessing import load_stopwords, preprocess_text
from app.core.config import settings

logger = logging.getLogger(__name__)

class ParentChildIndexer:
    def __init__(self):
        self.embedding_model = get_embeddings()
        self.collection_name = settings.COLLECTION_NAME
        self.vector_store = VectorStore.get_vector_store(
            embedding_model=self.embedding_model,
            collection_name=self.collection_name
        )
        self.doc_store = DocStore.get_doc_store(collection_name=self.collection_name)

    def index_documents(self, split_result: Dict[str, List[Document]]):
        parent_docs = split_result.get("parents", [])
        child_docs = split_result.get("children", [])
        logger.info(f"Indexing {len(parent_docs)} Parents & {len(child_docs)} Child")
        # SIMPAN PARENTS KE REDIS
        if parent_docs:
            try:
                parent_key_value_pairs = []
                for doc in parent_docs:
                    doc_id = doc.metadata.get("doc_id") or str(uuid.uuid4())
                    doc.metadata["doc_id"] = doc_id
                    parent_key_value_pairs.append((doc_id, doc))
                self.doc_store.mset(parent_key_value_pairs)
                logger.info(f"Berhasil menyimpan {len(parent_docs)} Parent Chunk ke Redis.")
            except Exception as e:
                logger.error(f"Error simpan Parent Chunk ke Redis: {e}")
                raise e
        # SIMPAN CHILDREN KE CHROMA
        if child_docs:
            try:
                valid_children = [d for d in child_docs if "parent_id" in d.metadata]
                self.vector_store.add_documents(valid_children)
                logger.info(f"Berhasil menyimpan {len(valid_children)} Child Chunk ke ChromaDB.")
            except Exception as e:
                logger.error(f"Error simpan Children Chunk ke Chroma: {e}")
                raise e
        return {"Parents indexed": len(parent_docs), "Children indexed": len(child_docs)}

class BM25Indexer:
    def __init__(self):
        self.vector_store = VectorStore.get_vector_store(
            collection_name=settings.COLLECTION_NAME,
            embedding_model=get_embeddings()
        )
    def build_and_save_index(self):
        # 1. Load stopwords
        stopwords = load_stopwords(settings.STOPWORDS_PATH)
        # 2. Fetch documents child dari vectordb
        try:
            result = self.vector_store.get(
                where={"type": "child"},
                include=["documents", "metadatas"]
            )
        except Exception as e:
            logger.error(f"Gagal fetch data dari Chroma: {e}")
            raise e
        raw_docs = result.get("documents", [])
        raw_metadatas = result.get("metadatas", [])
        if not raw_docs:
            logger.warning("Tidak ditemukan dokumen 'child' di Chroma")
            return
        logger.info(f"Indexing {len(raw_docs)} dokumen")
        # 3. Convert ke LangChain Document
        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw_docs, raw_metadatas)
        ]
        # 4. Build BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(documents, preprocess_func=partial(preprocess_text, stopwords=stopwords))
        # 5. Save index
        try:
            with open(settings.BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25_retriever, f)
            logger.info(f"BM25 Index berhasil disimpan di: {settings.BM25_INDEX_PATH}")
        except Exception as e:
            logger.error(f"Gagal menyimpan file pickle: {e}")
            raise e