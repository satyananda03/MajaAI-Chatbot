import logging
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    @staticmethod
    def get_vector_store(embedding_model: Embeddings, collection_name: str = settings.COLLECTION_NAME, persist_directory: str = settings.PERSIST_DIR):
        return Chroma(embedding_function=embedding_model, 
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                    collection_metadata={"hnsw:space": "cosine"}
                    )