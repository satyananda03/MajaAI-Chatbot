from app.modules.ingestion.loader import PDFDirectoryLoader
from app.modules.ingestion.splitter import ParentChildSplitter
from app.modules.ingestion.indexer import ParentChildIndexer, BM25Indexer
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

def run_ingestion_pipeline(folder_path: str):
    logger.info("Loading Data")
    loader = PDFDirectoryLoader()
    raw_docs = loader.load(folder_path)
    logger.info("Splitting Dokumen")
    splitter = ParentChildSplitter(
        parent_chunk_size=settings.PARENT_CHUNK_SIZE,
        parent_chunk_overlap=settings.PARENT_CHUNK_OVERLAP,
        child_chunk_size=settings.CHILD_CHUNK_SIZE,
        child_chunk_overlap=settings.CHILD_CHUNK_OVERLAP
    )
    split_result = splitter.split_documents(raw_docs)
    logger.info("Indexing Document Chunks")
    vector_indexer = ParentChildIndexer()
    result = vector_indexer.index_documents(split_result)
    logger.info(f"{result}")
    logger.info("Building BM25 Index")
    keyword_indexer = BM25Indexer()
    keyword_indexer.build_and_save_index()
    logger.info("Ingestion DONE")