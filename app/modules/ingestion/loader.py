import os
import glob
import logging
from typing import List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class PDFFileLoader:
    def load(self, source: str) -> List[Document]:
        try:
            logger.info(f"Loading PDF file: {source}")
            loader = PyMuPDFLoader(file_path=source)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source_type"] = "pdf"
                doc.metadata["file_path"] = source
            return documents
        except Exception as e:
            error_msg = f"Gagal memuat PDF {source}: {str(e)}"
            raise RuntimeError(error_msg) from e

class PDFDirectoryLoader:
    def __init__(self):
        self.single_loader = PDFFileLoader()

    def load(self, source: str, recursive: bool = True) -> List[Document]:
        documents: List[Document] = []
        search_pattern = os.path.join(source, "*.pdf")
        # Ambil list semua file path
        pdf_files = glob.glob(search_pattern)
        if not pdf_files:
            logger.warning(f"Tidak ditemukan file PDF di: {source}")
            return []
        logger.info(f"Ditemukan {len(pdf_files)} file PDF")
        # Loop setiap file di dalam direktori
        for file_path in pdf_files:
            # Skip hidden files
            if "/." in file_path or "\\." in file_path:
                continue
            # Panggil loader satuan
            docs = self.single_loader.load(file_path)
            documents.extend(docs)
        logger.info(f"Total {len(documents)} halaman berhasil dimuat dari direktori")
        return documents