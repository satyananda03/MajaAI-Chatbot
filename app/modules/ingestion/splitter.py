import re
import uuid
from collections import defaultdict
from typing import List, Dict, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

class ParentChildSplitter:
    def __init__(self, parent_chunk_size: int, parent_chunk_overlap: int, child_chunk_size: int, child_chunk_overlap: int):
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size,
                                                            chunk_overlap=parent_chunk_overlap,
                                                            separators=["\n", ". ", ", ", " ", ""])
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=child_chunk_size,
                                                            chunk_overlap=child_chunk_overlap,
                                                            separators=["\n\n", "\n", ". ", ", ", " ", ""])
                                                            
    def _preprocess_text(self, chunk_text: str) -> Tuple[str, List[str]]:
        # Ekstrak URL
        urls = re.findall(r'https?://\S+', chunk_text)
        # Replace URL dengan placeholder LINK
        text_clean = re.sub(r'https?://\S+', '[LINK]', chunk_text)
        # Hapus karakter yang tidak diinginkan tapi
        text_clean = re.sub(r'[^\w\s\[\],.:()%=+\-/]', '', text_clean)
        # Rapikan newline ganda > 2 menjadi 1
        text_clean = re.sub(r'\n{2,}', '\n', text_clean)
        # Rapikan spasi berlebihan
        text_clean = re.sub(r'[ ]{2,}', ' ', text_clean)
        # Fix multiple blank lines -> single newline
        text_clean = re.sub(r'\n\s*\n+', '\n', text_clean)
        return text_clean.strip(), urls

    def split_documents(self, documents: List[Document]) -> Dict[str, List[Document]]:
        # Grouping by Source/File Path
        docs_by_source = defaultdict(list)
        for doc in documents:
            source_key = doc.metadata.get("file_path")
            docs_by_source[source_key].append(doc)
        all_parent_docs = []
        all_child_docs = []
        logger.info(f"Processing split for {len(docs_by_source)} file")
        # Proses per File Source
        for source_name, doc_group in docs_by_source.items():
            # Gabungkan semua text dari satu file (merge pages)
            combined_text = "\n".join([d.page_content for d in doc_group])
            # Ambil metadata dasar dari halaman pertama
            base_metadata = doc_group[0].metadata.copy() if doc_group else {}
            # Buat satu dokumen besar sementara
            combined_doc = Document(page_content=combined_text, metadata=base_metadata)
            # Generate Parent Chunks
            parent_chunks = self.parent_splitter.split_documents([combined_doc])
            for p_doc in parent_chunks:
                # Cleaning Text
                clean_text, specific_urls = self._preprocess_text(p_doc.page_content)
                # Generate Parent ID (UUID)
                parent_id = str(uuid.uuid4())
                # Update Metadata Parent
                parent_meta = p_doc.metadata.copy()
                parent_meta.update({
                    "doc_id": parent_id,
                    "type": "parent",
                    "source": source_name,
                    "urls": specific_urls, 
                })
                final_parent_doc = Document(page_content=clean_text, metadata=parent_meta)
                all_parent_docs.append(final_parent_doc)
                # Generate Child Chunk dari Parent Chunk
                child_texts = self.child_splitter.split_text(clean_text)
                for c_text in child_texts:
                    child_meta = {
                        "parent_id": parent_id,
                        "type": "child",
                        "source": source_name,
                    }
                    child_doc = Document(page_content=c_text, metadata=child_meta)
                    all_child_docs.append(child_doc)

        logger.info(f"Splitting complete, Parents: {len(all_parent_docs)}, Children: {len(all_child_docs)}")
        return {"parents": all_parent_docs,
                "children": all_child_docs}