from typing import List
from langchain_core.documents import Document

def document_parser(docs: List[Document]) -> str:
    return "\n".join(doc.page_content for doc in docs)

def url_parser(docs: List[Document]) -> List[str]:
    all_urls = []
    for doc in docs:
        urls = doc.metadata.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        all_urls.extend(urls)
    return sorted(set(all_urls))