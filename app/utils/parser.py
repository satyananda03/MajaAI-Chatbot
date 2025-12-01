from typing import List
from langchain_core.documents import Document

def context_parser(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

def url_parser(docs: List[Document]) -> List[str]:
    urls = set()
    for doc in docs:
        doc_urls = doc.metadata.get("urls", [])
        if isinstance(doc_urls, str):
            urls.add(doc_urls)
        else:
            urls.update(doc_urls)
    return list(urls)