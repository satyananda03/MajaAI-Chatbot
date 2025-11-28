import os
import re
import logging
from typing import Set, List

logger = logging.getLogger(__name__)

def load_stopwords(path: str) -> Set[str]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                stopwords = {line.strip().lower() for line in f if line.strip()}
            logger.info("Stopwords berhasil di load")
            return stopwords
        else:
            logger.warning(f"Stopword file tidak ditemukan {path}")
            return set()
    except Exception as e:
        logger.error(f"Gagal load stopwords: {e}")
        return set()

def preprocess_text(text: str, stopwords: Set[str]) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text)
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens