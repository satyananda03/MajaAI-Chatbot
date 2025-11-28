import os
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from langchain_aws import BedrockEmbeddings
from app.core.config import settings
@lru_cache(maxsize=1)
def get_embeddings(provider: str = settings.EMBEDDING_PROVIDER, model_id: str = settings.EMBEDDING_MODEL_ID) -> Embeddings:
    if provider == "aws":
        os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
        os.environ["AWS_REGION"] = settings.AWS_REGION
        return BedrockEmbeddings(model_id=model_id,     
                                region_name=settings.AWS_REGION)
    else:
        raise ValueError(f"Provider embedding tidak didukung")
