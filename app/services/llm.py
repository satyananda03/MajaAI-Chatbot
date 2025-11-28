import os
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_aws import ChatBedrock
from app.core.config import settings

def create_llm(provider: str, model_id: str, temperature: float, max_tokens: int, streaming: bool) -> BaseChatModel:
    if provider == "aws":
        os.environ["AWS_ACCESS_KEY_ID"] = settings.AWS_ACCESS_KEY_ID
        os.environ["AWS_SECRET_ACCESS_KEY"] = settings.AWS_SECRET_ACCESS_KEY
        os.environ["AWS_REGION"] = settings.AWS_REGION
        return ChatBedrock(
            model_id=model_id,
            region_name=settings.AWS_REGION,
            streaming=streaming,
            max_tokens=max_tokens,
            temperature=temperature
        )
    else:
        raise ValueError(f"Provider LLM tidak didukung")

@lru_cache(maxsize=3)
def get_llm(provider: str = settings.LLM_PROVIDER, model_id: str = settings.LLM_ID, temperature: float = settings.TEMPERATURE, max_tokens: int = settings.MAX_TOKEN, streaming: bool = settings.STREAMING) -> BaseChatModel:
    return create_llm(provider, model_id, temperature, max_tokens, streaming)