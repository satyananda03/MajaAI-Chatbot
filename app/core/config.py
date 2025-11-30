from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # PROMPT
    PROMPT_VERSION: str 
    PROMPT_DIR: str

    # AWS
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    AWS_REGION: str

    # EMBEDDING
    EMBEDDING_PROVIDER: str
    EMBEDDING_MODEL_ID: str

    # LLM
    LLM_PROVIDER: str
    LLM_ID: str
    MAX_TOKEN: int
    TEMPERATURE: float
    STREAMING: bool

    # DATABASE
    REDIS_URL: str
    PERSIST_DIR: str
    COLLECTION_NAME: str

    # BM25
    BM25_INDEX_PATH: str
    STOPWORDS_PATH: str

    # RETRIEVER
    RETRIEVAL_STRATEGY: str
    MAX_RESULTS: int
    BM25_SEARCH_K: int
    VECTOR_SEARCH_K: int
    BM25_WEIGHT: float
    VECTOR_WEIGHT: float
    RRF_CONSTANT: int
    SCORE_THRESHOLD: float

    # INGESTION
    PARENT_CHUNK_SIZE: int
    PARENT_CHUNK_OVERLAP: int
    CHILD_CHUNK_SIZE: int
    CHILD_CHUNK_OVERLAP: int

    # Load .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()