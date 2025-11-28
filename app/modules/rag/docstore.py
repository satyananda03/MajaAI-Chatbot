import logging
from langchain_community.storage import RedisStore
from langchain.storage import EncoderBackedStore
from app.core.config import settings
from app.utils.document_serializer import encode, encode_key, decode

logger = logging.getLogger(__name__)

class DocStore:
    @staticmethod
    def get_doc_store(collection_name: str = settings.COLLECTION_NAME, redis_url: str = settings.REDIS_URL):
        namespace = f"docstore_{collection_name}"
        try:
            logger.info(f"Connecting to Redis DocStore {redis_url}, Namespace: {namespace})")
            raw_store = RedisStore(redis_url=redis_url, namespace=namespace)
            return EncoderBackedStore(store=raw_store,
                                    key_encoder=encode_key,
                                    value_serializer=encode,
                                    value_deserializer=decode)
        except Exception as e:
            logger.error(f"Gagal koneksi ke Redis {e}")
            raise e