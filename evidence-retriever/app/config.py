import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    MYSQL_DSN: str = os.getenv(
        "MYSQL_DSN",
        "mysql+pymysql://user:pass@host:3306/dbname?charset=utf8mb4",
    )
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "evidence_chunks")
    QWEN_EMBEDDING_API_KEY: str = os.getenv("QWEN_EMBEDDING_API_KEY", "")
    QWEN_EMBEDDING_BASE_URL: str = os.getenv("QWEN_EMBEDDING_BASE_URL", "")
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "")
    QWEN_EMBEDDING_TIMEOUT_SECONDS: int = int(
        os.getenv("QWEN_EMBEDDING_TIMEOUT_SECONDS", "30")
    )
    ENABLE_RERANK: bool = _env_bool("ENABLE_RERANK", False)
    QWEN_RERANK_BASE_URL: str = os.getenv("QWEN_RERANK_BASE_URL", "")
    QWEN_RERANK_MODEL: str = os.getenv("QWEN_RERANK_MODEL", "")
    QWEN_RERANK_TIMEOUT_SECONDS: int = int(
        os.getenv("QWEN_RERANK_TIMEOUT_SECONDS", "30")
    )


settings = Settings()
