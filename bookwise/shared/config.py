from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration manager for BookWise AI services.
    Supports environment variable loading with sensible defaults and fallback options.
    """
    SERVICE_NAME: str = "bookwise-service"
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"
    
    # API & Gateway Settings
    API_PREFIX: str = "/api/v1"
    SECRET_KEY: str = "bookwise-secret-key-change-in-production-32bytes"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database Settings (PostgreSQL with SQLite fallback for local test)
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/bookwise_db"
    FALLBACK_DATABASE_URL: str = "sqlite:///./bookwise_local.db"
    ALLOW_SQLITE_FALLBACK: bool = True  # set false in containers so a bad DATABASE_URL fails loudly
    
    # Vector Database Settings (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None
    QDRANT_COLLECTION_NAME: str = "book_embeddings"
    VECTOR_DIMENSION: int = 384  # sentence-transformers all-MiniLM-L6-v2 dimension
    
    # Redis Cache & Queue Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_QUEUE_NAME: str = "llm_explanation_jobs"
    CACHE_DEFAULT_TTL: int = 3600  # 1 hour
    
    # LLM Settings
    LLM_PROVIDER: str = "local"  # "openai", "ollama", "huggingface", "local"
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_TIMEOUT_SECONDS: float = 5.0
    LLM_JOB_WAIT_SECONDS: float = 25.0  # how long the gateway waits for the worker before using the fallback text
    
    # Embedding Model Settings
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    # Microservice Internal URLs (for Phase 2+)
    USER_SERVICE_URL: str = "http://localhost:8001"
    BOOK_SERVICE_URL: str = "http://localhost:8002"
    RETRIEVAL_SERVICE_URL: str = "http://localhost:8003"
    RECOMMENDATION_SERVICE_URL: str = "http://localhost:8004"
    ANALYTICS_SERVICE_URL: str = "http://localhost:8005"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
