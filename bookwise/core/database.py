from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from shared.config import settings
from shared.logging import get_logger

logger = get_logger("database-module")

Base = declarative_base()

# DB Engine Creation with Graceful Fallback
def _normalise_url(url: str) -> str:
    """SQLAlchemy 2.1 defaults bare postgresql:// to psycopg v3; we ship psycopg2, so pin the driver."""
    if url.startswith("postgresql://"):
        return "postgresql+psycopg2://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg2://" + url[len("postgres://"):]
    return url


def get_db_engine():
    db_url = _normalise_url(settings.DATABASE_URL)
    try:
        if db_url.startswith("sqlite"):
            engine = create_engine(db_url, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(db_url, pool_pre_ping=True, pool_size=10, max_overflow=20)
        with engine.connect():
            pass
        logger.info(f"Connected to primary database: {db_url.split('@')[-1] if '@' in db_url else db_url}")
        return engine
    except Exception as e:
        if not settings.ALLOW_SQLITE_FALLBACK:
            logger.error(f"Primary database connection failed ({e}) and ALLOW_SQLITE_FALLBACK=false.")
            raise
        logger.warning(f"Primary database connection failed ({e}). Falling back to SQLite local database.")
        engine = create_engine(settings.FALLBACK_DATABASE_URL, connect_args={"check_same_thread": False})
        return engine

engine = get_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# SQLAlchemy ORM Models

class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    favorite_genres = Column(JSON, default=list)  # list of strings
    preferred_book_length = Column(String(20), nullable=True)
    preferred_mood = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    feedbacks = relationship("FeedbackModel", back_populates="user")


class BookModel(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), index=True, nullable=False)
    author = Column(String(255), index=True, nullable=False)
    genres = Column(JSON, default=list)  # list of strings
    description = Column(Text, nullable=False)
    reviews_summary = Column(Text, nullable=True)
    rating = Column(Float, default=4.0)
    page_count = Column(Integer, default=300)
    publication_year = Column(Integer, default=2020)
    cover_image_url = Column(String(500), nullable=True)
    embedding_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    feedbacks = relationship("FeedbackModel", back_populates="book")


class FeedbackModel(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    book_id = Column(Integer, ForeignKey("books.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    is_helpful = Column(Boolean, default=True)
    comment = Column(Text, nullable=True)
    request_id = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("UserModel", back_populates="feedbacks")
    book = relationship("BookModel", back_populates="feedbacks")


def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables initialized successfully.")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
