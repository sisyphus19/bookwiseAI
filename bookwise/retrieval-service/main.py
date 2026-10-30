import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vector_db import vector_store
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import RetrievalRequest

logger = get_logger("retrieval-service")


def _rebuild_local_index_from_db() -> None:
    """In-memory mode starts empty on every boot; repopulate from the books table so /retrieve isn't blind."""
    if vector_store.points_count() > 0:
        return
    try:
        from core.database import BookModel, SessionLocal
        db = SessionLocal()
        try:
            rows = db.query(BookModel).all()
            books = [{
                "id": r.id, "title": r.title, "author": r.author, "genres": r.genres or [],
                "description": r.description or "", "rating": r.rating, "page_count": r.page_count,
                "publication_year": r.publication_year,
            } for r in rows]
        finally:
            db.close()
        if books:
            vector_store.upsert_books(books)
            logger.info(f"Rebuilt vector index from database: {len(books)} books.")
    except Exception as e:
        logger.warning(f"Could not rebuild vector index from database ({e}).")


@asynccontextmanager
async def lifespan(_: FastAPI):
    if vector_store.qdrant_client is None:  # Qdrant persists on its own; only the memory store needs a rebuild
        _rebuild_local_index_from_db()
    yield


app = FastAPI(title="BookWise Retrieval Service (RAG)", version="1.0.0", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "service": "retrieval-service", "points_count": vector_store.points_count()}

@app.get("/health/live")
def live():
    return {"status": "alive", "service": "retrieval-service"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post(f"{settings.API_PREFIX}/retrieve")
def retrieve_candidates(req: RetrievalRequest) -> list[dict[str, Any]]:
    candidates = vector_store.search_similar(query=req.query, top_k=req.top_k)
    return candidates

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
