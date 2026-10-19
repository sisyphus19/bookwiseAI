"""Tests for the retrieval -> grounded generation path (no network, no Docker required)."""
import asyncio
import importlib.util
import json
import os
import sys

import httpx
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.schemas import RankedCandidate, ScoreBreakdown
from core import llm_provider
from core.llm_provider import (
    PROMPT_TEMPLATE, LocalMockProvider, OpenAIProvider, LLMReasoningService, _fmt_book_line,
)
from core.recommendation_engine import recommendation_engine, FeatureExtractor


def _candidate(book_id=1, title="The Hunger Games", description="", pages=0, year=2008):
    sb = ScoreBreakdown(semantic_score=0.8, genre_affinity=0.5, popularity_prior=0.87, recency_factor=0.3, final_score=0.7)
    return RankedCandidate(id=book_id, title=title, author="Suzanne Collins", genres=[], description=description,
                           rating=4.34, page_count=pages, publication_year=year, score_breakdown=sb,
                           explainability_badges=[])


# ---------- prompt / grounding ----------

def test_prompt_template_formats_without_keyerror():
    """Regression: unescaped JSON braces made str.format raise, so the real-LLM path always fell back."""
    out = PROMPT_TEMPLATE.format(query="q", genres="None", mood="Any", books_summary="ID: 1")
    assert '"explanations"' in out and "{{" not in out


def test_book_line_marks_missing_fields_unknown():
    line = _fmt_book_line(_candidate(pages=0, year=0))
    assert "Pages: unknown" in line and "Year: unknown" in line and "Description: none available" in line


def test_mock_provider_does_not_fabricate():
    res = asyncio.run(LocalMockProvider().generate_explanations("dystopian", [_candidate()]))
    text = res[0]["explanation"].lower() + " ".join(res[0]["reasoning_bullets"]).lower()
    assert "0-page" not in text and "narrative depth" not in text and "0 pages" not in text
    assert "no description" in text


def test_ranker_has_no_page_badge_when_pages_unknown():
    cands = [{"id": 1, "title": "A", "author": "B", "genres": [], "description": "", "rating": 4.5,
              "page_count": 0, "publication_year": 0, "semantic_similarity": 0.8}]
    ranked = recommendation_engine.process_recommendations(cands, top_k=1)
    assert not any("pages" in b for b in ranked[0].explainability_badges)
    assert ranked[0].page_count == 0 and ranked[0].publication_year == 0
    assert FeatureExtractor().compute_recency_factor(0) == 0.5


# ---------- LLM provider failure handling ----------

def _patch_openai(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(llm_provider.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **{k: v for k, v in kw.items() if k == "timeout"}))


def test_openai_success_uses_model_output(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert "Do NOT invent" in body["messages"][1]["content"]
        content = json.dumps({"explanations": [{"book_id": 1, "reasoning_bullets": ["✓ x"], "explanation": "from llm"}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    _patch_openai(monkeypatch, handler)
    svc = LLMReasoningService()
    svc.provider = OpenAIProvider(api_key="sk-test")
    recs, fallback = asyncio.run(svc.get_explained_recommendations("q", [_candidate()]))
    assert fallback is False and recs[0].llm_explanation == "from llm"


@pytest.mark.parametrize("handler", [
    lambda r: httpx.Response(500, text="boom"),
    lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "not json"}}]}),
])
def test_openai_failure_sets_fallback_flag(monkeypatch, handler):
    _patch_openai(monkeypatch, handler)
    svc = LLMReasoningService()
    svc.provider = OpenAIProvider(api_key="sk-test")
    recs, fallback = asyncio.run(svc.get_explained_recommendations("q", [_candidate()]))
    assert fallback is True and len(recs) == 1 and "no description" in recs[0].llm_explanation.lower()


def test_openai_timeout_sets_fallback_flag(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)
    _patch_openai(monkeypatch, handler)
    svc = LLMReasoningService()
    svc.provider = OpenAIProvider(api_key="sk-test")
    _, fallback = asyncio.run(svc.get_explained_recommendations("q", [_candidate()]))
    assert fallback is True


# ---------- worker round trip (local queue; Redis not needed) ----------

def test_worker_roundtrip_and_failed_job():
    from llm_worker import worker
    if worker.REDIS_CONNECTED:
        pytest.skip("round-trip test targets the local-queue path")
    cand = _candidate().model_dump()
    worker.LLMQueueProducer.enqueue_job({"job_id": "t1", "query": "q", "candidates": [cand]})
    assert asyncio.run(worker.process_next_job()) is True
    res = worker.LLMQueueProducer.get_result("t1")
    assert res["status"] == "completed" and res["explained_recommendations"][0]["book"]["id"] == 1

    worker.LLMQueueProducer.enqueue_job({"job_id": "t2", "query": "q", "candidates": [{"bad": "shape"}]})
    asyncio.run(worker.process_next_job())
    assert worker.LLMQueueProducer.get_result("t2")["status"] == "failed"  # published, not silently dropped


# ---------- Qdrant: query_points against a real engine (qdrant-client local mode) ----------

@pytest.fixture
def qdrant_store():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    from core.vector_db import VectorStoreManager
    store = VectorStoreManager(collection_name="test_books")
    store.qdrant_client = QdrantClient(":memory:")
    store.qdrant_client.create_collection("test_books", vectors_config=qm.VectorParams(size=settings.VECTOR_DIMENSION, distance=qm.Distance.COSINE))
    return store


def test_qdrant_upsert_count_and_query_points(qdrant_store):
    books = [{"id": i, "title": t, "author": a, "rating": 4.0, "publication_year": 2000}
             for i, (t, a) in enumerate([("The Hunger Games", "Suzanne Collins"), ("Harry Potter and the Sorcerer's Stone", "J.K. Rowling"),
                                         ("Pride and Prejudice", "Jane Austen")], start=1)]
    assert qdrant_store.points_count() == 0
    qdrant_store.upsert_books(books)
    qdrant_store.upsert_books(books)  # idempotent
    assert qdrant_store.points_count() == 3
    hits = qdrant_store.search_similar("Hunger Games Suzanne Collins", top_k=2)
    assert hits and hits[0]["title"] == "The Hunger Games" and 0 <= hits[0]["semantic_similarity"] <= 1.0001


def test_embedding_dimension_mismatch_is_rejected(qdrant_store, monkeypatch):
    # simulate switching EMBEDDING_MODEL_NAME to a model whose output size differs from VECTOR_DIMENSION
    monkeypatch.setattr(qdrant_store.embedding_mgr, "embed_batch", lambda texts, batch_size=64: [[0.0] * 8 for _ in texts])
    with pytest.raises(ValueError, match="Embedding dimension"):
        qdrant_store.upsert_books([{"id": 1, "title": "t", "author": "a"}])


# ---------- ETL ----------

def test_etl_clips_to_column_limits_and_dedupes(tmp_path, monkeypatch):
    import etl
    csv = tmp_path / "Books.csv"
    long_author = "A" * 400
    csv.write_text(
        ",ISBN,book_id,Publication Year,Author,Title,AvgRating,Image-URL,Image-URL-S\n"
        f"0,1,1,2001,{long_author},Book One,4.0,http://x/1.jpg,http://x/1s.jpg\n"
        "1,2,1,2002,Dup,Dup Id,3.0,,\n"
        "2,3,2,,Someone,,3.0,,\n"
        "3,4,3,1999,Real Author,Book Three,9.9,,\n")
    monkeypatch.setattr(etl, "BOOKS_CSV", csv)
    books = etl.load_books_from_csv()
    assert [b["id"] for b in books] == [1, 3]               # duplicate id and blank title dropped
    assert len(books[0]["author"]) <= 255                    # Postgres VARCHAR(255) safe
    assert books[1]["rating"] == 5.0                         # clamped


def test_etl_requires_columns(tmp_path, monkeypatch):
    import etl
    csv = tmp_path / "Books.csv"
    csv.write_text("a,b\n1,2\n")
    monkeypatch.setattr(etl, "BOOKS_CSV", csv)
    with pytest.raises(ValueError, match="missing required columns"):
        etl.load_books_from_csv()


def test_etl_reset_needs_confirmation():
    import etl
    with pytest.raises(SystemExit):
        etl.main(["--reset"])


def test_etl_is_idempotent_and_non_destructive():
    import etl
    from core.database import SessionLocal, BookModel
    etl.run_etl(limit=50)
    etl.run_etl(limit=50)
    db = SessionLocal()
    try:
        assert db.query(BookModel).filter(BookModel.id <= 50).count() == 50
        extra = db.query(BookModel).count()
    finally:
        db.close()
    etl.run_etl(limit=10)             # a smaller run must not delete other rows
    db = SessionLocal()
    try:
        assert db.query(BookModel).count() == extra
    finally:
        db.close()


# ---------- Gateway orchestration (downstream services mocked) ----------

@pytest.fixture(scope="module")
def _gateway_module():
    # Loaded once: re-importing would re-register the gateway's Prometheus metrics.
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api-gateway", "main.py")
    spec = importlib.util.spec_from_file_location("gateway_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gateway(_gateway_module, monkeypatch):
    mod = _gateway_module
    monkeypatch.setattr(mod.cache, "get_json", lambda k: None)
    monkeypatch.setattr(mod.cache, "set_json", lambda k, v, ttl=None: None)
    monkeypatch.setattr(settings, "LLM_JOB_WAIT_SECONDS", 0.3)
    return mod


def _mock_downstream(monkeypatch, mod, retrieve):
    real = httpx.AsyncClient

    def handler(request):
        if request.url.path.endswith("/retrieve"):
            return retrieve(request)
        if request.url.path.endswith("/rank"):
            cand = _candidate().model_dump()
            return httpx.Response(200, json=[cand])
        return httpx.Response(404)
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), timeout=kw.get("timeout")))


def test_gateway_reports_empty_index_instead_of_silent_200(gateway, monkeypatch):
    from fastapi.testclient import TestClient
    _mock_downstream(monkeypatch, gateway, lambda r: httpx.Response(200, json=[]))
    res = TestClient(gateway.app).post(f"{settings.API_PREFIX}/recommend", json={"query": "dystopian"})
    data = res.json()
    assert res.status_code == 200 and data["total_candidates_retrieved"] == 0
    assert data["recommendations"] == [] and "empty" in data["message"].lower()


def test_gateway_reports_retrieval_outage(gateway, monkeypatch):
    from fastapi.testclient import TestClient
    _mock_downstream(monkeypatch, gateway, lambda r: httpx.Response(503))
    data = TestClient(gateway.app).post(f"{settings.API_PREFIX}/recommend", json={"query": "x"}).json()
    assert data["recommendations"] == [] and "unavailable" in data["message"].lower()


def test_gateway_falls_back_when_worker_is_down(gateway, monkeypatch):
    from fastapi.testclient import TestClient
    cand = _candidate().model_dump()
    cand["semantic_similarity"] = 0.8
    _mock_downstream(monkeypatch, gateway, lambda r: httpx.Response(200, json=[cand]))
    data = TestClient(gateway.app).post(f"{settings.API_PREFIX}/recommend", json={"query": "x", "top_k": 1}).json()
    assert data["llm_fallback_used"] is True and len(data["recommendations"]) == 1
    assert "unavailable" in data["recommendations"][0]["llm_explanation"].lower()


def test_gateway_catalog_ask_is_grounded(gateway, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(settings, "OPENAI_API_KEY", None)
    rec = {"id": 7, "title": "The Hunger Games", "author": "Suzanne Collins", "rating": 4.34, "publication_year": 2008}
    _mock_downstream(monkeypatch, gateway, lambda r: httpx.Response(200, json=[rec]))
    data = TestClient(gateway.app).post(f"{settings.API_PREFIX}/catalog/ask", json={"question": "books by Suzanne Collins"}).json()
    assert data["sources"][0]["book_id"] == 7 and "metadata" in data["answer"].lower()
