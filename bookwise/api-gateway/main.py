import asyncio
import os
import sys
import time
import uuid

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import cache
from core.catalog_qa import _source_record, answer_catalog_question
from llm_worker.worker import LLMQueueProducer
from shared.config import settings
from shared.logging import get_logger
from shared.schemas import (
    CatalogAskRequest,
    CatalogAskResponse,
    HealthCheckResponse,
    RecommendationRequest,
    RecommendationResponse,
)

logger = get_logger("api-gateway")

app = FastAPI(
    title="BookWise Recommendation API Gateway",
    description="Distributed API Gateway orchestrating microservices with end-to-end request tracing and resilient fallbacks.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GATEWAY_REQUESTS = Counter("gateway_requests_total", "Gateway requests count", ["method", "endpoint", "status"])
GATEWAY_LATENCY = Histogram("gateway_request_duration_seconds", "Gateway latency in seconds", ["endpoint"])

@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response: Response = await call_next(request)
    latency_ms = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id

    GATEWAY_REQUESTS.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    GATEWAY_LATENCY.labels(endpoint=request.url.path).observe(latency_ms / 1000.0)

    logger.info(f"Gateway {request.method} {request.url.path} -> {response.status_code}", extra={"request_id": request_id, "latency_ms": latency_ms})
    return response

@app.get("/health")
def health():
    return HealthCheckResponse(
        status="healthy",
        service="api-gateway",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        dependencies={"downstream_services": "ready"}
    )

@app.get("/health/live")
def live():
    return HealthCheckResponse(status="alive", service="api-gateway", timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"))

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

FALLBACK_EXPLANATION = (
    "The AI explanation service is unavailable right now, so this pick is based on how closely it matches "
    "your search and its average rating."
)


def _empty_response(req, request_id, start_time, retrieved, message):
    return RecommendationResponse(
        query=req.query, request_id=request_id, recommendations=[],
        total_candidates_retrieved=retrieved,
        execution_time_ms=round((time.time() - start_time) * 1000, 2),
        llm_fallback_used=False, message=message,
    )


def _fallback_recommendations(ranked_candidates, top_k):
    """Rule-based recommendations (no LLM) used when the worker is down, slow, or reports a failure."""
    recs = []
    for c in ranked_candidates[:top_k]:
        sb = c.get("score_breakdown") or {}
        recs.append({
            "book": c,
            "final_score": sb.get("final_score", 0.0),
            "score_breakdown": sb,
            "reasoning_bullets": c.get("explainability_badges") or ["✓ Matches your search"],
            "llm_explanation": FALLBACK_EXPLANATION,
        })
    return recs


@app.post(f"{settings.API_PREFIX}/recommend", response_model=RecommendationResponse)
async def orchestrate_recommendation(req: RecommendationRequest, request: Request):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    headers = {"X-Request-ID": request_id}

    cache_key = cache.make_key(
        "recommend", query=req.query, genres=req.favorite_genres, mood=req.preferred_mood,
        max_pages=req.max_page_count, top_k=req.top_k, provider=settings.LLM_PROVIDER,
    )
    cached = cache.get_json(cache_key)
    if cached:
        cached.update(request_id=request_id, execution_time_ms=round((time.time() - start_time) * 1000, 2))
        return RecommendationResponse(**cached)

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Retrieval. A failed call and an empty index are different problems, so report them differently.
        retrieval_url = f"{settings.RETRIEVAL_SERVICE_URL}{settings.API_PREFIX}/retrieve"
        try:
            r_res = await client.post(retrieval_url, json={"query": req.query, "top_k": 20}, headers=headers)
            r_res.raise_for_status()
            candidates = r_res.json()
        except Exception as e:
            logger.warning(f"Retrieval Service call failed ({e}).")
            return _empty_response(req, request_id, start_time, 0,
                                   "The retrieval service is unavailable right now. Please try again shortly.")

        if not candidates:
            return _empty_response(req, request_id, start_time, 0,
                                   "The book index is empty, so there is nothing to recommend yet. "
                                   "Run the ETL (python etl.py) to load books.")

        # 2. Ranking
        rank_url = f"{settings.RECOMMENDATION_SERVICE_URL}{settings.API_PREFIX}/rank"
        try:
            rk_res = await client.post(
                rank_url,
                json={
                    "candidates": candidates,
                    "favorite_genres": req.favorite_genres,
                    "max_page_count": req.max_page_count,
                    "top_k": req.top_k,
                },
                headers=headers,
            )
            rk_res.raise_for_status()
            ranked_candidates = rk_res.json()
        except Exception as e:
            logger.warning(f"Ranking Service call failed ({e}).")
            return _empty_response(req, request_id, start_time, len(candidates),
                                   "The ranking service is unavailable right now. Please try again shortly.")

    # 3. Enqueue the LLM explanation job and wait for the worker (bounded by LLM_JOB_WAIT_SECONDS)
    job_id = f"job-{request_id[:8]}-{uuid.uuid4().hex[:6]}"
    LLMQueueProducer.enqueue_job({
        "job_id": job_id,
        "query": req.query,
        "candidates": ranked_candidates,
        "genres": req.favorite_genres,
        "mood": req.preferred_mood,
    })

    result = None
    deadline = time.time() + settings.LLM_JOB_WAIT_SECONDS
    while time.time() < deadline:
        result = LLMQueueProducer.get_result(job_id)
        if result:
            break
        await asyncio.sleep(0.1)

    execution_time_ms = round((time.time() - start_time) * 1000, 2)

    if result and result.get("status") == "completed":
        response = RecommendationResponse(
            query=req.query,
            request_id=request_id,
            recommendations=result["explained_recommendations"],
            total_candidates_retrieved=len(candidates),
            execution_time_ms=execution_time_ms,
            llm_fallback_used=result.get("fallback_used", False),
        )
        if not response.llm_fallback_used:  # never cache degraded answers
            cache.set_json(cache_key, response.model_dump(mode="json"))
        return response

    reason = result.get("error") if result else "worker did not respond in time"
    logger.warning(f"LLM job {job_id} not completed ({reason}); serving rule-based fallback.")
    return RecommendationResponse(
        query=req.query,
        request_id=request_id,
        recommendations=_fallback_recommendations(ranked_candidates, req.top_k),
        total_candidates_retrieved=len(candidates),
        execution_time_ms=execution_time_ms,
        llm_fallback_used=True,
        message="AI explanations are unavailable; showing ranked results with basic reasoning.",
    )


@app.post(f"{settings.API_PREFIX}/catalog/ask", response_model=CatalogAskResponse)
async def catalog_ask(req: CatalogAskRequest):
    """Metadata-grounded Q&A over the catalog (title/author/rating/year only; no book text)."""
    retrieval_url = f"{settings.RETRIEVAL_SERVICE_URL}{settings.API_PREFIX}/retrieve"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post(retrieval_url, json={"query": req.question, "top_k": req.top_k})
            res.raise_for_status()
            retrieved = res.json()
    except Exception as e:
        logger.warning(f"Catalog ask: retrieval failed ({e}).")
        return CatalogAskResponse(question=req.question, sources=[],
                                  answer="The retrieval service is unavailable right now. Please try again shortly.")

    answer = await answer_catalog_question(req.question, retrieved)
    return CatalogAskResponse(question=req.question, answer=answer, sources=[_source_record(b) for b in retrieved])


@app.get("/stats")
async def stats():
    """Quick operational view: how many books are indexed and which LLM provider is configured."""
    points = None
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            res = await client.get(f"{settings.RETRIEVAL_SERVICE_URL}/health")
            points = res.json().get("points_count")
    except Exception as e:
        logger.warning(f"Stats: retrieval health check failed ({e}).")
    return {"indexed_books": points, "llm_provider": settings.LLM_PROVIDER, "environment": settings.ENVIRONMENT}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
