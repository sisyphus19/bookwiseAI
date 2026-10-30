import time
import uuid
import httpx
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import RecommendationRequest, RecommendationResponse, HealthCheckResponse
from llm_worker.worker import LLMQueueProducer

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

@app.post(f"{settings.API_PREFIX}/recommend", response_model=RecommendationResponse)
async def orchestrate_recommendation(req: RecommendationRequest, request: Request):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    headers = {"X-Request-ID": request_id}

    async with httpx.AsyncClient(timeout=10.0) as client:
        # 1. Call Retrieval Service
        retrieval_url = f"{settings.RETRIEVAL_SERVICE_URL}{settings.API_PREFIX}/retrieve"
        try:
            r_res = await client.post(retrieval_url, json={"query": req.query, "top_k": 20}, headers=headers)
            candidates = r_res.json() if r_res.status_code == 200 else []
        except Exception as e:
            logger.warning(f"Retrieval Service call failed ({e}). Using internal fallback.")
            candidates = []

        # 2. Call Recommendation/Ranking Service
        rank_url = f"{settings.RECOMMENDATION_SERVICE_URL}{settings.API_PREFIX}/rank"
        try:
            rk_res = await client.post(
                rank_url,
                json={
                    "candidates": candidates,
                    "favorite_genres": req.favorite_genres,
                    "max_page_count": req.max_page_count,
                    "top_k": req.top_k
                },
                headers=headers
            )
            ranked_candidates = rk_res.json() if rk_res.status_code == 200 else []
        except Exception as e:
            logger.warning(f"Ranking Service call failed ({e}).")
            ranked_candidates = []

    # 3. Enqueue LLM Explanation Job to Redis Queue Worker
    job_id = f"job-{request_id[:8]}"
    LLMQueueProducer.enqueue_job({
        "job_id": job_id,
        "query": req.query,
        "candidates": ranked_candidates,
        "genres": req.favorite_genres,
        "mood": req.preferred_mood
    })

    # Poll or get job result with fast timeout (with instant local fallback)
    result = None
    for _ in range(10):
        result = LLMQueueProducer.get_result(job_id)
        if result:
            break
        await asyncio.sleep(0.1)

    execution_time_ms = (time.time() - start_time) * 1000

    if result:
        return RecommendationResponse(
            query=req.query,
            request_id=request_id,
            recommendations=result["explained_recommendations"],
            total_candidates_retrieved=len(candidates),
            execution_time_ms=round(execution_time_ms, 2),
            llm_fallback_used=result.get("fallback_used", False)
        )

    # Fallback if queue worker is delayed
    return RecommendationResponse(
        query=req.query,
        request_id=request_id,
        recommendations=[
            {
                "book": c,
                "final_score": c.get("score_breakdown", {}).get("final_score", 0.8),
                "score_breakdown": c.get("score_breakdown", {"semantic_score":0.8,"genre_affinity":0.8,"popularity_prior":0.8,"recency_factor":0.8,"final_score":0.8}),
                "reasoning_bullets": c.get("explainability_badges", ["✓ Recommended book"]),
                "llm_explanation": f"Recommended for: '{req.query}'"
            }
            for c in ranked_candidates[:req.top_k]
        ],
        total_candidates_retrieved=len(candidates),
        execution_time_ms=round(execution_time_ms, 2),
        llm_fallback_used=True
    )

import asyncio
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
