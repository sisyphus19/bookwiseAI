import os
import sys
import json
import time
import asyncio
from typing import Dict, Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import settings
from shared.logging import get_logger
from shared.schemas import RankedCandidate, ScoreBreakdown
from core.llm_provider import llm_service

logger = get_logger("llm-worker")

# Redis Connection with Memory Queue Fallback
try:
    import redis
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()
    REDIS_CONNECTED = True
    logger.info(f"LLM Worker connected to Redis: {settings.REDIS_URL}")
except Exception as e:
    REDIS_CONNECTED = False
    logger.warning(f"Redis connection failed ({e}). Worker running with local queue fallback.")
    local_queue = []
    local_results = {}

class LLMQueueProducer:
    @staticmethod
    def enqueue_job(job_data: Dict[str, Any]) -> str:
        job_id = job_data.get("job_id")
        payload = json.dumps(job_data)
        
        if REDIS_CONNECTED:
            redis_client.rpush(settings.REDIS_QUEUE_NAME, payload)
        else:
            local_queue.append(payload)
        return job_id

    @staticmethod
    def get_result(job_id: str) -> Optional[Dict[str, Any]]:
        if REDIS_CONNECTED:
            res = redis_client.get(f"job_result:{job_id}")
            if res:
                return json.loads(res)
        else:
            return local_results.get(job_id)
        return None

def _store_result(job_id: str, payload: Dict[str, Any]) -> None:
    if REDIS_CONNECTED:
        redis_client.setex(f"job_result:{job_id}", settings.CACHE_DEFAULT_TTL, json.dumps(payload))
    else:
        local_results[job_id] = payload


async def process_next_job():
    raw_payload = None
    if REDIS_CONNECTED:
        # Pop from Redis queue
        data = redis_client.blpop(settings.REDIS_QUEUE_NAME, timeout=1)
        if data:
            _, raw_payload = data
    else:
        if local_queue:
            raw_payload = local_queue.pop(0)

    if not raw_payload:
        return False

    job_id = None
    try:
        job_data = json.loads(raw_payload)
        job_id = job_data["job_id"]
        query = job_data["query"]
        raw_candidates = job_data["candidates"]
        genres = job_data.get("genres")
        mood = job_data.get("mood")

        ranked_candidates = []
        for c in raw_candidates:
            sb = c["score_breakdown"]
            breakdown = ScoreBreakdown(**sb)
            ranked_candidates.append(RankedCandidate(
                id=c["id"],
                title=c["title"],
                author=c["author"],
                genres=c["genres"],
                description=c["description"],
                rating=c["rating"],
                page_count=c["page_count"],
                publication_year=c["publication_year"],
                score_breakdown=breakdown,
                explainability_badges=c.get("explainability_badges", [])
            ))

        explained, fallback_used = await llm_service.get_explained_recommendations(
            query=query,
            ranked_candidates=ranked_candidates,
            genres=genres,
            mood=mood
        )

        result_payload = {
            "status": "completed",
            "job_id": job_id,
            "explained_recommendations": [e.model_dump(mode="json") for e in explained],
            "fallback_used": fallback_used,
            "provider": llm_service.provider_name,
        }
        _store_result(job_id, result_payload)
        logger.info(f"Processed LLM job {job_id} via {llm_service.provider_name} (fallback_used={fallback_used}).")
        return True

    except Exception as e:
        logger.exception(f"Error processing LLM worker job {job_id}: {e}")
        if job_id:  # publish the failure so the gateway can fall back immediately instead of timing out
            _store_result(job_id, {"status": "failed", "job_id": job_id, "error": str(e)})
        return True  # a job WAS consumed; returning False would make the loop sleep as if the queue were empty

async def start_worker_loop():
    logger.info("LLM Worker daemon starting consumer loop...")
    while True:
        processed = await process_next_job()
        if not processed:
            await asyncio.sleep(0.5)

if __name__ == "__main__":
    asyncio.run(start_worker_loop())
