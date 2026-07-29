import json
import httpx
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from shared.config import settings
from shared.schemas import RankedCandidate, ExplainedRecommendation, BookResponse
from shared.logging import get_logger

logger = get_logger("llm-provider")

PROMPT_TEMPLATE = """
System: You are BookWise AI, an expert literary recommendation assistant.
User Query: "{query}"
User Preferences: Genres: {genres}, Preferred Mood: {mood}

Candidate Books:
{books_summary}

Task:
Analyze the user's intent and explain why each book is recommended.
Provide a concise, engaging explanation paragraph and 3 bullet points starting with '✓' highlighting why this book matches.

Return JSON in format:
{
  "explanations": [
    {
      "book_id": <id>,
      "reasoning_bullets": ["✓ ...", "✓ ...", "✓ ..."],
      "explanation": "..."
    }
  ]
}
"""

class BaseLLMProvider(ABC):
    @abstractmethod
    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: List[RankedCandidate],
        genres: Optional[List[str]] = None,
        mood: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        pass


class LocalMockProvider(BaseLLMProvider):
    """
    High-quality offline / zero-cost explanation engine.
    Generates rich, contextual reasoning bullet points and personalized explanations.
    """
    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: List[RankedCandidate],
        genres: Optional[List[str]] = None,
        mood: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        results = []
        for item in ranked_candidates:
            bullets = list(item.explainability_badges)
            if not bullets:
                bullets = [
                    f"✓ Matches search intent for '{query}'",
                    f"✓ Highly rated at {item.rating:.1f} stars",
                    f"✓ Explores {', '.join(item.genres[:2]) if item.genres else 'captivating'} themes"
                ]

            genre_str = f" in {', '.join(item.genres[:2])}" if item.genres else ""
            explanation = (
                f"'{item.title}' by {item.author} is a compelling selection for your request '{query}'."
                f" It features rich narrative depth{genre_str} and balances engaging character dynamics"
                f" with a well-paced {item.page_count}-page structure."
            )

            results.append({
                "book_id": item.id,
                "reasoning_bullets": bullets[:4],
                "explanation": explanation
            })
        return results


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, api_key: str, model: str = settings.OPENAI_MODEL):
        self.api_key = api_key
        self.model = model
        self.mock = LocalMockProvider()

    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: List[RankedCandidate],
        genres: Optional[List[str]] = None,
        mood: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.info("OpenAI API key missing. Using local mock generator.")
            return await self.mock.generate_explanations(query, ranked_candidates, genres, mood)

        books_summary = "\n".join([
            f"ID: {b.id} | Title: '{b.title}' by {b.author} | Genres: {', '.join(b.genres)} | Pages: {b.page_count} | Rating: {b.rating} | Description: {b.description[:150]}..."
            for b in ranked_candidates
        ])

        prompt = PROMPT_TEMPLATE.format(
            query=query,
            genres=", ".join(genres) if genres else "None specified",
            mood=mood if mood else "Any",
            books_summary=books_summary
        )

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful JSON-only literary assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        "response_format": {"type": "json_object"}
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    parsed = json.loads(content)
                    return parsed.get("explanations", [])
                else:
                    logger.warning(f"OpenAI error status {response.status_code}. Using fallback.")
        except Exception as e:
            logger.warning(f"OpenAI request failed ({e}). Falling back to local mock.")

        return await self.mock.generate_explanations(query, ranked_candidates, genres, mood)


class OllamaProvider(BaseLLMProvider):
    def __init__(self, base_url: str = settings.OLLAMA_BASE_URL, model: str = settings.OLLAMA_MODEL):
        self.base_url = base_url
        self.model = model
        self.mock = LocalMockProvider()

    async def generate_explanations(
        self,
        query: str,
        ranked_candidates: List[RankedCandidate],
        genres: Optional[List[str]] = None,
        mood: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        books_summary = "\n".join([
            f"ID: {b.id} | Title: '{b.title}' by {b.author} | Description: {b.description[:100]}..."
            for b in ranked_candidates
        ])
        prompt = PROMPT_TEMPLATE.format(
            query=query,
            genres=", ".join(genres) if genres else "None",
            mood=mood if mood else "Any",
            books_summary=books_summary
        )

        try:
            async with httpx.AsyncClient(timeout=settings.LLM_TIMEOUT_SECONDS) as client:
                res = await client.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "format": "json", "stream": False}
                )
                if res.status_code == 200:
                    parsed = json.loads(res.json().get("response", "{}"))
                    return parsed.get("explanations", [])
        except Exception as e:
            logger.warning(f"Ollama call failed ({e}). Falling back to local mock.")

        return await self.mock.generate_explanations(query, ranked_candidates, genres, mood)


class LLMReasoningService:
    """Manager class instantiating appropriate provider and executing explanation pipeline with fallbacks."""
    def __init__(self):
        provider_type = settings.LLM_PROVIDER.lower()
        if provider_type == "openai" and settings.OPENAI_API_KEY:
            self.provider = OpenAIProvider(api_key=settings.OPENAI_API_KEY)
        elif provider_type == "ollama":
            self.provider = OllamaProvider()
        else:
            self.provider = LocalMockProvider()

        self.fallback_provider = LocalMockProvider()

    async def get_explained_recommendations(
        self,
        query: str,
        ranked_candidates: List[RankedCandidate],
        genres: Optional[List[str]] = None,
        mood: Optional[str] = None
    ) -> tuple[List[ExplainedRecommendation], bool]:
        fallback_used = False
        try:
            raw_explanations = await self.provider.generate_explanations(query, ranked_candidates, genres, mood)
            expl_map = {item["book_id"]: item for item in raw_explanations if "book_id" in item}
        except Exception as e:
            logger.error(f"LLM Provider execution exception ({e}). Activating fallback generator.")
            fallback_used = True
            raw_explanations = await self.fallback_provider.generate_explanations(query, ranked_candidates, genres, mood)
            expl_map = {item["book_id"]: item for item in raw_explanations if "book_id" in item}

        explained_results = []
        for cand in ranked_candidates:
            expl_data = expl_map.get(cand.id)
            if not expl_data:
                fallback_used = True
                fallback_items = await self.fallback_provider.generate_explanations(query, [cand], genres, mood)
                expl_data = fallback_items[0]

            book_resp = BookResponse(
                id=cand.id,
                title=cand.title,
                author=cand.author,
                genres=cand.genres,
                description=cand.description,
                rating=cand.rating,
                page_count=cand.page_count,
                publication_year=cand.publication_year
            )

            explained_results.append(ExplainedRecommendation(
                book=book_resp,
                final_score=cand.score_breakdown.final_score,
                score_breakdown=cand.score_breakdown,
                reasoning_bullets=expl_data.get("reasoning_bullets", cand.explainability_badges),
                llm_explanation=expl_data.get("explanation", f"Recommended for query: '{query}'")
            ))

        return explained_results, fallback_used

llm_service = LLMReasoningService()
