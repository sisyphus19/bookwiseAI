import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from shared.schemas import ScoreBreakdown, RankedCandidate, CandidateBook
from shared.logging import get_logger

logger = get_logger("recommendation-engine")

class CandidateFilter:
    """Filters retrieved candidate books based on user hard boundaries and safety constraints."""
    def filter_candidates(
        self,
        candidates: List[Dict[str, Any]],
        max_page_count: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        filtered = []
        for cand in candidates:
            if max_page_count and cand.get("page_count", 0) > max_page_count:
                continue
            filtered.append(cand)
        return filtered if filtered else candidates  # Fallback to candidates if all filtered out


class FeatureExtractor:
    """Extracts features for hybrid scoring: semantic similarity, genre affinity, popularity prior, and recency factor."""
    def compute_genre_affinity(self, book_genres: List[str], preferred_genres: Optional[List[str]]) -> float:
        if not preferred_genres or not book_genres:
            return 0.5  # Neutral default for cold start users
        
        book_set = set(g.lower() for g in book_genres)
        pref_set = set(g.lower() for g in preferred_genres)
        
        intersection = book_set.intersection(pref_set)
        if not intersection:
            return 0.1
        return len(intersection) / max(len(book_set), len(pref_set))

    def compute_popularity_prior(self, rating: float) -> float:
        # Scale 0..5 rating to 0..1 range
        return max(0.0, min(1.0, rating / 5.0))

    def compute_recency_factor(self, publication_year: int) -> float:
        current_year = datetime.now().year
        age = max(0, current_year - publication_year)
        # Exponential decay half-life ~10 years
        return math.exp(-0.05 * age)


class ScoringEngine:
    """
    Computes hybrid composite score combining:
    FinalScore = 0.45 * Semantic + 0.25 * GenreAffinity + 0.15 * Popularity + 0.15 * Recency
    """
    def __init__(self):
        self.w_semantic = 0.60
        self.w_genre = 0.00
        self.w_popularity = 0.20
        self.w_recency = 0.20

    def score_candidate(
        self,
        candidate: Dict[str, Any],
        genre_affinity: float,
        popularity_prior: float,
        recency_factor: float
    ) -> ScoreBreakdown:
        semantic_sim = float(candidate.get("semantic_similarity", 0.5))
        
        final_score = (
            self.w_semantic * semantic_sim +
            self.w_genre * genre_affinity +
            self.w_popularity * popularity_prior +
            self.w_recency * recency_factor
        )

        return ScoreBreakdown(
            semantic_score=round(semantic_sim, 4),
            genre_affinity=round(genre_affinity, 4),
            popularity_prior=round(popularity_prior, 4),
            recency_factor=round(recency_factor, 4),
            final_score=round(final_score, 4)
        )


class Ranker:
    """Sorts candidate books by final composite score and generates rule-based explainability badges."""
    def generate_badges(self, candidate: Dict[str, Any], score_breakdown: ScoreBreakdown, preferred_genres: Optional[List[str]]) -> List[str]:
        badges = []
        
        if score_breakdown.semantic_score >= 0.70:
            badges.append("✓ Strong semantic match for query")
        elif score_breakdown.semantic_score >= 0.50:
            badges.append("✓ Relevant theme match")
            
        if preferred_genres and score_breakdown.genre_affinity > 0.3:
            matched = [g for g in candidate.get("genres", []) if any(p.lower() in g.lower() for p in preferred_genres)]
            if matched:
                badges.append(f"✓ Matches favorite genre: {matched[0]}")
                
        page_count = candidate.get("page_count", 300)
        if page_count < 300:
            badges.append(f"✓ Quick read ({page_count} pages)")
        elif page_count <= 500:
            badges.append(f"✓ Medium length ({page_count} pages)")
        else:
            badges.append(f"✓ Epic novel ({page_count} pages)")
            
        if score_breakdown.popularity_prior >= 0.85:
            badges.append(f"✓ Highly acclaimed ({candidate.get('rating', 4.5):.1f}★)")
            
        if candidate.get("publication_year", 2020) >= 2020:
            badges.append(f"✓ Modern release ({candidate.get('publication_year')})")

        return badges

    def rank(
        self,
        candidates_with_scores: List[tuple[Dict[str, Any], ScoreBreakdown]],
        preferred_genres: Optional[List[str]] = None,
        top_k: int = 5
    ) -> List[RankedCandidate]:
        ranked = []
        for cand, score in candidates_with_scores:
            badges = self.generate_badges(cand, score, preferred_genres)
            ranked.append(RankedCandidate(
                id=cand["id"],
                title=cand["title"],
                author=cand["author"],
                genres=cand.get("genres", []),
                description=cand.get("description", ""),
                rating=cand.get("rating", 4.0),
                page_count=cand.get("page_count", 300),
                publication_year=cand.get("publication_year", 2020),
                score_breakdown=score,
                explainability_badges=badges
            ))

        ranked.sort(key=lambda x: x.score_breakdown.final_score, reverse=True)
        return ranked[:top_k]


class RecommendationEngine:
    """
    Modular recommendation engine pipeline integrating:
    CandidateFilter -> FeatureExtractor -> ScoringEngine -> Ranker
    """
    def __init__(self):
        self.filter = CandidateFilter()
        self.feature_extractor = FeatureExtractor()
        self.scoring_engine = ScoringEngine()
        self.ranker = Ranker()

    def process_recommendations(
        self,
        retrieved_candidates: List[Dict[str, Any]],
        favorite_genres: Optional[List[str]] = None,
        max_page_count: Optional[int] = None,
        top_k: int = 5
    ) -> List[RankedCandidate]:
        # 1. Filter
        filtered_candidates = self.filter.filter_candidates(retrieved_candidates, max_page_count)
        
        # 2. Extract Features & Score
        scored_candidates = []
        for cand in filtered_candidates:
            affinity = self.feature_extractor.compute_genre_affinity(cand.get("genres", []), favorite_genres)
            popularity = self.feature_extractor.compute_popularity_prior(cand.get("rating", 4.0))
            recency = self.feature_extractor.compute_recency_factor(cand.get("publication_year", 2020))
            
            score_breakdown = self.scoring_engine.score_candidate(cand, affinity, popularity, recency)
            scored_candidates.append((cand, score_breakdown))
            
        # 3. Rank & Badge
        ranked_results = self.ranker.rank(scored_candidates, favorite_genres, top_k)
        return ranked_results

recommendation_engine = RecommendationEngine()
