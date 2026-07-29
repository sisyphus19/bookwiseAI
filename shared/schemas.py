from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

# --- Auth & User Schemas ---

class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="bookworm42")
    email: EmailStr = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6, example="SecurePass123!")

class UserLoginRequest(BaseModel):
    username: str = Field(..., example="bookworm42")
    password: str = Field(..., example="SecurePass123!")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str

class UserPreferencesUpdate(BaseModel):
    favorite_genres: List[str] = Field(default_factory=list, example=["Sci-Fi", "Philosophy"])
    preferred_book_length: Optional[str] = Field(None, example="medium")  # short (<300), medium (300-500), long (>500)
    preferred_mood: Optional[str] = Field(None, example="thought-provoking")

class UserProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    favorite_genres: List[str]
    preferred_book_length: Optional[str]
    preferred_mood: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# --- Book Schemas ---

class BookBase(BaseModel):
    title: str
    author: str
    genres: List[str]
    description: str
    rating: float = Field(..., ge=0.0, le=5.0)
    page_count: int
    publication_year: int
    cover_image_url: Optional[str] = None

class BookResponse(BookBase):
    id: int
    reviews_summary: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- Retrieval Schemas ---

class RetrievalRequest(BaseModel):
    query: str = Field(..., example="philosophical sci-fi books like Dune")
    top_k: int = Field(default=20, ge=1, le=50)
    genres: Optional[List[str]] = None
    max_page_count: Optional[int] = None

class CandidateBook(BaseModel):
    id: int
    title: str
    author: str
    genres: List[str]
    description: str
    rating: float
    page_count: int
    publication_year: int
    semantic_similarity: float

# --- Reranking & Recommendation Schemas ---

class ScoreBreakdown(BaseModel):
    semantic_score: float
    genre_affinity: float
    popularity_prior: float
    recency_factor: float
    final_score: float

class RankedCandidate(BaseModel):
    id: int
    title: str
    author: str
    genres: List[str]
    description: str
    rating: float
    page_count: int
    publication_year: int
    score_breakdown: ScoreBreakdown
    explainability_badges: List[str] = Field(default_factory=list)

class RecommendationRequest(BaseModel):
    query: str = Field(..., example="Suggest fantasy books with political intrigue but less violence")
    user_id: Optional[int] = None
    favorite_genres: Optional[List[str]] = None
    preferred_mood: Optional[str] = None
    max_page_count: Optional[int] = None
    top_k: int = Field(default=5, ge=1, le=20)

class ExplainedRecommendation(BaseModel):
    book: BookResponse
    final_score: float
    score_breakdown: ScoreBreakdown
    reasoning_bullets: List[str]  # e.g., ["✓ shares philosophical sci-fi themes", "✓ medium-length novel (~280 pages)"]
    llm_explanation: str

class RecommendationResponse(BaseModel):
    query: str
    request_id: str
    recommendations: List[ExplainedRecommendation]
    total_candidates_retrieved: int
    execution_time_ms: float
    llm_fallback_used: bool = False

# --- Feedback Schema ---

class FeedbackRequest(BaseModel):
    recommendation_id: Optional[str] = None
    book_id: int
    rating: int = Field(..., ge=1, le=5, example=5)  # 1 to 5 stars or thumbs up/down mapped
    is_helpful: bool = True
    comment: Optional[str] = None

class FeedbackResponse(BaseModel):
    status: str = "success"
    message: str = "Feedback recorded successfully"
    feedback_id: int

# --- Health & Metrics Schemas ---

class HealthCheckResponse(BaseModel):
    status: str = "healthy"
    service: str
    timestamp: datetime
    version: str = "1.0.0"
    dependencies: Dict[str, str] = Field(default_factory=dict)
