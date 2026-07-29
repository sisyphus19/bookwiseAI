import pytest
import os
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
from etl import run_etl
from shared.config import settings
from core.database import Base, engine

client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_test_data():
    """Resets the database schema and re-seeds data before running tests.
    Drop-all + create-all ensures idempotent runs against the SQLite fallback file."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    run_etl()


def test_health_check_endpoints():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "database" in data["dependencies"]

    live_res = client.get("/health/live")
    assert live_res.status_code == 200
    assert live_res.json()["status"] == "alive"

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text

def test_user_registration_and_login():
    # Register
    reg_payload = {
        "username": "testuser_rag",
        "email": "testuser@example.com",
        "password": "Password123!"
    }
    reg_res = client.post(f"{settings.API_PREFIX}/users/register", json=reg_payload)
    assert reg_res.status_code == 201
    user_data = reg_res.json()
    assert user_data["username"] == "testuser_rag"

    # Login
    login_payload = {
        "username": "testuser_rag",
        "password": "Password123!"
    }
    login_res = client.post(f"{settings.API_PREFIX}/users/login", json=login_payload)
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]

    # Get Profile with token
    headers = {"Authorization": f"Bearer {token}"}
    profile_res = client.get(f"{settings.API_PREFIX}/users/me", headers=headers)
    assert profile_res.status_code == 200
    assert profile_res.json()["email"] == "testuser@example.com"

    # Update Preferences
    pref_payload = {
        "favorite_genres": ["Sci-Fi", "Philosophy"],
        "preferred_book_length": "medium",
        "preferred_mood": "thought-provoking"
    }
    pref_res = client.put(f"{settings.API_PREFIX}/users/preferences", json=pref_payload, headers=headers)
    assert pref_res.status_code == 200
    assert "Philosophy" in pref_res.json()["favorite_genres"]

def test_books_catalog_endpoints():
    response = client.get(f"{settings.API_PREFIX}/books")
    assert response.status_code == 200
    books = response.json()
    assert len(books) > 0
    first_book_id = books[0]["id"]

    detail_res = client.get(f"{settings.API_PREFIX}/books/{first_book_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == first_book_id

def test_recommendation_pipeline_with_request_tracing():
    custom_request_id = "test-req-id-12345"
    payload = {
        "query": "Recommend philosophical sci-fi books like Dune",
        "favorite_genres": ["Sci-Fi", "Philosophy"],
        "top_k": 3
    }
    headers = {"X-Request-ID": custom_request_id}
    
    res = client.post(f"{settings.API_PREFIX}/recommend", json=payload, headers=headers)
    assert res.status_code == 200
    assert res.headers.get("X-Request-ID") == custom_request_id

    data = res.json()
    assert data["request_id"] == custom_request_id
    assert len(data["recommendations"]) == 3
    
    top_rec = data["recommendations"][0]
    assert "book" in top_rec
    assert "score_breakdown" in top_rec
    assert "reasoning_bullets" in top_rec
    assert "llm_explanation" in top_rec
    assert len(top_rec["reasoning_bullets"]) > 0
    assert any("✓" in bullet for bullet in top_rec["reasoning_bullets"])

def test_recommendation_feedback_loop():
    feedback_payload = {
        "book_id": 1,
        "rating": 5,
        "is_helpful": True,
        "comment": "Spot on recommendation for Dune!"
    }
    res = client.post(f"{settings.API_PREFIX}/feedback", json=feedback_payload)
    assert res.status_code == 201
    data = res.json()
    assert data["status"] == "success"
    assert "feedback_id" in data
