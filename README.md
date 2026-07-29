# BookWise AI: Distributed AI Book Recommendation Platform (v4)

BookWise AI is a production-grade, distributed AI book recommendation platform built via a **capability-driven progressive implementation strategy**. It features multi-factor hybrid reranking, high-dimensional vector search via Qdrant, async LLM explainability workers via Redis queues, structured JSON request tracing (`request_id`), API versioning (`/api/v1/`), Helm charts, and GitOps deployments.

---

## 🌟 Key Architecture Highlights

1. **API Versioning**: All REST APIs operate under `/api/v1/` (`/api/v1/recommend`, `/api/v1/users/login`, `/api/v1/feedback`, `/api/v1/books`).
2. **Distributed Request Tracing**: End-to-end `X-Request-ID` propagation across API Gateway, Retrieval Service, Recommendation Engine, LLM Worker, and Analytics.
3. **Multi-Factor Hybrid Reranking**: Candidate Filter -> Feature Extractor -> Scoring Engine ($0.45 \cdot \text{Semantic} + 0.25 \cdot \text{Genre} + 0.15 \cdot \text{Popularity} + 0.15 \cdot \text{Recency}$) -> Ranker.
4. **Redis Multi-Role Architecture**: Single Redis instance providing dual functions: **Async LLM Task Queue** and **Distributed Recommendation/User Profile Cache**.
5. **Resilient LLM Explainability**: LLM Reasoning Provider with non-blocking fallback templates ensuring top recommendations are delivered in under 100ms even if LLM workers time out.
6. **Production Kubernetes & GitOps**: Full K8s manifests with resource limits, liveness & readiness probes, HPA rules, parameterized Helm charts, and ArgoCD application manifests.

---

## 🏗️ System Microservices Architecture

```
                       ┌─────────────────────────┐
                       │     React Dashboard     │
                       └────────────┬────────────┘
                                    │
                                    ▼ HTTP / REST
                       ┌─────────────────────────┐
                       │       API Gateway       │  <--- Generates & injects X-Request-ID
                       └────────────┬────────────┘
                                    │
    ┌───────────────────┬───────────┴───────────┬───────────────────┐
    │                   │                       │                   │
    ▼                   ▼                       ▼                   ▼
┌──────────────┐  ┌──────────────┐    ┌──────────────────┐  ┌──────────────────┐
│ User Service │  │ Book Service │    │Retrieval Service │  │Recommendation Svc│
└──────┬───────┘  └──────┬───────┘    └────────┬─────────┘  └────────┬─────────┘
       │                 │                     │                     │
       ▼                 ▼                     ▼                     ▼
 ┌───────────┐     ┌───────────┐         ┌───────────┐         ┌───────────┐
 │ PostgreSQL│     │ PostgreSQL│         │ Qdrant DB │         │Feature    │
 └───────────┘     └───────────┘         └───────────┘         │Extractors │
                                                               └─────┬─────┘
                                                                     │
                                                                     ▼
                                                               ┌───────────┐
                                                               │Redis Queue│
                                                               └─────┬─────┘
                                                                     │
                                                                     ▼
                                                               ┌───────────┐
                                                               │ LLM Worker│
                                                               └───────────┘
```

---

## 🚀 Quickstart Guide

### 1. Run Locally with SQLite Fallback & Monolith Mode
```bash
cd bookwise
pip install -r requirements.txt
python etl.py
python main.py
```
Access docs at `http://localhost:8000/docs`.

### 2. Run Full Microservice Stack via Docker Compose
```bash
cd bookwise
docker-compose up --build
```
- **API Gateway**: `http://localhost:8000`
- **Frontend Dashboard**: Open `frontend/index.html` in your browser.

---

## 🧪 Testing

Run pytest suite with automated data seeding:
```bash
pytest tests/test_phase1.py -v
```

---

## 📚 Documentation Suite

- 📐 **[Architecture](docs/Architecture.md)**: Deep dive into microservices dataflow, RAG vector retrieval, and Redis queue design.
- 💡 **[Design Decisions](docs/DesignDecisions.md)**: Hybrid scoring math, database choices, and versioning conventions.
- 🚀 **[Deployment](docs/Deployment.md)**: Kubernetes, Helm Chart parameters, and ArgoCD GitOps configuration.
- 📈 **[Scaling](docs/Scaling.md)**: Horizontal Pod Autoscaler (HPA) rules, Redis caching strategy, and Qdrant indexing.
- 🛡️ **[Failure Scenarios](docs/FailureScenarios.md)**: Resilient fallback mechanisms, timeout management, and circuit breakers.
