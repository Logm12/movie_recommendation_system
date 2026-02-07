# VDT GraphRec Pro: Architecture Analysis & Engineering Improvements

**Date:** January 30, 2026  
**Prepared For:** Viettel Digital Talent (AI/Data Science Track)  
**Context:** Project Review, CV Enhancement, and Code Refactoring Proposal  

---

## 1. Executive Summary

The **VDT GraphRec Pro** project demonstrates a solid understanding of modern full-stack development and recommendation system concepts. The architecture correctly separates concerns between the Frontend (React), Backend (FastAPI), Database (PostgreSQL), and Vector Search Engine (Qdrant).

However, to meet the standards of a "Major AI/Data Science" project for Viettel Digital Talent, several critical improvements are required. Most notably, the current **AI training loop is non-functional** (using a placeholder loss), and the backend lacks production-grade observability and testing.

This document outlines the architectural improvement plan, focusing on **Modularity**, **Testability**, and **Debuggability**, leveraging advanced engineering skills.

---

## 2. Architecture Analysis

### 2.1 Current Architecture ("As-Is")

The project follows a **Layered Monolith** pattern for the backend, communicating with external services.

```mermaid
graph TD
    User[User / Browser] -->|HTTP| FE[Frontend (React/Vite)]
    FE -->|HTTP| BE[Backend (FastAPI)]
    
    subgraph Data Layer
        BE -->|SQL| PG[(PostgreSQL)]
        BE -->|gRPC/HTTP| Q[Qdrant Vector DB]
    end
    
    subgraph AI Engine
        Train[Train Script] -->|Read| Data[MovieLens Data]
        Train -->|Write| Model[User Embeddings (.pt)]
        Train -->|Upsert| Q
    end
    
    Model -.->|Load| BE
```

**Strengths:**
*   **Clear Separation of Concerns**: Repositories are separated from Services.
*   **Modern Stack**: Use of FastAPI, Pydantic, and Qdrant is industry-standard.
*   **Containerization**: Docker Compose is correctly utilized.

**Weaknesses:**
*   **Dependency Injection**: The `get_service` implementation in `main.py` and `services` module uses a global singleton pattern which hampers testing validation.
*   **Observability**: Reliance on `print()` statements makes debugging in production (Docker) difficult.
*   **AI Validity**: The LightGCN model training is effectively a stub.

### 2.2 Proposed Architecture ("To-Be")

To improve modularity, we propose enforcing **Dependency Injection (DI)** and introducing a **Structured Logging** layer.

```mermaid
graph TD
    subgraph Backend Core
        Config[Configuration]
        Logger[Structured Logger]
    end
    
    subgraph Service Layer (DI)
        Repo[Repositories]
        Svc[Recommendation Service]
    end
    
    API[API Endpoints] -->|Inject| Svc
    Svc -->|Inject| Repo
    Svc -->|Inject| Logger
```

---

## 3. Critical Findings: AI Engine

> [!WARNING]
> **CRITICAL ISSUE**: The training script `ai_engine/train.py` contains a placeholder loss function.

```python
# Current Implementation (ai_engine/train.py)
loss = torch.tensor(0.0, requires_grad=True) # Placeholder
```

**Impact:** The model **does not learn**. The embeddings generated are random initializations. This invalidates the "AI" component of the project.

**Recommendation:**
Implement **BPR (Bayesian Personalized Ranking) Loss**.
1.  Create a `BPRLoss` class.
2.  Implement efficient **Negative Sampling** (for every positive interaction $(u, i)$, sample a negative item $j$ that $u$ didn't interact with).
3.  Optimize: $L = -\sum \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda \|\Theta\|^2$.

---

## 4. Engineering Improvements

### 4.1 Modularity: Dependency Injection Refactoring

The current codebase uses global variables for services. Refactor to use FastAPI's dependency injection system more effectively.

**Before:**
```python
# services/recommendation_service.py
_service = None
def get_service():
    global _service
    if not _service: _service = RecommendationService()
    return _service
```

**After (Proposed):**
```python
# backend/dependencies.py
from functools import lru_cache
from fastapi import Depends
from services.recommendation_service import RecommendationService
from repositories.movie_repository import MovieRepository

@lru_cache()
def get_movie_repo() -> MovieRepository:
    return MovieRepository()

@lru_cache()
def get_recommendation_service(
    movie_repo: MovieRepository = Depends(get_movie_repo)
) -> RecommendationService:
    return RecommendationService(movie_repo=movie_repo)
```

This allows you to easily **mock** `MovieRepository` when testing `RecommendationService`.

### 4.2 Testability: Unit vs. Integration

Currently, tests in `backend/tests/test_api.py` are integration tests that require a running DB/Model.

**Goal:** Achieve >80% code coverage.

**Plan:**
1.  **Add `pytest-mock`**: capable of mocking Qdrant and Postgres calls.
2.  **Unit Test `RecommendationService`**:
    *   Mock `vector_repo.search_similar` to return fixed dummy vectors.
    *   Verify logic for "Cold Start" genre ranking without needing a DB.
3.  **Refactor `test_service.py`**:
    ```python
    def test_cold_start_ranking(mock_repo):
        service = RecommendationService(repo=mock_repo)
        # ... assert service correctly ranks Action movies higher
    ```

### 4.3 Debuggability: Structured Logging & Correlation IDs

"Debug Auto" skill implies the system should help you debug. `print()` statements are insufficient.

**Recommendation:**
Replace `print` with `loguru` or `structlog`.

```python
# backend/logger.py
import sys
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO")
logger.add("logs/app.log", rotation="500 MB", serialize=True) # JSON logs
```

**Usage:**
```python
logger.info("Cold start request received", genres=request.genres, request_id=ctx.request_id)
```
This produces JSON logs that can be ingested by ELK/Loki, showing **exactly** what happened with structured data fields.

---

## 5. Skills Application Checklist

This proposal aligns with the requested `.agent/skills` framework:

*   **Architecture Master**:
    *   *Simplicity*: Kept the stack simple (FastAPI/React) but refined the wiring (DI).
    *   *Trade-offs*: Acknowledged that a full Microservices architecture is overkill, so "Modular Monolith" is chosen.
*   **Debug Auto**:
    *   Implemented structured logging to enable automated log analysis.
*   **Code Quality**:
    *   Enforced Pydantic for validation.
    *   Recommended `ruff` or `black` for formatting.

---

## 6. Roadmap for "Viettel Digital Talent"

To present this as a top-tier project:

1.  **Fix `train.py` immediately**. A working model is non-negotiable for an AI track.
2.  **Refactor Backend**: Apply the DI pattern.
3.  **Add 5-10 Unit Tests**: Show you understand "Testing Pyramid".
4.  **Add Logging**: Show you understand "Production Readiness".
5.  **Documentation Update**: Update `README.md` to reflect the "Advanced Architecture".

This approach transforms the project from a "Student Demo" to a "Junior Engineer POC".
