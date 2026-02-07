# 🎬 VDT GraphRec Pro

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![React](https://img.shields.io/badge/React-18.2-61DAFB.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A Hybrid Movie Recommendation System powered by Graph Neural Networks and Vector Search**

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [API Reference](#api-reference) • [Tech Stack](#tech-stack)

</div>

---

## 📖 Overview

VDT GraphRec Pro is a production-ready movie recommendation system that combines **LightGCN** (Light Graph Convolutional Network) for collaborative filtering with **Qdrant** vector database for real-time similarity search. The system supports both personalized recommendations for known users and cold-start recommendations for guest users.

### 🎯 Key Highlights

- **LightGCN Model**: State-of-the-art graph neural network for learning user-item embeddings
- **Neural Search**: Natural language movie discovery using Sentence-BERT
- **Cold-Start Support**: Guest users can get recommendations based on genre preferences
- **Robustness**: Enterprise-grade health checks and error handling
- **Real Movie Posters**: Integration with TMDB API for authentic movie artwork
- **Sub-100ms Latency**: Optimized for real-time recommendation serving
- **Full Docker Deployment**: One command to run the entire stack

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Personalized Recommendations** | Graph-based collaborative filtering for 610 users |
| 🧠 **Neural Search** | Semantic search understanding natural language queries |
| 🆕 **Guest Mode (Cold-Start)** | Recommendations based on genre/keyword preferences |
| 🖼️ **Real Movie Posters** | TMDB integration for authentic movie artwork |
| ⚡ **High Performance** | <100ms API response time |
| 🎨 **Modern UI** | Netflix-inspired dark theme with smooth animations |
| 🐳 **Containerized** | Full Docker Compose deployment |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        VDT GraphRec Pro                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Browser   │────▶│   React +   │────▶│    Nginx    │      │
│   │             │◀────│   Mantine   │◀────│   :3000     │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              FastAPI Backend (:8000)                     │  │
│   │  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐   │  │
│   │  │ config  │  │  models  │  │       services       │   │  │
│   │  └─────────┘  └──────────┘  │ RecommendationService│   │  │
│   │  ┌─────────────────────────────────────────────────┐   │  │
│   │  │              repositories                        │   │  │
│   │  │   MovieRepository    │    VectorRepository       │   │  │
│   │  └─────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                    │                        │                   │
│                    ▼                        ▼                   │
│   │  PostgreSQL (:5432) │    │      Qdrant (:6333)         │   │
│   │  - movies table     │    │  - 9,742 movie vectors      │   │
│   │  - poster_url       │    │  - 384 dim (SBERT)          │   │
│   └─────────────────────┘    └─────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/Logm12/movie_recommendation_graphrec.git
cd movie_recommendation_graphrec

# Start all services
docker-compose up -d --build

# Wait for services to initialize (about 30 seconds)
# Then open http://localhost:3000 in your browser
```

### Verify Installation

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","embeddings_loaded":true,"known_users":610}
```

---

## 📚 API Reference

### Health Check

```http
GET /
GET /health
```

### Get Recommendations (Known User)

```http
GET /recommend/{user_id}?top_k=10
```

**Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `user_id` | int | User ID (1-610) |
| `top_k` | int | Number of recommendations (default: 10) |

**Response:**
```json
{
  "user_id": 1,
  "recommendations": [
    {
      "id": 3430,
      "title": "Death Wish (1974)",
      "genres": "Action|Crime|Drama",
      "poster_url": "https://image.tmdb.org/t/p/w500/...",
      "score": 0.43
    }
  ]
}
```

### Cold-Start Recommendations (Guest)

```http
POST /recommend/cold_start
Content-Type: application/json

{
  "genres": ["Action", "Sci-Fi"],
  "keywords": ["space"],
  "selected_movie_ids": [],
  "top_k": 10
}
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **AI Model** | LightGCN (PyTorch) + SBERT | Graph Collaborative Filtering + Semantic Search |
| **Vector DB** | Qdrant | Fast similarity search with HNSW |
| **Backend** | FastAPI | High-performance async API |
| **Frontend** | React + Vite + Mantine | Modern responsive UI |
| **Database** | PostgreSQL | Movie metadata storage |
| **Cache** | Redis | Caching for high-speed performance |
| **Container** | Docker Compose | Multi-service orchestration |
| **Animation** | Framer Motion | Smooth UI transitions |

---

## 📁 Project Structure

```
movie_recommendation_graphrec/
├── ai_engine/                 # Model training scripts
│   ├── model.py               # LightGCN implementation
│   ├── train.py               # Training script
│   ├── ingest_data.py         # Data ingestion
│   └── enrich_posters.py      # TMDB poster fetcher
├── backend/                   # FastAPI application
│   ├── config.py              # Configuration
│   ├── main.py                # API endpoints
│   ├── models/                # Pydantic schemas
│   ├── repositories/          # Data access layer
│   └── services/              # Business logic
├── frontend/                  # React application
│   ├── src/
│   │   ├── App.tsx            # Main component
│   │   ├── hooks/             # Custom React hooks
│   │   ├── services/          # API client
│   │   └── types/             # TypeScript types
│   └── Dockerfile
├── data/                      # MovieLens dataset
├── docker-compose.yml         # Service orchestration
├── PROJECT_REPORT.md          # Technical documentation
└── README.md                  # This file
```

---

## 🔬 Model Details

### LightGCN Algorithm

LightGCN simplifies graph convolution for collaborative filtering by removing feature transformation and non-linear activation:

```
e_u^(k+1) = Σ (1/√|N_u|√|N_i|) × e_i^(k)
```

**Key advantages:**
- Captures high-order connectivity patterns
- Lightweight and efficient training
- State-of-the-art performance on MovieLens

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Embedding Dimension | 64 |
| Number of Layers | 3 |
| Learning Rate | 0.001 |
| Batch Size | 1024 |
| Epochs | 100 |

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| API Latency (p50) | 45ms |
| API Latency (p99) | 98ms |
| Recall@20 | 0.142 |
| NDCG@20 | 0.094 |
| Users Supported | 610 |
| Movies in Database | 9,742 |

---

## 🖼️ Screenshots

### Main Dashboard
*Netflix-inspired dark theme with movie recommendations*

### Guest Mode
*Cold-start recommendations based on genre preferences*

---

## 🔧 Development

### Local Development (without Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Run Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) for the dataset
- [LightGCN Paper](https://arxiv.org/abs/2002.02126) for the algorithm
- [TMDB](https://www.themoviedb.org/) for movie posters
- [Qdrant](https://qdrant.tech/) for vector database

---

<div align="center">

**Built with ❤️ for Viettel Digital Talent Program**

[⬆ Back to top](#-vdt-graphrec-pro)

</div>
