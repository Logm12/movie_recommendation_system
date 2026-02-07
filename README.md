# VDT GraphRec Pro

## Overview

VDT GraphRec Pro is a hybrid movie recommendation system that integrates Collaborative Filtering using LightGCN (Light Graph Convolutional Network) with Content-Based Retrieval using Vector Search (Qdrant). The system addresses the cold-start problem by allowing guest users to receive recommendations based on genre and keyword preferences, while providing personalized ranking for registered users based on their interaction history.

## Architecture

The system follows a microservices-based architecture, containerized using Docker.

```
┌─────────────────────────────────────────────────────────────────┐
│                        VDT GraphRec Pro                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│   │   Browser   │────▶│ React Client│────▶│    Nginx    │       │
│   │             │◀────│             │◀────│   (Proxy)   │       │
│   └─────────────┘     └─────────────┘     └─────────────┘       │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                FastAPI Backend Service                  │   │
│   │  ┌─────────┐  ┌──────────┐  ┌──────────────────────┐    │   │
│   │  │ config  │  │  models  │  │       services       │    │   │
│   │  └─────────┘  └──────────┘  │ RecommendationService│    │   │
│   │  ┌─────────────────────────────────────────────────┐    │   │
│   │  │              repositories                       │    │   │
│   │  │   MovieRepository    │    VectorRepository      │    │   │
│   │  └─────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                    │                        │                   │
│                    ▼                        ▼                   │
│   │      PostgreSQL         │    │        Qdrant           │    │
│   │   (Relational Data)     │    │    (Vector Data)        │    │
│   └─────────────────────┘    └─────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

*   **Frontend**: React application utilizing Vite for build tooling and Mantine UI for component styling.
*   **Backend**: FastAPI service implementing the recommendation logic, data retrieval, and API endpoints.
*   **Database**: PostgreSQL for storing structured movie metadata and user information.
*   **Vector Database**: Qdrant for storing and searching high-dimensional embeddings of movies and users.
*   **AI Engine**: Offline training pipeline using PyTorch to train the LightGCN model and generation of SBERT embeddings.

## Key Features

*   **Graph-Based Collaborative Filtering**: Utilizes LightGCN to model high-order connectivity between users and items, capturing collaborative signals effectively.
*   **Semantic Search**: Implements Sentence-BERT to encode movie metadata, enabling natural language search and content-based retrieval.
*   **Cold-Start Mitigation**: Provides a "Guest Mode" that generates recommendations by aggregating vector embeddings of selected genres and keywords.
*   **Performance Optimization**: Designed for low-latency inference, with sub-100ms response times for recommendation endpoints.
*   **Containerization**: Full system definition in `docker-compose.yml` for reproducible deployments.

## Quick Start

### Prerequisites

*   Docker Engine 20.10+
*   Docker Compose V2+

### Deployment

1.  Clone the repository:
    ```bash
    git clone https://github.com/Logm12/movie_recommendation_graphrec.git
    cd movie_recommendation_graphrec
    ```

2.  Start the services:
    ```bash
    docker-compose up -d --build
    ```

3.  Access the application:
    *   Frontend: `http://localhost:3000`
    *   Backend API Docs: `http://localhost:8000/docs`

### Health Check

Verify the system status using the health endpoint:

```bash
curl http://localhost:8000/health
```

Expected output:
```json
{"status":"healthy","embeddings_loaded":true,"known_users":610}
```

## API Reference

### Recommendations (Authenticated User)

**Endpoint**: `GET /recommend/{user_id}`

Retrieves personalized movie recommendations for a specific user ID based on the trained LightGCN model.

**Parameters**:
*   `user_id` (integer): The unique identifier of the user (1-610).
*   `top_k` (integer, optional): Number of recommendations to return. Default: 10.

### Cold-Start Recommendations (Guest)

**Endpoint**: `POST /recommend/cold_start`

Generates recommendations based on explicit user preferences (genres, keywords) using vector arithmetic in the embedding space.

**Payload**:
```json
{
  "genres": ["Action", "Sci-Fi"],
  "keywords": ["space travel", "future"],
  "selected_movie_ids": [],
  "top_k": 10
}
```

## Technology Stack

*   **Language**: Python 3.10+, TypeScript
*   **Frameworks**: FastAPI, React
*   **Data Stores**: PostgreSQL, Qdrant, Redis
*   **Machine Learning**: PyTorch, Sentence-Transformers
*   **Infrastructure**: Docker, Nginx

## Project Structure

```
.
├── ai_engine/          # Model training and data processing scripts
├── backend/            # FastAPI application source code
├── frontend/           # React application source code
├── data/               # Raw dataset storage
├── docker-compose.yml  # Service orchestration configuration
├── PROJECT_REPORT.md   # Detailed technical report
└── README.md           # This file
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
