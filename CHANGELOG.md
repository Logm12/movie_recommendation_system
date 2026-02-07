# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0] - 2026-01-30

### Added
- **Neural Search Engine**: Integrated `sentence-transformers` (SBERT) for semantic movie search.
- **Natural Language Query**: Users can now search for movies using descriptive text (e.g., "sci-fi movies about time travel").
- **Robustness**: Implemented `GlobalExceptionHandlerMiddleware` for standardized error responses.
- **Health Checks**: Added detailed `/health` endpoint checking connectivity to PostgreSQL and Qdrant.
- **Discovery Modal**: Updated "Guest Mode" interface to support text-based search in addition to genre selection.

### Changed
- **Architecture**: Updated architecture to include `ContentSearchService` alongside `RecommendationService`.
- **E2E Testing**: Updated Playwright tests to verify the new Neural Search flow.
- **Dependencies**: Added `sentence-transformers` and updated `playwright` dependencies.

### Fixed
- **Port Conflicts**: Resolved issues with frontend docker container using stale code by running local updated instance.
- **Error Handling**: Fixed 500 Internal Server Error handling to return structured 5xx responses.

## [1.0.0] - 2026-01-13

### Added
- Initial release of VDT GraphRec Pro using LightGCN and Qdrant.
- Personalized recommendations for 610 users.
- Guest mode with genre-based recommendations.
- Docker Compose deployment.
