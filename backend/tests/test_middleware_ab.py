import pytest
from unittest.mock import Mock, patch
from starlette.types import Scope, Receive, Send
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.testclient import TestClient
from middleware import ABTestingMiddleware


@pytest.mark.asyncio
async def test_ab_middleware_unauthenticated():
    """Unauthenticated users should still get assigned a group (based on random/session)."""
    app = Starlette()
    app.add_middleware(ABTestingMiddleware)

    @app.route("/")
    async def homepage(request):
        return JSONResponse({"ab_group": request.scope.get("ab_group")})

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["ab_group"] in ["control", "treatment"]


@pytest.mark.asyncio
async def test_ab_middleware_authenticated():
    """Authenticated users (X-User-ID) should get deterministic group."""
    app = Starlette()
    app.add_middleware(ABTestingMiddleware)

    @app.route("/")
    async def homepage(request):
        return JSONResponse({"ab_group": request.scope.get("ab_group")})

    client = TestClient(app)

    # User 1 -> Deterministic Group
    response1 = client.get("/", headers={"X-User-ID": "1"})
    group1 = response1.json()["ab_group"]

    # Run again for User 1
    response2 = client.get("/", headers={"X-User-ID": "1"})
    group2 = response2.json()["ab_group"]

    assert group1 == group2
    assert group1 in ["control", "treatment"]
