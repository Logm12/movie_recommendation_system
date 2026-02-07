import pytest
from starlette.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi import FastAPI
from middleware.error_handler import GlobalExceptionHandlerMiddleware

# Mock App
app = FastAPI()


# 1. Broken Endpoint
@app.get("/error")
def trigger_error():
    raise ValueError("Intentional Crash")


# 2. Add Middleware (The SUT - System Under Test)
app.add_middleware(GlobalExceptionHandlerMiddleware)

client = TestClient(app)


def test_global_exception_handler_catches_errors():
    """
    Test that the middleware catches unhandled exceptions
    and returns a structured 500 JSON response.
    """
    response = client.get("/error")

    # Assertions
    assert response.status_code == 500
    data = response.json()
    assert data["error"] == "Internal Server Error"
    assert "request_id" in data
    # We should NOT see the raw stack trace in production response (security)
    assert "Intentional Crash" not in data.values()
