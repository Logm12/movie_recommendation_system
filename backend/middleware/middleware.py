"""
Request ID Middleware for VDT GraphRec Pro

Adds a unique request ID to each incoming request for tracing.
The ID is:
- Generated if not present in X-Request-ID header
- Stored in context variable for use throughout the request
- Added to response headers

Following @observability-engineer principles for request tracing.
"""

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from logger import set_request_id, get_request_id, logger
from features import get_user_experiment_group


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Get from header or generate new
        request_id = request.headers.get("X-Request-ID")
        rid = set_request_id(request_id)

        # Log the incoming request
        logger.info(
            f"{request.method} {request.url.path}",
            method=request.method,
            path=str(request.url.path),
            client=request.client.host if request.client else "unknown",
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = rid

        # Log the response
        logger.info(
            f"Response {response.status_code}",
            status_code=response.status_code,
        )

        return response


class ABTestingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to assign users to A/B testing groups.

    Logic:
    1. Check for logged-in user (X-User-ID header).
    2. If logged in, hash user_id to get persistent group.
    3. If guest, use session ID or random bucket (stored in cookie/header).
    4. Attach `ab_group` to request scope and response headers.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # 1. Identify User
        # In a real app, this would come from JWT/Session.
        # Here we trust the internal X-User-ID header or fallback to Guest.
        user_id = request.headers.get("X-User-ID")

        if user_id:
            # Persistent group for logged-in users
            group = get_user_experiment_group(user_id)
        else:
            # Guests: Ideally verify persistent cookie, but for now simple assignment
            # Implementation detail: For guests, we might want to use IP or session
            # For this MVP, we treat guests as "control" unless specified
            group = "control"

        # 2. Attach to Request Scope (for endpoints to use)
        request.scope["ab_group"] = group

        # 3. Process Request
        response = await call_next(request)

        # 4. Add to Response Headers (for frontend debugging)
        response.headers["X-AB-Group"] = group

        # Log A/B assignment for analytics
        if user_id:
            with logger.contextualize(ab_group=group):
                logger.info(f"User {user_id} assigned to {group}")

        return response
