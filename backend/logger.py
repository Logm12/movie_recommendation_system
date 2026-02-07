"""
Structured Logging Configuration for VDT GraphRec Pro

Uses loguru for:
- JSON-formatted logs (machine-readable)
- Beautiful console output (human-readable)
- Correlation IDs for request tracing
- Automatic log rotation

Following @observability-engineer and @kaizen principles:
- Simple, effective observability
- No over-engineering
"""

import sys
import uuid
from contextvars import ContextVar
from typing import Optional

from loguru import logger

# Context variable to store the current request ID
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def get_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_id_ctx.get()


def set_request_id(request_id: Optional[str] = None) -> str:
    """Set a new request ID in context. Generates one if not provided."""
    rid = request_id or str(uuid.uuid4())[:8]
    request_id_ctx.set(rid)
    return rid


def request_id_filter(record):
    """Add request_id to every log record."""
    record["extra"]["request_id"] = get_request_id() or "N/A"
    return True


def configure_logging(log_level: str = "INFO", json_logs: bool = False):
    """
    Configure logging with structured output.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: If True, output JSON. If False, colored console output.
    """
    # Remove default handler
    logger.remove()

    # Format for console (human-readable)
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>[{extra[request_id]}]</cyan> | "
        "<level>{message}</level>"
    )

    # Format for JSON (machine-readable)
    json_format = (
        '{{"timestamp":"{time:YYYY-MM-DDTHH:mm:ss.SSSZ}",'
        '"level":"{level}",'
        '"request_id":"{extra[request_id]}",'
        '"message":"{message}",'
        '"module":"{module}",'
        '"function":"{function}",'
        '"line":{line}}}'
    )

    if json_logs:
        logger.add(
            sys.stdout,
            format=json_format,
            level=log_level,
            filter=request_id_filter,
            serialize=False,  # We're using our own JSON format
        )
    else:
        logger.add(
            sys.stdout,
            format=console_format,
            level=log_level,
            filter=request_id_filter,
            colorize=True,
        )

    logger.info("Logging configured", log_level=log_level, json_mode=json_logs)


# Export commonly used functions
__all__ = [
    "logger",
    "configure_logging",
    "get_request_id",
    "set_request_id",
    "request_id_ctx",
]
