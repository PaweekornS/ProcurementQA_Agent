# -*- coding: utf-8 -*-
"""
api/errors.py

Error responses that never leak internals. Exception details (connection strings, file paths,
stack traces) go to the server log under a short error_id; clients only get the id to quote.
"""

import logging
import uuid

from fastapi import HTTPException, status

logger = logging.getLogger("api.errors")


def internal_error(exc: BaseException, operation: str) -> HTTPException:
    """Log `exc` with a correlation id and return a generic 500 for the client."""
    error_id = uuid.uuid4().hex[:12]
    logger.error("%s failed [error_id=%s]", operation, error_id, exc_info=exc)
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"{operation} failed. Reference error_id={error_id} when reporting this issue.",
    )


def service_busy(exc: BaseException) -> HTTPException:
    """503 with Retry-After when every QA slot is taken (see QA_MAX_CONCURRENCY)."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=str(exc),
        headers={"Retry-After": "30"},
    )
