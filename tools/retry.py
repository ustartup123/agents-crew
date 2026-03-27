"""
tools/retry.py — Retry decorator with exponential backoff for API calls.

Usage:
    @retry(max_retries=3, base_delay=1.0, retryable_exceptions=(SlackApiError,))
    def my_api_call():
        ...
"""

import functools
import logging
import time
from typing import Sequence, Type

logger = logging.getLogger(__name__)


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: Sequence[Type[Exception]] = (Exception,),
):
    """Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts (not counting the first call).
        base_delay: Initial delay in seconds between retries.
        max_delay: Maximum delay cap in seconds.
        retryable_exceptions: Tuple of exception types that trigger a retry.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except tuple(retryable_exceptions) as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}: {e}"
                        )
            raise last_exception
        return wrapper
    return decorator
