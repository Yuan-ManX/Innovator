"""
Innovator Retry Mechanism

A unified, elegant retry system designed for Innovator Agent framework.

Features:
- Async-first retry decorator
- Exponential backoff with configurable limits
- Agent / Stage / Client aware logging
- Fully decoupled from business logic
- Safe retry boundaries with explicit exception control
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Callable, Type, TypeVar

logger = logging.getLogger("innovator.retry")

T = TypeVar("T")


# ============================================================
# Retry Configuration
# ============================================================

class RetryConfig:
    """Retry configuration used across Innovator Agent framework."""

    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: tuple[Type[Exception], ...] = (Exception,),
    ):
        self.enabled = enabled
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions

    def calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.initial_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


# ============================================================
# Exceptions
# ============================================================

class RetryExhaustedError(Exception):
    """Raised when retry attempts are exhausted."""

    def __init__(self, last_exception: Exception, attempts: int):
        self.last_exception = last_exception
        self.attempts = attempts
        super().__init__(
            f"Retry exhausted after {attempts} attempts. "
            f"Last error: {str(last_exception)}"
        )


# ============================================================
# Retry Decorator (Async)
# ============================================================

def async_retry(
    config: RetryConfig | None = None,
    *,
    name: str | None = None,
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Async retry decorator for Innovator components.

    Args:
        config: RetryConfig instance
        name: Optional logical name (Agent / Stage / Client name)
        on_retry: Optional callback on retry
    """

    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("async_retry can only be applied to async functions")

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            if not config.enabled:
                return await func(*args, **kwargs)

            last_exception: Exception | None = None
            label = name or func.__qualname__

            for attempt in range(config.max_retries + 1):
                try:
                    return await func(*args, **kwargs)

                except config.retryable_exceptions as e:
                    last_exception = e

                    # Stop retrying
                    if attempt >= config.max_retries:
                        logger.error(
                            "[Retry Exhausted] %s failed after %d attempts: %s",
                            label,
                            attempt + 1,
                            str(e),
                        )
                        raise RetryExhaustedError(e, attempt + 1) from e

                    delay = config.calculate_delay(attempt)

                    logger.warning(
                        "[Retry] %s attempt %d failed (%s), retrying in %.2fs",
                        label,
                        attempt + 1,
                        type(e).__name__,
                        delay,
                    )

                    if on_retry:
                        try:
                            on_retry(e, attempt + 1)
                        except Exception as callback_error:
                            logger.debug(
                                "Retry callback error ignored: %s",
                                callback_error,
                            )

                    await asyncio.sleep(delay)

            # Safety net (should never hit)
            assert last_exception is not None
            raise last_exception

        return wrapper

    return decorator


# ============================================================
# Utility Helpers
# ============================================================

def retry_from_config(config: Any, *, name: str | None = None) -> Callable:
    """
    Helper to create retry decorator directly from Innovator Config.

    Example:
        retry = retry_from_config(config.llm.retry, name="AnthropicClient")
    """
    return async_retry(
        config=RetryConfig(
            enabled=config.enabled,
            max_retries=config.max_retries,
            initial_delay=config.initial_delay,
            max_delay=config.max_delay,
            exponential_base=config.exponential_base,
        ),
        name=name,
    )
