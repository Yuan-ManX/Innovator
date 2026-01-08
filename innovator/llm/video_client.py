"""
Video Render Client Adapter
Supports: Sora, Runway, Pika
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ==============================
# Base class
# ==============================
class VideoRenderClientBase(ABC):
    """Abstract base class for video rendering clients."""

    def __init__(self, api_key: str, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @abstractmethod
    async def render(
        self,
        prompt: str,
        style: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        extra_params: dict | None = None,
    ) -> dict:
        """Generate a video from text prompt.

        Returns a dict containing at least:
        {
            "video_url": str,
            "meta": dict
        }
        """
        pass


# ==============================
# Sora Adapter
# ==============================
class SoraClient(VideoRenderClientBase):
    """Adapter for Sora video generation API."""

    def __init__(self, api_key: str, api_base: str = ""):
        super().__init__(api_key, api_base)
        import httpx
        self.client = httpx.AsyncClient(base_url=self.api_base, timeout=180)

    async def render(
        self,
        prompt: str,
        style: str | None = None,
        duration: int | None = 5,
        resolution: str | None = "720p",
        extra_params: dict | None = None,
    ) -> dict:
        payload = {
            "prompt": prompt,
            "style": style,
            "duration": duration,
            "resolution": resolution,
        }
        if extra_params:
            payload.update(extra_params)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post("/v1/generate", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# ==============================
# Runway Adapter
# ==============================
class RunwayClient(VideoRenderClientBase):
    """Adapter for Runway ML video generation."""

    def __init__(self, api_key: str, api_base: str = ""):
        super().__init__(api_key, api_base)
        import httpx
        self.client = httpx.AsyncClient(base_url=self.api_base, timeout=300)

    async def render(
        self,
        prompt: str,
        style: str | None = None,
        duration: int | None = 5,
        resolution: str | None = "720p",
        extra_params: dict | None = None,
    ) -> dict:
        payload = {
            "text_prompt": prompt,
            "video_style": style,
            "length_seconds": duration,
            "resolution": resolution,
        }
        if extra_params:
            payload.update(extra_params)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post("/videos/generate", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# ==============================
# Pika Adapter
# ==============================
class PikaClient(VideoRenderClientBase):
    """Adapter for Pika video generation."""

    def __init__(self, api_key: str, api_base: str = ""):
        super().__init__(api_key, api_base)
        import httpx
        self.client = httpx.AsyncClient(base_url=self.api_base, timeout=180)

    async def render(
        self,
        prompt: str,
        style: str | None = None,
        duration: int | None = 5,
        resolution: str | None = "720p",
        extra_params: dict | None = None,
    ) -> dict:
        payload = {
            "prompt": prompt,
            "style": style,
            "duration": duration,
            "resolution": resolution,
        }
        if extra_params:
            payload.update(extra_params)

        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self.client.post("/render/video", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


# ==============================
# Factory
# ==============================
class VideoRenderClientFactory:
    @staticmethod
    def from_config(provider: str, api_key: str, api_base: str | None = None) -> VideoRenderClientBase:
        provider = provider.lower()
        if provider == "sora":
            return SoraClient(api_key, api_base or "")
        elif provider == "runway":
            return RunwayClient(api_key, api_base or "")
        elif provider == "pika":
            return PikaClient(api_key, api_base or "")
        else:
            raise ValueError(f"Unsupported video render provider: {provider}")
