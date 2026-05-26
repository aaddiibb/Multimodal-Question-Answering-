"""Qwen/DashScope client utilities for the MAMMQA pipeline.

This file uses the OpenAI Python SDK only as an OpenAI-compatible HTTP client.
Requests go to DashScope/Qwen because the client is created with a DashScope
base_url and a DASHSCOPE_API_KEY. It does not use an OpenAI API key.
"""

from __future__ import annotations

import base64
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from openai import OpenAI
try:
    from openai import APIConnectionError, APIStatusError, AuthenticationError, BadRequestError, RateLimitError
except Exception:  # pragma: no cover - older SDK fallback
    APIConnectionError = APIStatusError = AuthenticationError = BadRequestError = RateLimitError = Exception


QWEN_REGION_BASE_URLS: Dict[str, str] = {
    "beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "china": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "virginia": "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "international": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "singapore": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}

DEFAULT_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "qwen-plus")
DEFAULT_VISION_MODEL = os.getenv("QWEN_VISION_MODEL", "qwen3-vl-plus")
DEFAULT_REGION = os.getenv("QWEN_REGION", "intl")
DEFAULT_BASE_URL = os.getenv("QWEN_BASE_URL") or QWEN_REGION_BASE_URLS.get(DEFAULT_REGION.lower(), QWEN_REGION_BASE_URLS["intl"])


def get_dashscope_api_key(secret_name: str = "DASHSCOPE_API_KEY") -> Optional[str]:
    """Return DASHSCOPE_API_KEY from env or Kaggle Secrets when available."""
    key = os.getenv(secret_name)
    if key:
        return key
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore

        return UserSecretsClient().get_secret(secret_name)
    except Exception:
        return None


def make_qwen_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    region: Optional[str] = None,
) -> OpenAI:
    """Create an OpenAI-compatible client pointed at DashScope/Qwen."""
    key = api_key or get_dashscope_api_key()
    if not key or key in {"sk-YOUR-KEY-HERE", "YOUR_DASHSCOPE_API_KEY"}:
        raise RuntimeError(
            "DASHSCOPE_API_KEY is missing. Add it as an environment variable or Kaggle Secret named DASHSCOPE_API_KEY."
        )
    if base_url is None:
        if region:
            base_url = QWEN_REGION_BASE_URLS.get(region.lower())
            if base_url is None:
                raise ValueError(f"Unknown Qwen region {region!r}. Choose one of {sorted(QWEN_REGION_BASE_URLS)} or pass base_url.")
        else:
            base_url = DEFAULT_BASE_URL
    return OpenAI(api_key=key, base_url=base_url)


def is_data_url(value: str) -> bool:
    return isinstance(value, str) and value.startswith("data:")


def is_http_url(value: str) -> bool:
    return isinstance(value, str) and (value.startswith("http://") or value.startswith("https://"))


def encode_image_file_to_data_url(path: str | Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if not mime:
        suffix = path.suffix.lower().lstrip(".") or "jpeg"
        mime = f"image/{'jpeg' if suffix in {'jpg', 'jpeg'} else suffix}"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def image_to_message_part(image: str | Path | Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a URL/data-url/local-path/dict to OpenAI-compatible image_url part."""
    if not image:
        return None
    if isinstance(image, dict):
        if image.get("type") == "image_url":
            return image
        url = image.get("url") or image.get("image_url") or image.get("path")
        if isinstance(url, dict):
            url = url.get("url")
        if not url:
            return None
        image = url
    s = str(image)
    # Ignore synthetic placeholders like "[Image: title]".
    if s.startswith("[Image:"):
        return None
    if is_http_url(s) or is_data_url(s):
        return {"type": "image_url", "image_url": {"url": s}}
    p = Path(s)
    if p.exists() and p.is_file():
        return {"type": "image_url", "image_url": {"url": encode_image_file_to_data_url(p)}}
    return None


def images_to_message_parts(images: Optional[Sequence[str | Path | Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if not images:
        return []
    parts: List[Dict[str, Any]] = []
    for image in images:
        part = image_to_message_part(image)
        if part is not None:
            parts.append(part)
    return parts


def make_content(text: str, images: Optional[Sequence[str | Path | Dict[str, Any]]] = None) -> str | List[Dict[str, Any]]:
    """Return string content for text-only calls, or multimodal content if images exist."""
    img_parts = images_to_message_parts(images)
    if not img_parts:
        return text
    return [{"type": "text", "text": text}] + img_parts


def _error_text(exc: BaseException) -> str:
    pieces = [str(exc)]
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            pieces.append(response.text)
        except Exception:
            pass
    body = getattr(exc, "body", None)
    if body:
        pieces.append(str(body))
    return "\n".join(pieces)


def _is_fatal_error(exc: BaseException) -> bool:
    msg = _error_text(exc).lower()
    fatal_markers = [
        "invalid api key",
        "invalid_api_key",
        "authentication",
        "unauthorized",
        "permission",
        "insufficient_quota",
        "quota",
        "billing",
        "access denied",
        "model not found",
        "does not exist",
        "not supported",
    ]
    return any(marker in msg for marker in fatal_markers) and "rate" not in msg


def extract_response_text(response: Any) -> str:
    """Extract assistant text from a Chat Completions response or stream aggregate."""
    try:
        content = response.choices[0].message.content
        if isinstance(content, list):
            return "".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in content)
        return content or ""
    except Exception:
        return str(response)


def call_qwen(
    client: OpenAI,
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_TEXT_MODEL,
    temperature: float = 0.2,
    top_p: float = 0.8,
    max_tokens: int = 700,
    max_retries: int = 5,
    timeout: Optional[float] = None,
    verbose: bool = False,
    **extra: Any,
) -> str:
    """Call Qwen through DashScope OpenAI-compatible chat completions."""
    for attempt in range(max_retries + 1):
        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            if timeout is not None:
                kwargs["timeout"] = timeout
            kwargs.update(extra)
            response = client.chat.completions.create(**kwargs)
            return extract_response_text(response)
        except Exception as exc:  # noqa: BLE001 - preserve rich API exceptions from compatible services
            msg = _error_text(exc)
            if _is_fatal_error(exc):
                raise RuntimeError(
                    "Qwen/DashScope returned a non-retryable error. Check DASHSCOPE_API_KEY, QWEN_BASE_URL/region, "
                    f"model name, quota, and account permissions. Original error:\n{msg}"
                ) from exc
            if attempt >= max_retries:
                raise RuntimeError(f"Qwen call failed after {max_retries} retries. Last error:\n{msg}") from exc
            wait = min(90.0, (2 ** attempt) + random.uniform(0.0, 2.0))
            if verbose:
                print(f"Qwen temporary error; retry {attempt + 1}/{max_retries} after {wait:.1f}s")
            time.sleep(wait)
    raise AssertionError("unreachable")
