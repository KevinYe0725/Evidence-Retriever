import hashlib
import time
from typing import List

import httpx

from app.config import settings


class EmbeddingError(RuntimeError):
    pass


class QwenEmbedderClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: int = 30,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self._use_mock():
            return self._mock_embeddings(texts)
        try:
            payload = self._build_request_payload(texts)
            response_data = self._post_embeddings(payload)
            embeddings = self._parse_response(response_data)
        except NotImplementedError as exc:
            raise EmbeddingError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Qwen embedding HTTP error: {exc}") from exc
        except Exception as exc:
            raise EmbeddingError(f"Qwen embedding failed: {exc}") from exc

        if len(embeddings) != len(texts):
            raise EmbeddingError("Embedding response size mismatch")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embeddings = self.embed_texts([text])
        if not embeddings:
            raise EmbeddingError("Empty embedding response")
        return embeddings[0]

    def _use_mock(self) -> bool:
        return self.base_url.lower() in {"", "mock", "mock://"}

    def _build_request_payload(self, texts: List[str]) -> dict:
        if not self.model:
            raise EmbeddingError("Qwen embedding model is required")
        return {
            "model": self.model,
            "input": texts,
            "encoding_format": "float",
        }

    def _build_headers(self) -> dict:
        if not self.api_key:
            raise EmbeddingError("Qwen embedding API key is required")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _post_embeddings(self, payload: dict) -> dict:
        url = self._build_url()
        headers = self._build_headers()
        timeout = httpx.Timeout(self.timeout_seconds)
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=timeout) as client:
                    response = client.post(url, headers=headers, json=payload)
                    response.raise_for_status()
                    return response.json()
            except (httpx.TimeoutException, httpx.HTTPError) as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(self.retry_backoff_seconds * (2**attempt))

        if last_exc:
            raise last_exc
        raise EmbeddingError("Unknown Qwen embedding error")

    def _parse_response(self, response_data: dict) -> List[List[float]]:
        if not isinstance(response_data, dict):
            raise EmbeddingError("Invalid embedding response format")
        data = response_data.get("data", [])
        if not isinstance(data, list):
            raise EmbeddingError("Embedding response missing data list")
        try:
            sorted_items = sorted(data, key=lambda item: item.get("index", 0))
        except Exception as exc:
            raise EmbeddingError("Embedding response data invalid") from exc
        embeddings: List[List[float]] = []
        for item in sorted_items:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise EmbeddingError("Embedding item missing embedding vector")
            embeddings.append([float(x) for x in embedding])
        return embeddings

    def _build_url(self) -> str:
        if not self.base_url:
            raise EmbeddingError("Qwen embedding base URL is required")
        base = self.base_url.rstrip("/")
        if base.endswith("/embeddings"):
            return base
        return f"{base}/embeddings"

    def _mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        dim = 128
        embeddings: List[List[float]] = []
        for text in texts:
            digest = hashlib.md5(text.encode("utf-8")).digest()
            values = [b / 255.0 for b in digest]
            vec = (values * (dim // len(values) + 1))[:dim]
            embeddings.append(vec)
        return embeddings


_default_embedder = QwenEmbedderClient(
    base_url=settings.QWEN_EMBEDDING_BASE_URL,
    api_key=settings.QWEN_EMBEDDING_API_KEY,
    model=settings.QWEN_EMBEDDING_MODEL,
    timeout_seconds=settings.QWEN_EMBEDDING_TIMEOUT_SECONDS,
)


def get_default_embedder() -> QwenEmbedderClient:
    return _default_embedder
