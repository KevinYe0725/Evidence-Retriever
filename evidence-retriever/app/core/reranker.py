import time
from typing import List

import httpx


class RerankError(RuntimeError):
    pass


class QwenRerankerClient:
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

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        payload = self._build_request_payload(query, documents)
        try:
            response_data = self._post_rerank(payload)
            scores = self._parse_response(response_data, len(documents))
        except httpx.HTTPError as exc:
            raise RerankError(f"Qwen rerank HTTP error: {exc}") from exc
        except Exception as exc:
            raise RerankError(f"Qwen rerank failed: {exc}") from exc

        if len(scores) != len(documents):
            raise RerankError("Rerank response size mismatch")
        return scores

    def _build_request_payload(self, query: str, documents: List[str]) -> dict:
        if not self.model:
            raise RerankError("Qwen rerank model is required")
        truncated = self._truncate_documents(documents)
        return {
            "model": self.model,
            "query": query,
            "documents": truncated,
        }

    def _truncate_documents(self, documents: List[str], max_chars: int = 2000) -> List[str]:
        # Conservative truncation to avoid exceeding provider limits; adjust per docs.
        return [doc[:max_chars] for doc in documents]

    def _build_headers(self) -> dict:
        if not self.api_key:
            raise RerankError("Qwen rerank API key is required")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_url(self) -> str:
        if not self.base_url:
            raise RerankError("Qwen rerank base URL is required")
        base = self.base_url.rstrip("/")
        if base.endswith("/rerank"):
            return base
        return f"{base}/rerank"

    def _post_rerank(self, payload: dict) -> dict:
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
        raise RerankError("Unknown Qwen rerank error")

    def _parse_response(self, response_data: dict, expected_len: int) -> List[float]:
        if not isinstance(response_data, dict):
            raise RerankError("Invalid rerank response format")
        results = response_data.get("results")
        if not isinstance(results, list):
            raise RerankError("Rerank response missing results list")

        scores: List[float | None] = [None] * expected_len
        for item in results:
            if not isinstance(item, dict):
                raise RerankError("Invalid rerank result item")
            index = item.get("index")
            relevance_score = item.get("relevance_score")
            if not isinstance(index, int):
                raise RerankError("Rerank result index invalid")
            if index < 0 or index >= expected_len:
                raise RerankError("Rerank result index out of range")
            try:
                scores[index] = float(relevance_score)
            except (TypeError, ValueError) as exc:
                raise RerankError("Rerank result score invalid") from exc

        if any(score is None for score in scores):
            raise RerankError("Rerank response missing scores")
        return [score for score in scores if score is not None]
