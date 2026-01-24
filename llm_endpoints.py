import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
from openai import OpenAI


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


@dataclass(frozen=True)
class LLMEndpoints:
    # vLLM OpenAI-compatible servers
    vllm_generation_base_url: str = os.getenv(
        "VLLM_GENERATION_BASE_URL", "http://localhost:8000/v1"
    )
    vllm_embeddings_base_url: str = os.getenv(
        "VLLM_EMBEDDINGS_BASE_URL",
        os.getenv("VLLM_GENERATION_BASE_URL", "http://localhost:8000/v1"),
    )
    vllm_api_key: str = os.getenv("VLLM_API_KEY", "local")

    # Models
    vllm_generation_model: str = os.getenv(
        "VLLM_GENERATION_MODEL", "Qwen/Qwen3-4B-Thinking-2507"
    )
    vllm_embeddings_model: str = os.getenv(
        "VLLM_EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # OpenAI judge/classifier
    openai_judge_model: str = os.getenv("OPENAI_JUDGE_MODEL", "gpt-5-mini")

    # Default generation params for Qwen/Qwen3-4B-Thinking-2507
    default_temperature: float = float(os.getenv("VLLM_TEMPERATURE", "0.6"))
    default_top_p: float = float(os.getenv("VLLM_TOP_P", "0.95"))
    default_top_k: int = int(os.getenv("VLLM_TOP_K", "20"))
    default_max_tokens: int = int(os.getenv("VLLM_MAX_TOKENS", "16384"))

    def generation_base_url(self) -> str:
        return _normalize_base_url(self.vllm_generation_base_url)

    def embeddings_base_url(self) -> str:
        return _normalize_base_url(self.vllm_embeddings_base_url)

    def generation_completions_url(self) -> str:
        return f"{self.generation_base_url()}/completions"

    def generation_chat_completions_url(self) -> str:
        return f"{self.generation_base_url()}/chat/completions"

    def embeddings_url(self) -> str:
        return f"{self.embeddings_base_url()}/embeddings"


ENDPOINTS = LLMEndpoints()


def openai_compat_client(*, base_url: str, api_key: Optional[str] = None) -> OpenAI:
    # The OpenAI SDK requires a non-empty api_key string even if the server ignores it.
    return OpenAI(
        base_url=_normalize_base_url(base_url),
        api_key=api_key or ENDPOINTS.vllm_api_key,
    )


class OpenAICompatEmbeddingModel:
    """SentenceTransformer-like wrapper backed by an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        default_batch_size: int = 256,
    ):
        self._client = openai_compat_client(base_url=base_url, api_key=api_key)
        self._model = model
        self._default_batch_size = default_batch_size

    def encode(
        self, texts: Iterable[str], batch_size: Optional[int] = None
    ) -> np.ndarray:
        items = list(texts)
        if not items:
            return np.zeros((0, 0), dtype=np.float32)

        bs = batch_size or self._default_batch_size
        out: List[List[float]] = []
        for i in range(0, len(items), bs):
            batch = items[i : i + bs]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            # Preserve input order.
            out.extend([d.embedding for d in resp.data])
        return np.asarray(out, dtype=np.float32)


def get_vllm_embedding_model(*, batch_size: int = 256) -> OpenAICompatEmbeddingModel:
    return OpenAICompatEmbeddingModel(
        base_url=ENDPOINTS.embeddings_base_url(),
        model=ENDPOINTS.vllm_embeddings_model,
        api_key=ENDPOINTS.vllm_api_key,
        default_batch_size=batch_size,
    )
