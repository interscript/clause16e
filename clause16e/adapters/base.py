"""ModelAdapter interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional


class ModelAdapter(ABC):
    """Uniform probe interface over any candidate model."""

    name: str = "adapter"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        """Greedy-or-sampled continuation of ``prompt``."""

    @abstractmethod
    def sequence_logprob(self, text: str) -> tuple[float, int]:
        """Teacher-forced ``(total_logprob, num_tokens)`` of ``text``.

        Membership inference uses the per-token mean; adapters MUST
        return the token count so long/short documents compare fairly.
        """

    def vocab(self) -> Optional[list[str]]:
        """Tokenizer vocabulary strings, if exposed (WordNet probe)."""
        return None

    def embed_texts(self, texts: list[str]) -> Optional[list[list[float]]]:
        """Mean-pooled embeddings, if exposed (optional WordNet probe)."""
        return None

    def describe(self) -> dict:
        return {"adapter": self.name}


_REGISTRY: dict[str, Callable[[dict], ModelAdapter]] = {}


def register_adapter(name: str, factory: Callable[[dict], ModelAdapter]) -> None:
    _REGISTRY[name] = factory


def create_adapter(name: str, params: dict) -> ModelAdapter:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown adapter {name!r}; known: {sorted(_REGISTRY)}. "
            "Register custom runtime surfaces with register_adapter()."
        )
    return _REGISTRY[name](params)
