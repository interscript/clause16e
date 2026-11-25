"""Model adapters: the harness is model-agnostic behind this interface.

Any candidate model is exercised through three capabilities:
- ``generate`` (verbatim + extraction probes),
- ``sequence_logprob`` (perplexity-based membership inference),
- optional ``vocab`` / ``embed_texts`` (reference-resource leakage probe).

Adapters ship for: locally-run HF transformers (``hf``, ByT5-family
included), a self-hosted HTTP inference surface (``http`` — the
released runtime), and a deterministic ``mock`` used by the test
suite. Register new runtime surfaces via ``register_adapter``.
"""

from .base import ModelAdapter, create_adapter, register_adapter
from .hf_local import HFLocalAdapter
from .http_surface import HTTPSurfaceAdapter
from .mock import MockAdapter

__all__ = [
    "ModelAdapter",
    "create_adapter",
    "register_adapter",
    "HFLocalAdapter",
    "HTTPSurfaceAdapter",
    "MockAdapter",
]
