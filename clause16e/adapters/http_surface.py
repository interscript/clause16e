"""Self-hosted HTTP inference-surface adapter (the released runtime).

For probing the surface that will actually ship. The base URL must be
a locally controlled runtime (e.g. a self-hosted runtime served on
loopback); corpus-derived prompts are only sent when
``surface.local = true`` is attested in the run config. Optional auth
comes from an environment variable named in ``token_env`` — tokens are
never stored in config files.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

from .base import ModelAdapter, register_adapter


class HTTPSurfaceAdapter(ModelAdapter):
    name = "http"

    def __init__(self, params: dict):
        self.base_url = params["base_url"].rstrip("/")
        self.timeout = float(params.get("timeout", 120))
        self.token_env = params.get("token_env")
        headers = {"Content-Type": "application/json"}
        if self.token_env:
            token = os.environ.get(self.token_env)
            if token:
                headers["Authorization"] = f"Bearer {token}"
        self.headers = headers
        self.model_id = params.get("model_id", self.base_url)

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str) -> Optional[dict]:
        req = urllib.request.Request(
            f"{self.base_url}{path}", headers=self.headers, method="GET"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise

    def generate(self, prompt, max_new_tokens, temperature, seed):
        out = self._post(
            "/v1/generate",
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "seed": seed,
            },
        )
        return out["text"]

    def sequence_logprob(self, text):
        out = self._post("/v1/logprob", {"text": text})
        return (float(out["log_prob"]), int(out["num_tokens"]))

    def vocab(self) -> Optional[list[str]]:
        out = self._get("/v1/vocab")
        if out is None:
            return None
        return list(out.get("vocab", []))

    def embed_texts(self, texts):
        try:
            out = self._post("/v1/embed", {"texts": texts})
        except (urllib.error.HTTPError, urllib.error.URLError):
            return None
        return out.get("embeddings")

    def describe(self):
        return {"adapter": self.name, "model_id": self.model_id, "base_url": self.base_url}


register_adapter("http", lambda params: HTTPSurfaceAdapter(params))
