"""Locally-run HF transformers adapter (ByT5/mt5/T5 and causal LMs).

Runs fully on this machine, satisfying the local-processing requirement
for corpus-conditioned prompts (corpus-derived prompts never leave it).
Heavy deps are imported lazily so the harness core stays installable
without torch.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .base import ModelAdapter, register_adapter


def _torch_and_transformers():
    try:
        import torch  # noqa: F401
        import transformers
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "adapter 'hf' requires locally installed torch+transformers "
            "(pip install 'clause16e[hf]' or the project venv)"
        ) from e
    return torch, transformers


def strip_prompt(gen_ids, input_len: int, is_encoder_decoder: bool):
    """Continuation ids from one raw ``generate`` output row.

    Causal LMs return prompt+continuation concatenated; encoder-decoder
    models return DECODER ids only, so slicing by the encoder input
    length would silently drop every generation shorter than the prompt
    and fabricate empty outputs (a trivially-passing probe record).
    """
    if is_encoder_decoder:
        return gen_ids
    return gen_ids[input_len:]


class HFLocalAdapter(ModelAdapter):
    name = "hf"

    def __init__(self, params: dict):
        torch, transformers = _torch_and_transformers()
        self.torch = torch
        self.model_path = params["name_or_path"]
        self.device = params.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_length = int(params.get("max_length", 4096))
        import transformers as tf

        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.model_path)
        config = tf.AutoConfig.from_pretrained(self.model_path)
        self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        cls = (
            tf.AutoModelForSeq2SeqLM
            if self.is_encoder_decoder
            else tf.AutoModelForCausalLM
        )
        self.model = cls.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _encode(self, text: str):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

    def generate(self, prompt, max_new_tokens, temperature, seed):
        torch = self.torch
        torch.manual_seed(seed)
        inputs = self._encode(prompt).to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4) if temperature > 0 else None,
                pad_token_id=self.tokenizer.pad_token_id or 0,
            )
        gen_ids = strip_prompt(
            out[0], inputs["input_ids"].shape[1], self.is_encoder_decoder
        )
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def sequence_logprob(self, text):
        torch = self.torch
        enc = self._encode(text)
        if self.is_encoder_decoder:
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            labels = inputs["input_ids"].clone()
            with torch.no_grad():
                out = self.model(**inputs, labels=labels)
            n = int(labels.ne(self.tokenizer.pad_token_id or -100).sum())
        else:
            inputs = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                out = self.model(**inputs, labels=inputs["input_ids"])
            n = max(1, int(inputs["input_ids"].shape[1]) - 1)
        total = float(out.loss) * n
        return (total, max(1, n))

    def vocab(self) -> Optional[list[str]]:
        return list(self.tokenizer.get_vocab().keys())

    def embed_texts(self, texts):
        torch = self.torch
        out = []
        for text in texts:
            enc = self._encode(text).to(self.device)
            with torch.no_grad():
                if self.is_encoder_decoder:
                    hidden = self.model.get_encoder()(**enc).last_hidden_state
                else:
                    hidden = self.model(**enc, output_hidden_states=True).hidden_states[-1]
                pooled = hidden.mean(dim=1)[0]
            norm = float(torch.dot(pooled, pooled)) ** 0.5 or 1.0
            out.append([float(x) / norm for x in pooled.tolist()])
        return out

    def describe(self):
        return {
            "adapter": self.name,
            "model_path": self.model_path,
            "model_sha8": hashlib.sha256(self.model_path.encode()).hexdigest()[:8],
            "device": self.device,
            "encoder_decoder": self.is_encoder_decoder,
        }


register_adapter("hf", lambda params: HFLocalAdapter(params))
