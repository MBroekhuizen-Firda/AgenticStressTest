"""Token counting.

Exact counting needs the model's tokenizer, which means a dependency and a
download. The harness therefore has three tiers and tells you which one it
used, because the context axis of the whole matrix is expressed in tokens
and a silent 30 % error there would invalidate the results.

  1. ``tokenizers``   (HuggingFace, small, fast, no torch)  -- exact
  2. ``transformers`` (if it happens to be installed anyway) -- exact
  3. characters / ratio                                     -- estimate

The fallback ratio is calibrated for source code plus English/Dutch prose
mixed together, which is what an agent transcript looks like. Measured on
Qwen3-Coder's tokenizer over a few thousand lines of Python, PHP and
JavaScript it sits between 3.3 and 3.8 characters per token; 3.5 is the
default. ``stresstest calibrate`` re-measures it against your own corpus
when a real tokenizer is available.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

DEFAULT_CHARS_PER_TOKEN = 3.5

# Every chat message costs a few tokens of role/delimiter framing on top of
# its content. The exact number depends on the chat template; 4 is a good
# average for ChatML-style templates and the error is negligible next to a
# 32k context.
MESSAGE_OVERHEAD_TOKENS = 4


class TokenCounter:
    """Counts tokens in text and in whole message lists."""

    def __init__(self, backend: str, name: str, chars_per_token: float,
                 encode: Any = None) -> None:
        self.backend = backend        # "tokenizers" | "transformers" | "estimate"
        self.name = name
        self.chars_per_token = chars_per_token
        self._encode = encode

    @property
    def exact(self) -> bool:
        return self.backend != "estimate"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encode is not None:
            return int(self._encode(text))
        return int(math.ceil(len(text) / self.chars_per_token))

    def count_messages(self, messages: Sequence[dict]) -> int:
        total = 0
        for message in messages:
            total += MESSAGE_OVERHEAD_TOKENS
            content = message.get("content")
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self.count(str(part.get("text", "")))
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                total += self.count(str(function.get("name", "")))
                total += self.count(str(function.get("arguments", "")))
                total += 6
        return total

    def count_tools(self, tools: Sequence[dict]) -> int:
        """Tool definitions are serialised into the prompt by the template."""
        import json
        return self.count(json.dumps(list(tools), separators=(",", ":")))

    def describe(self) -> str:
        if self.exact:
            return f"exact ({self.backend}: {self.name})"
        return f"estimate ({self.chars_per_token:.2f} chars/token)"


def build_counter(model_id: str,
                  chars_per_token: float = DEFAULT_CHARS_PER_TOKEN,
                  prefer_exact: bool = True,
                  tokenizer_path: str | None = None) -> TokenCounter:
    """Return the best counter available, without ever raising."""
    if prefer_exact:
        source = tokenizer_path or model_id
        counter = _try_tokenizers(source)
        if counter is not None:
            return counter
        counter = _try_transformers(source)
        if counter is not None:
            return counter
    return TokenCounter("estimate", model_id, chars_per_token)


def _try_tokenizers(source: str) -> TokenCounter | None:
    try:
        from tokenizers import Tokenizer  # type: ignore
    except Exception:
        return None
    try:
        if os.path.isdir(source):
            candidate = os.path.join(source, "tokenizer.json")
            tokenizer = Tokenizer.from_file(candidate)
        elif os.path.isfile(source):
            tokenizer = Tokenizer.from_file(source)
        else:
            tokenizer = Tokenizer.from_pretrained(source)
    except Exception:
        return None
    return TokenCounter("tokenizers", source, DEFAULT_CHARS_PER_TOKEN,
                        encode=lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids))


def _try_transformers(source: str) -> TokenCounter | None:
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=False)
    except Exception:
        return None
    return TokenCounter("transformers", source, DEFAULT_CHARS_PER_TOKEN,
                        encode=lambda text: len(tokenizer.encode(text, add_special_tokens=False)))


def calibrate_ratio(counter: TokenCounter, samples: Sequence[str]) -> float | None:
    """Measure characters per token over real corpus text.

    Only meaningful when an exact tokenizer is available; returns None
    otherwise so the caller can say so instead of inventing a number.
    """
    if not counter.exact or not samples:
        return None
    characters = sum(len(sample) for sample in samples)
    tokens = sum(counter.count(sample) for sample in samples)
    if tokens == 0:
        return None
    return characters / tokens
