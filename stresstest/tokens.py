"""Token counting.

Exact counting needs the model's tokenizer, which means a dependency and a
download. The harness therefore has three tiers and tells you which one it
used, because the context axis of the whole matrix is expressed in tokens
and a silent 30 % error there would invalidate the results.

  1. ``tokenizers``   (HuggingFace, small, fast, no torch)  -- exact
  2. ``transformers`` (if it happens to be installed anyway) -- exact
  3. characters / ratio                                     -- estimate

The fallback ratio matters more than it looks: the context size is an axis
of the whole matrix, so a systematic error there moves the conclusion about
how much memory a class needs. Measured on this harness's own corpus with
the Qwen3-Coder tokenizer, source code runs at about 4.7 characters per
token (PHP 4.80, Python 4.80, JavaScript 4.69, Markdown 4.29) and a
complete agent message list at about 4.6 once the JSON escaping and the
per-message framing are included. That 4.6 is the default.

The old rule of thumb of 3.5 characters per token comes from English prose
and is wrong for code by roughly a third: with 3.5 the harness built a "32k"
context that really held 17.7k tokens. Run ``stresstest calibrate`` to
re-measure the ratio against your own corpus and model.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

DEFAULT_CHARS_PER_TOKEN = 4.6

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


def best_ratio(measure, low: float = 3.0, high: float = 6.0,
               step: float = 0.05) -> tuple[float, float]:
    """Find the characters-per-token value that makes the estimate match.

    ``measure(ratio)`` returns the list of relative errors that ratio
    produces. We minimise the largest absolute error rather than the mean,
    because a ratio that is right on average but 15 % out at 100k context
    would still mis-size the runs that decide the memory question.
    """
    best_value, best_error = low, float("inf")
    ratio = low
    while ratio <= high + 1e-9:
        errors = measure(round(ratio, 4))
        worst = max(abs(error) for error in errors) if errors else float("inf")
        if worst < best_error:
            best_value, best_error = round(ratio, 4), worst
        ratio += step
    return best_value, best_error
