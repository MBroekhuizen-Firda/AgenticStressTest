"""Minimal async HTTP/1.1 client with server-sent-events support.

Why not ``requests``/``httpx``/``aiohttp``: the harness has to still work in
two years on a freshly rented GPU box, and every dependency is a chance for
that to fail. This is ~250 lines of standard library that speak exactly the
subset of HTTP the OpenAI streaming API needs.

Timing note: ``ttft_s`` is measured from the moment the request bytes have
been written to the socket until the first content token arrives. TCP/TLS
connection setup is measured separately as ``connect_s`` so it can never be
mistaken for model latency.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator
from urllib.parse import urlsplit


class HttpError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


@dataclass
class Endpoint:
    base_url: str
    api_key: str | None = None
    model: str = "local-model"
    timeout_s: float = 600.0
    connect_timeout_s: float = 30.0
    verify_tls: bool = True
    extra_headers: dict[str, str] = field(default_factory=dict)

    @property
    def parts(self):
        split = urlsplit(self.base_url)
        scheme = split.scheme or "http"
        host = split.hostname or "127.0.0.1"
        port = split.port or (443 if scheme == "https" else 80)
        path = split.path.rstrip("/")
        return scheme, host, port, path

    def url_for(self, suffix: str) -> str:
        scheme, host, port, path = self.parts
        return f"{scheme}://{host}:{port}{path}{suffix}"


class _Connection:
    """One request per connection. Simple, and avoids the whole class of
    keep-alive bugs; vLLM handles connection churn without complaint."""

    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint
        self.reader: asyncio.StreamReader | None = None
        self.writer: asyncio.StreamWriter | None = None

    async def open(self) -> float:
        scheme, host, port, _ = self.endpoint.parts
        context = None
        if scheme == "https":
            context = ssl.create_default_context()
            if not self.endpoint.verify_tls:
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
        started = time.monotonic()
        self.reader, self.writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=context),
            timeout=self.endpoint.connect_timeout_s,
        )
        return time.monotonic() - started

    async def send(self, method: str, path: str, body: bytes | None,
                   headers: dict[str, str]) -> None:
        assert self.writer is not None
        _, host, port, _ = self.endpoint.parts
        lines = [f"{method} {path} HTTP/1.1", f"Host: {host}:{port}",
                 "Connection: close", "Accept-Encoding: identity"]
        for key, value in headers.items():
            lines.append(f"{key}: {value}")
        if body is not None:
            lines.append(f"Content-Length: {len(body)}")
        request = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
        self.writer.write(request + (body or b""))
        await self.writer.drain()

    async def read_headers(self) -> tuple[int, dict[str, str]]:
        assert self.reader is not None
        status_line = await self.reader.readline()
        if not status_line:
            raise HttpError(0, "connection closed before response")
        parts = status_line.decode("latin-1").split(" ", 2)
        status = int(parts[1]) if len(parts) > 1 else 0
        headers: dict[str, str] = {}
        while True:
            line = await self.reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode("latin-1").partition(":")
            headers[key.strip().lower()] = value.strip()
        return status, headers

    async def iter_body(self, headers: dict[str, str]) -> AsyncIterator[bytes]:
        assert self.reader is not None
        if headers.get("transfer-encoding", "").lower() == "chunked":
            while True:
                size_line = await self.reader.readline()
                if not size_line:
                    return
                try:
                    size = int(size_line.split(b";")[0].strip() or b"0", 16)
                except ValueError:
                    return
                if size == 0:
                    await self.reader.readline()
                    return
                chunk = await self.reader.readexactly(size)
                await self.reader.readexactly(2)  # trailing CRLF
                yield chunk
        elif "content-length" in headers:
            remaining = int(headers["content-length"])
            while remaining > 0:
                chunk = await self.reader.read(min(65536, remaining))
                if not chunk:
                    return
                remaining -= len(chunk)
                yield chunk
        else:
            while True:
                chunk = await self.reader.read(65536)
                if not chunk:
                    return
                yield chunk

    async def close(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
            self.writer = None
            self.reader = None


@dataclass
class StreamResult:
    """Everything one completion tells us."""
    ok: bool = False
    error: str | None = None
    status: int = 0
    connect_s: float = 0.0
    ttft_s: float = float("nan")
    total_s: float = 0.0
    output_tokens: int = 0
    prompt_tokens: int = 0
    reported_prompt_tokens: int | None = None
    reported_output_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    text: str = ""
    inter_token_latencies: list[float] = field(default_factory=list)

    @property
    def decode_tokens_per_s(self) -> float:
        """Tokens per second once generation has started -- what a student
        perceives as 'the model is typing'."""
        if self.output_tokens <= 1 or self.ttft_s != self.ttft_s:
            return float("nan")
        decode_time = self.total_s - self.ttft_s
        if decode_time <= 0:
            return float("nan")
        return (self.output_tokens - 1) / decode_time


class OpenAIClient:
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint."""

    def __init__(self, endpoint: Endpoint) -> None:
        self.endpoint = endpoint
        # Some servers reject unknown fields. We downgrade once and remember,
        # rather than failing an eight-minute run over a flag.
        self.supports_extra_body = True
        self.supports_tool_messages = True
        self._downgrade_lock = asyncio.Lock()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": "firda-stresstest/1.0"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"
        headers.update(self.endpoint.extra_headers)
        return headers

    async def chat_stream(self, messages: list[dict], *,
                          max_tokens: int,
                          temperature: float = 0.0,
                          tools: list[dict] | None = None,
                          extra_body: dict[str, Any] | None = None) -> StreamResult:
        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
        if extra_body and self.supports_extra_body:
            payload.update(extra_body)

        result = StreamResult()
        _, _, _, base_path = self.endpoint.parts
        connection = _Connection(self.endpoint)
        started = time.monotonic()
        try:
            result.connect_s = await connection.open()
            body = json.dumps(payload).encode("utf-8")
            await connection.send("POST", f"{base_path}/chat/completions", body, self._headers())
            send_done = time.monotonic()
            status, headers = await asyncio.wait_for(
                connection.read_headers(), timeout=self.endpoint.timeout_s)
            result.status = status
            if status != 200:
                raw = b""
                async for chunk in connection.iter_body(headers):
                    raw += chunk
                    if len(raw) > 64_000:
                        break
                text = raw.decode("utf-8", "replace")
                if self.supports_extra_body and extra_body and _looks_like_field_error(text):
                    async with self._downgrade_lock:
                        self.supports_extra_body = False
                    await connection.close()
                    return await self.chat_stream(messages, max_tokens=max_tokens,
                                                  temperature=temperature, tools=tools,
                                                  extra_body=None)
                if self.supports_tool_messages and _looks_like_tool_error(text):
                    async with self._downgrade_lock:
                        self.supports_tool_messages = False
                result.error = f"http {status}: {text[:300]}"
                return result

            await self._consume_sse(connection, headers, result, send_done)
            result.ok = result.error is None
            return result
        except asyncio.TimeoutError:
            result.error = f"timeout after {self.endpoint.timeout_s:.0f}s"
            return result
        except Exception as exc:  # noqa: BLE001 - a failed request is data, not a crash
            result.error = f"{type(exc).__name__}: {exc}"
            return result
        finally:
            result.total_s = time.monotonic() - started
            await connection.close()

    async def _consume_sse(self, connection: _Connection, headers: dict[str, str],
                           result: StreamResult, send_done: float) -> None:
        buffer = b""
        last_token_at: float | None = None
        deadline = send_done + self.endpoint.timeout_s
        body_iter = connection.iter_body(headers)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result.error = "timeout while streaming"
                return
            try:
                chunk = await asyncio.wait_for(body_iter.__anext__(), timeout=remaining)
            except StopAsyncIteration:
                break
            except asyncio.TimeoutError:
                result.error = "timeout while streaming"
                return
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip()
                if not line or not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if usage:
                    result.reported_prompt_tokens = usage.get("prompt_tokens")
                    result.reported_output_tokens = usage.get("completion_tokens")
                    details = usage.get("prompt_tokens_details") or {}
                    if isinstance(details, dict) and details.get("cached_tokens") is not None:
                        result.cached_prompt_tokens = details.get("cached_tokens")
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = delta.get("content") or ""
                    if not piece:
                        for call in delta.get("tool_calls") or []:
                            piece += str((call.get("function") or {}).get("arguments") or "")
                    if not piece:
                        continue
                    stamp = time.monotonic()
                    if result.output_tokens == 0:
                        result.ttft_s = stamp - send_done
                    elif last_token_at is not None:
                        result.inter_token_latencies.append(stamp - last_token_at)
                    last_token_at = stamp
                    result.output_tokens += 1
                    result.text += piece
        if result.output_tokens == 0 and result.error is None:
            result.error = "empty response"

    async def probe(self) -> dict[str, Any]:
        """Ask the endpoint what it is. Used by ``stresstest doctor``."""
        _, _, _, base_path = self.endpoint.parts
        connection = _Connection(self.endpoint)
        try:
            await connection.open()
            await connection.send("GET", f"{base_path}/models", None, self._headers())
            status, headers = await connection.read_headers()
            raw = b""
            async for chunk in connection.iter_body(headers):
                raw += chunk
            if status != 200:
                return {"ok": False, "status": status, "body": raw.decode("utf-8", "replace")[:300]}
            payload = json.loads(raw.decode("utf-8", "replace"))
            models = payload.get("data", [])
            return {
                "ok": True,
                "models": [m.get("id") for m in models],
                # vLLM reports the context window it was started with. Worth
                # knowing before a run: a request above it is simply rejected.
                "max_model_len": next((m.get("max_model_len") for m in models
                                       if m.get("max_model_len")), None),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            await connection.close()


async def http_get_text(url: str, timeout_s: float = 10.0,
                        verify_tls: bool = True) -> str:
    """Plain GET returning text. Used for the Prometheus endpoint."""
    split = urlsplit(url)
    endpoint = Endpoint(base_url=f"{split.scheme}://{split.hostname}:{split.port or (443 if split.scheme == 'https' else 80)}",
                        timeout_s=timeout_s, connect_timeout_s=timeout_s,
                        verify_tls=verify_tls)
    connection = _Connection(endpoint)
    try:
        await connection.open()
        path = split.path or "/"
        if split.query:
            path += "?" + split.query
        await connection.send("GET", path, None, {"Accept": "text/plain"})
        status, headers = await asyncio.wait_for(connection.read_headers(), timeout=timeout_s)
        raw = b""
        async for chunk in connection.iter_body(headers):
            raw += chunk
        if status != 200:
            raise HttpError(status, raw.decode("utf-8", "replace"))
        return raw.decode("utf-8", "replace")
    finally:
        await connection.close()


def _looks_like_field_error(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in
               ("extra fields not permitted", "unexpected keyword", "unrecognized",
                "extra_forbidden", "ignore_eos", "unknown field"))


def _looks_like_tool_error(text: str) -> bool:
    lowered = text.lower()
    return any(needle in lowered for needle in
               ("tool", "does not support tools", "chat template"))
