"""A fake vLLM, so the harness can be validated without renting a GPU.

It is not a model. It is a queue-and-memory simulator that speaks the
OpenAI streaming API and exposes a Prometheus ``/metrics`` endpoint with the
same metric names vLLM uses, and it reproduces the four behaviours the
experiment is about:

*  block-level prefix caching, so a shared project skeleton really is cheaper
*  a KV cache of finite size, with eviction
*  a run queue with a ``max_num_seqs`` limit
*  preemption when the KV cache cannot hold every running sequence

Use it to check that the whole pipeline works and that the report comes out
right before you spend money on hardware. Numbers from the mock say nothing
about real hardware -- they are only there to exercise the harness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOCK_TOKENS = 256
CHARS_PER_TOKEN = 3.5


class Simulator:
    def __init__(self, *, gpu_memory_gb: float, model_gb: float,
                 kv_bytes_per_token: float, prefill_tokens_per_s: float,
                 decode_tokens_per_s: float, max_num_seqs: int,
                 max_model_len: int) -> None:
        self.lock = threading.Lock()
        usable = max(gpu_memory_gb - model_gb, 0.5) * 1e9
        self.kv_capacity_tokens = int(usable / max(kv_bytes_per_token, 1.0))
        self.prefill_tokens_per_s = prefill_tokens_per_s
        self.decode_tokens_per_s = decode_tokens_per_s
        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len

        self.cache: "OrderedDict[str, int]" = OrderedDict()
        self.cached_tokens = 0
        self.running: dict[int, int] = {}
        self.waiting = 0
        self.next_id = 0

        self.m_prefix_queries = 0
        self.m_prefix_hits = 0
        self.m_preemptions = 0
        self.m_prompt_tokens = 0
        self.m_generation_tokens = 0
        self.m_requests_ok = 0

    # ---------------------------------------------------------------- cache

    def _block_hashes(self, text: str) -> list[str]:
        chunk = int(BLOCK_TOKENS * CHARS_PER_TOKEN)
        hashes, previous = [], "root"
        for start in range(0, len(text), chunk):
            digest = hashlib.blake2b(
                (previous + text[start:start + chunk]).encode("utf-8", "replace"),
                digest_size=16).hexdigest()
            hashes.append(digest)
            previous = digest
        return hashes

    def _evict(self, needed: int) -> None:
        while self.cached_tokens + needed > self.kv_capacity_tokens and self.cache:
            _, tokens = self.cache.popitem(last=False)
            self.cached_tokens -= tokens

    def admit(self, prompt_text: str) -> tuple[int, int, int, float]:
        """Returns (request_id, prompt_tokens, uncached_tokens, queue_wait_s)."""
        hashes = self._block_hashes(prompt_text)
        prompt_tokens = min(max(int(len(prompt_text) / CHARS_PER_TOKEN), 1), self.max_model_len)
        with self.lock:
            hit_blocks = 0
            for digest in hashes:
                if digest in self.cache:
                    self.cache.move_to_end(digest)
                    hit_blocks += 1
                else:
                    break
            self.m_prefix_queries += len(hashes)
            self.m_prefix_hits += hit_blocks

            uncached = max(prompt_tokens - hit_blocks * BLOCK_TOKENS, 1)
            self._evict(uncached)
            for digest in hashes[hit_blocks:]:
                if digest not in self.cache:
                    self.cache[digest] = BLOCK_TOKENS
                    self.cached_tokens += BLOCK_TOKENS

            # Preempt when the running set no longer fits in the KV pool.
            live = sum(self.running.values()) + prompt_tokens
            while live > self.kv_capacity_tokens and self.running:
                victim = next(iter(self.running))
                live -= self.running.pop(victim)
                self.m_preemptions += 1

            request_id = self.next_id
            self.next_id += 1
            queued = max(0, len(self.running) + 1 - self.max_num_seqs)
            if queued:
                self.waiting += 1
            self.running[request_id] = prompt_tokens
            self.m_prompt_tokens += prompt_tokens
            running_now = len(self.running)

        # Queue wait grows once more sequences than max_num_seqs are in flight.
        queue_wait = 0.0
        if queued:
            queue_wait = queued * (BLOCK_TOKENS * 4) / self.prefill_tokens_per_s
        # Prefill shares the GPU with everything else that is running.
        share = max(1.0, running_now / 4.0)
        prefill_time = uncached / self.prefill_tokens_per_s * share
        return request_id, prompt_tokens, uncached, queue_wait + prefill_time

    def decode_delay(self) -> float:
        with self.lock:
            running = max(1, len(self.running))
        # Aggregate decode throughput is roughly constant, so per-stream speed
        # falls as concurrency rises -- the effect students actually feel.
        per_stream = self.decode_tokens_per_s / (1.0 + 0.35 * (running - 1))
        return 1.0 / max(per_stream, 0.5)

    def finish(self, request_id: int, generated: int) -> None:
        with self.lock:
            if self.waiting > 0:
                self.waiting -= 1
            self.running.pop(request_id, None)
            self.m_generation_tokens += generated
            self.m_requests_ok += 1

    def metrics_text(self, model: str) -> str:
        with self.lock:
            usage = min(1.0, sum(self.running.values()) / max(self.kv_capacity_tokens, 1))
            lines = [
                f'vllm:num_requests_running{{model_name="{model}"}} {len(self.running)}',
                f'vllm:num_requests_waiting{{model_name="{model}"}} {self.waiting}',
                f'vllm:gpu_cache_usage_perc{{model_name="{model}"}} {usage:.6f}',
                f'vllm:kv_cache_usage_perc{{model_name="{model}"}} {usage:.6f}',
                f'vllm:num_preemptions_total{{model_name="{model}"}} {self.m_preemptions}',
                f'vllm:prefix_cache_queries_total{{model_name="{model}"}} {self.m_prefix_queries}',
                f'vllm:prefix_cache_hits_total{{model_name="{model}"}} {self.m_prefix_hits}',
                f'vllm:prompt_tokens_total{{model_name="{model}"}} {self.m_prompt_tokens}',
                f'vllm:generation_tokens_total{{model_name="{model}"}} {self.m_generation_tokens}',
                f'vllm:request_success_total{{model_name="{model}",finished_reason="length"}} {self.m_requests_ok}',
            ]
        header = ["# HELP vllm:num_preemptions_total Cumulative preemptions",
                  "# TYPE vllm:num_preemptions_total counter"]
        return "\n".join(header + lines) + "\n"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    simulator: Simulator
    model_name: str = "mock-model"

    def log_message(self, fmt, *args):  # noqa: A003 - silence the default logger
        pass

    def do_GET(self):  # noqa: N802
        if self.path.rstrip("/") in ("/metrics",):
            body = self.simulator.metrics_text(self.model_name).encode()
            self._send(200, body, "text/plain; version=0.0.4")
        elif self.path.rstrip("/").endswith("/models"):
            body = json.dumps({"object": "list", "data": [{
                "id": self.model_name, "object": "model", "owned_by": "mock",
                "max_model_len": self.simulator.max_model_len,
            }]}).encode()
            self._send(200, body, "application/json")
        elif self.path.rstrip("/") in ("/health", "/ping"):
            self._send(200, b"ok", "text/plain")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self):  # noqa: N802
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error":"bad json"}', "application/json")
            return

        prompt_text = json.dumps(payload.get("messages", []), sort_keys=False)
        prompt_text += json.dumps(payload.get("tools", []), sort_keys=False)
        max_tokens = int(payload.get("max_tokens", 128))
        simulator = self.simulator

        request_id, prompt_tokens, uncached, first_token_delay = simulator.admit(prompt_text)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        time.sleep(first_token_delay)
        generated = 0
        try:
            for index in range(max_tokens):
                if index:
                    time.sleep(simulator.decode_delay())
                self._chunk(self._sse({"choices": [{"index": 0, "delta": {"content": "tok "}}]}))
                generated += 1
            self._chunk(self._sse({
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": generated,
                          "total_tokens": prompt_tokens + generated,
                          "prompt_tokens_details": {
                              "cached_tokens": max(prompt_tokens - uncached, 0)}},
            }))
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            simulator.finish(request_id, generated)

    # ------------------------------------------------------------- plumbing

    def _sse(self, obj: dict) -> bytes:
        obj.setdefault("id", "chatcmpl-mock")
        obj.setdefault("object", "chat.completion.chunk")
        obj.setdefault("model", self.model_name)
        return b"data: " + json.dumps(obj).encode() + b"\n\n"

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(host: str, port: int, **kwargs) -> ThreadingHTTPServer:
    simulator = Simulator(**kwargs)
    handler = type("BoundHandler", (Handler,), {"simulator": simulator})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    server.simulator = simulator  # type: ignore[attr-defined]
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fake vLLM for validating the harness")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-gb", type=float, default=96.0)
    parser.add_argument("--model-gb", type=float, default=31.0)
    parser.add_argument("--kv-bytes-per-token", type=float, default=45_000.0)
    parser.add_argument("--prefill-tokens-per-s", type=float, default=9000.0)
    parser.add_argument("--decode-tokens-per-s", type=float, default=95.0)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=131072)
    args = parser.parse_args(argv)

    server = build_server(
        args.host, args.port,
        gpu_memory_gb=args.gpu_memory_gb, model_gb=args.model_gb,
        kv_bytes_per_token=args.kv_bytes_per_token,
        prefill_tokens_per_s=args.prefill_tokens_per_s,
        decode_tokens_per_s=args.decode_tokens_per_s,
        max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len)
    simulator = server.simulator  # type: ignore[attr-defined]
    print(f"mock vLLM on http://{args.host}:{args.port}  "
          f"(KV pool {simulator.kv_capacity_tokens:,} tokens, "
          f"max_num_seqs={args.max_num_seqs})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
