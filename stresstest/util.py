"""Small helpers shared across the harness. Standard library only."""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------

def now() -> float:
    """Monotonic clock, for durations. Never walk backwards on NTP steps."""
    return time.monotonic()


def wall() -> float:
    """Wall clock, for timestamps that must line up with server logs."""
    return time.time()


def iso(ts: float | None = None) -> str:
    if ts is None:
        ts = wall()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + "Z"


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}u{minutes:02d}m"


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile. p in [0, 100]. Empty -> nan."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (p / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return float(ordered[low])
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def summarize(values: Sequence[float]) -> dict[str, float]:
    """The five numbers we report for every latency distribution."""
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "max": float(max(values)) if values else float("nan"),
    }


def sample_lognormal(rng: random.Random, low: float, high: float,
                     clamp_low: float | None = None,
                     clamp_high: float | None = None) -> float:
    """Draw from a lognormal shaped so that ``low`` is roughly the 10th and
    ``high`` roughly the 90th percentile.

    Real think time is skewed: most pauses are short, a few are very long.
    A uniform draw over [low, high] would flatten exactly the tail that
    creates the interesting bursts, so we do not use one.
    """
    if low <= 0:
        low = 0.001
    if high <= low:
        return float(low)
    mu = (math.log(low) + math.log(high)) / 2.0
    sigma = (math.log(high) - math.log(low)) / (2.0 * 1.2815515655446004)
    value = math.exp(rng.gauss(mu, sigma))
    lo = clamp_low if clamp_low is not None else low * 0.25
    hi = clamp_high if clamp_high is not None else high * 4.0
    return float(min(max(value, lo), hi))


# --------------------------------------------------------------------------
# Config loading (JSON with // comments, so it stays human editable
# without pulling in a YAML or TOML dependency)
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r'^\s*//')


def strip_json_comments(text: str) -> str:
    return "\n".join("" if _COMMENT_RE.match(line) else line
                     for line in text.splitlines())


def load_jsonc(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    try:
        return json.loads(strip_json_comments(raw))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config {path} is not valid JSON: {exc}") from exc


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False, default=_json_default)
        handle.write("\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def write_csv(path: str, rows: Iterable[dict], columns: Sequence[str] | None = None) -> int:
    import csv
    rows = list(rows)
    if not rows:
        # Still create the file so downstream tooling does not trip over a
        # missing path; an empty CSV is a legitimate result.
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        open(path, "w", encoding="utf-8").close()
        return 0
    if columns is None:
        columns = list(rows[0].keys())
        for row in rows:
            for key in row:
                if key not in columns:
                    columns = list(columns) + [key]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        # csv defaults to CRLF; everything else this harness writes is LF, and a
        # regenerated report should not show up as a diff in every line.
        writer = csv.DictWriter(handle, fieldnames=list(columns),
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in columns})
    return len(rows)


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, 4)
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return value


# --------------------------------------------------------------------------
# Console output
# --------------------------------------------------------------------------

_COLORS = {
    "green": "\033[32m",
    "amber": "\033[33m",
    "red": "\033[31m",
    "grey": "\033[90m",
    "bold": "\033[1m",
    "reset": "\033[0m",
}


def supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stderr.isatty()


def colored(text: str, color: str) -> str:
    if not supports_color() or color not in _COLORS:
        return text
    return f"{_COLORS[color]}{text}{_COLORS['reset']}"


def log(message: str, *, color: str | None = None) -> None:
    """Progress goes to stderr so that stdout stays parseable."""
    stamp = time.strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(colored(line, color) if color else line, file=sys.stderr, flush=True)
