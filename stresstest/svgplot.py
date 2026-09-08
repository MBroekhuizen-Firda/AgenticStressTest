"""Charts without a plotting library.

matplotlib is used when it is installed. When it is not -- and on a freshly
rented GPU box it usually is not -- these functions write the same four
charts as plain SVG, which every browser and every document editor opens.
"""

from __future__ import annotations

import html
import math
from typing import Sequence

FONT = "system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
COLOURS = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#4a5568"]
GRADE_FILL = {"groen": "#38a169", "oranje": "#dd6b20", "rood": "#c53030",
              "onbekend": "#a0aec0"}


def _escape(text: object) -> str:
    return html.escape(str(text), quote=True)


def _header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.0f}" y="28" text-anchor="middle" font-size="17" '
        f'font-weight="600" fill="#1a202c">{_escape(title)}</text>',
    ]


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    if not (high > low):
        return [low]
    span = high - low
    raw = span / max(count, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for factor in (1, 2, 2.5, 5, 10):
        step = magnitude * factor
        if step >= raw:
            break
    start = math.floor(low / step) * step
    ticks, value = [], start
    while value <= high + step * 0.5:
        if value >= low - step * 0.5:
            ticks.append(round(value, 10))
        value += step
    return ticks


def heatmap(path: str, *, title: str, row_labels: Sequence[str],
            column_labels: Sequence[str], grades: Sequence[Sequence[str]],
            annotations: Sequence[Sequence[str]], subtitle: str = "") -> None:
    """The coloured matrix: students on one axis, context size on the other."""
    cell_w, cell_h = 150, 62
    left, top = 150, 78
    width = left + cell_w * len(column_labels) + 40
    height = top + cell_h * len(row_labels) + 90
    parts = _header(width, height, title)
    if subtitle:
        parts.append(f'<text x="{width / 2:.0f}" y="48" text-anchor="middle" '
                     f'font-size="12" fill="#4a5568">{_escape(subtitle)}</text>')
    for index, label in enumerate(column_labels):
        x = left + cell_w * index + cell_w / 2
        parts.append(f'<text x="{x:.0f}" y="{top - 10}" text-anchor="middle" '
                     f'font-size="13" fill="#2d3748">{_escape(label)}</text>')
    for row, label in enumerate(row_labels):
        y = top + cell_h * row + cell_h / 2
        parts.append(f'<text x="{left - 14}" y="{y + 5:.0f}" text-anchor="end" '
                     f'font-size="13" fill="#2d3748">{_escape(label)}</text>')
        for column in range(len(column_labels)):
            grade = grades[row][column] if column < len(grades[row]) else "onbekend"
            fill = GRADE_FILL.get(grade, "#a0aec0")
            x = left + cell_w * column
            y0 = top + cell_h * row
            parts.append(f'<rect x="{x}" y="{y0}" width="{cell_w - 4}" '
                         f'height="{cell_h - 4}" rx="4" fill="{fill}"/>')
            text = annotations[row][column] if column < len(annotations[row]) else ""
            for line_no, line in enumerate(str(text).split("\n")[:2]):
                parts.append(
                    f'<text x="{x + (cell_w - 4) / 2:.0f}" '
                    f'y="{y0 + 26 + line_no * 16:.0f}" text-anchor="middle" '
                    f'font-size="12" fill="#ffffff">{_escape(line)}</text>')
    legend_y = height - 40
    for index, (name, colour) in enumerate(
            [("groen", GRADE_FILL["groen"]), ("oranje", GRADE_FILL["oranje"]),
             ("rood", GRADE_FILL["rood"]), ("niet gemeten", GRADE_FILL["onbekend"])]):
        x = left + index * 130
        parts.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" rx="3" fill="{colour}"/>')
        parts.append(f'<text x="{x + 20}" y="{legend_y}" font-size="12" '
                     f'fill="#2d3748">{_escape(name)}</text>')
    parts.append("</svg>")
    _write(path, parts)


def line_chart(path: str, *, title: str, x_label: str, y_label: str,
               series: Sequence[dict], hlines: Sequence[dict] = ()) -> None:
    """series: [{"name": str, "x": [...], "y": [...]}]"""
    width, height = 860, 520
    left, right, top, bottom = 80, 200, 60, 70
    plot_w, plot_h = width - left - right, height - top - bottom
    xs = [x for s in series for x in s["x"]]
    ys = [y for s in series for y in s["y"] if y == y]
    ys += [line["y"] for line in hlines]
    if not xs or not ys:
        _write(path, _header(width, height, title) + ["</svg>"])
        return
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys) * 1.12
    if x_max == x_min:
        x_max = x_min + 1
    if y_max <= y_min:
        y_max = y_min + 1

    def px(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def py(value: float) -> float:
        return top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    parts = _header(width, height, title)
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
                 f'fill="#f7fafc" stroke="#e2e8f0"/>')
    for tick in _nice_ticks(y_min, y_max):
        y = py(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" '
                     f'font-size="11" fill="#4a5568">{_fmt(tick)}</text>')
    for tick in _nice_ticks(x_min, x_max):
        x = px(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" '
                     f'stroke="#edf2f7"/>')
        parts.append(f'<text x="{x:.1f}" y="{top + plot_h + 18}" text-anchor="middle" '
                     f'font-size="11" fill="#4a5568">{_fmt(tick)}</text>')
    for line in hlines:
        y = py(line["y"])
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
                     f'stroke="{line.get("colour", "#c53030")}" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{left + 6}" y="{y - 5:.1f}" font-size="11" '
                     f'fill="{line.get("colour", "#c53030")}">{_escape(line.get("label", ""))}</text>')
    for index, item in enumerate(series):
        colour = COLOURS[index % len(COLOURS)]
        points = [(px(x), py(y)) for x, y in zip(item["x"], item["y"]) if y == y]
        if points:
            path_d = " ".join(("M" if i == 0 else "L") + f"{x:.1f},{y:.1f}"
                              for i, (x, y) in enumerate(points))
            parts.append(f'<path d="{path_d}" fill="none" stroke="{colour}" stroke-width="2.2"/>')
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{colour}"/>')
        legend_y = top + 18 + index * 20
        parts.append(f'<line x1="{left + plot_w + 16}" y1="{legend_y - 4}" '
                     f'x2="{left + plot_w + 40}" y2="{legend_y - 4}" stroke="{colour}" stroke-width="2.2"/>')
        parts.append(f'<text x="{left + plot_w + 46}" y="{legend_y}" font-size="12" '
                     f'fill="#2d3748">{_escape(item["name"])}</text>')
    parts.append(f'<text x="{left + plot_w / 2:.0f}" y="{height - 18}" text-anchor="middle" '
                 f'font-size="12" fill="#2d3748">{_escape(x_label)}</text>')
    parts.append(f'<text x="18" y="{top + plot_h / 2:.0f}" font-size="12" fill="#2d3748" '
                 f'transform="rotate(-90 18 {top + plot_h / 2:.0f})" '
                 f'text-anchor="middle">{_escape(y_label)}</text>')
    parts.append("</svg>")
    _write(path, parts)


def bar_chart(path: str, *, title: str, labels: Sequence[str],
              values: Sequence[float], y_label: str,
              colours: Sequence[str] | None = None) -> None:
    width, height = 860, 480
    left, right, top, bottom = 80, 40, 60, 140
    plot_w, plot_h = width - left - right, height - top - bottom
    parts = _header(width, height, title)
    top_value = max(list(values) + [1.0]) * 1.15
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
                 f'fill="#f7fafc" stroke="#e2e8f0"/>')
    for tick in _nice_ticks(0, top_value):
        y = top + plot_h - tick / top_value * plot_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="#4a5568">{_fmt(tick)}</text>')
    if labels:
        slot = plot_w / len(labels)
        for index, (label, value) in enumerate(zip(labels, values)):
            bar_h = (value / top_value) * plot_h if value == value else 0
            x = left + slot * index + slot * 0.18
            w = slot * 0.64
            colour = (colours[index] if colours and index < len(colours)
                      else COLOURS[index % len(COLOURS)])
            parts.append(f'<rect x="{x:.1f}" y="{top + plot_h - bar_h:.1f}" width="{w:.1f}" '
                         f'height="{bar_h:.1f}" fill="{colour}" rx="3"/>')
            parts.append(f'<text x="{x + w / 2:.1f}" y="{top + plot_h - bar_h - 6:.1f}" '
                         f'text-anchor="middle" font-size="11" fill="#2d3748">{_fmt(value)}</text>')
            parts.append(f'<text x="{x + w / 2:.1f}" y="{top + plot_h + 16:.1f}" '
                         f'text-anchor="end" font-size="11" fill="#2d3748" '
                         f'transform="rotate(-35 {x + w / 2:.1f} {top + plot_h + 16:.1f})">'
                         f'{_escape(label)}</text>')
    parts.append(f'<text x="18" y="{top + plot_h / 2:.0f}" font-size="12" fill="#2d3748" '
                 f'transform="rotate(-90 18 {top + plot_h / 2:.0f})" text-anchor="middle">'
                 f'{_escape(y_label)}</text>')
    parts.append("</svg>")
    _write(path, parts)


def _fmt(value: float) -> str:
    if value != value:
        return ""
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _write(path: str, parts: Sequence[str]) -> None:
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(parts) + "\n")
