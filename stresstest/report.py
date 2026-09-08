"""Writing results out: CSV, JSON, charts and the Dutch conclusion.

Everything a run produced goes to disk raw, so a later question can be
answered without re-renting a GPU. On top of that the harness writes the
coloured matrix and the four charts, and generates RESULTATEN.md with the
conclusion in plain language for whoever has to judge the budget request.
"""

from __future__ import annotations

import json
import os
from typing import Any, Sequence

from . import pngplot, svgplot
from .grading import GREEN, RED
from .runner import RunResult
from .util import iso, log, write_csv, write_json

SUMMARY_COLUMNS = [
    "run_id", "label", "kind", "students", "context_tokens", "activity",
    "shared_fraction", "grade", "grade_brief", "ttft_p50_s", "ttft_p90_s",
    "ttft_p99_s", "burst_p50_s", "burst_p90_s", "decode_tps_p50", "requests",
    "requests_failed", "error_rate", "bursts", "prefix_cache_hit_rate",
    "preemptions", "kv_cache_usage_avg", "kv_cache_usage_peak",
    "queue_depth_avg", "queue_depth_peak", "server_prefill_tps",
    "server_decode_tps", "client_output_tps", "max_students_ok",
    "reasons", "warnings", "duration_s",
]


def _draw(kind: str, svg_path: str, **kwargs) -> list[str]:
    """Write the SVG (always) and a PNG next to it when matplotlib is present."""
    getattr(svgplot, kind)(svg_path, **kwargs)
    written = [svg_path]
    if pngplot.available():
        png_path = svg_path[:-4] + ".png"
        try:
            getattr(pngplot, kind)(png_path, **kwargs)
            written.append(png_path)
        except Exception as exc:  # noqa: BLE001 - a chart is never worth a crash
            log(f"PNG-versie van {os.path.basename(svg_path)} mislukt: {exc}",
                color="amber")
    return written


class ResultsWriter:
    def __init__(self, directory: str, config: dict, environment: dict) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, "runs"), exist_ok=True)
        write_json(os.path.join(directory, "config.json"), config)
        write_json(os.path.join(directory, "environment.json"), environment)
        self.config = config
        self.environment = environment
        self.results: list[RunResult] = []

    # --------------------------------------------------------------- per run

    def add(self, result: RunResult) -> None:
        self.results.append(result)
        run_dir = os.path.join(self.directory, "runs", result.spec.run_id)
        os.makedirs(run_dir, exist_ok=True)
        write_json(os.path.join(run_dir, "run.json"), {
            "spec": result.spec.to_dict(),
            "started": iso(result.started_wall),
            "finished": iso(result.finished_wall),
            "composition": result.composition,
            "aggregate": result.aggregate,
            "server": result.server,
            "grade": result.grade.to_dict(),
            "grade_brief_definition": result.grade_brief.to_dict(),
            "ramp": result.ramp,
        })
        if self.config.get("output", {}).get("write_request_csv", True):
            write_csv(os.path.join(run_dir, "requests.csv"),
                      [r.to_row() for r in result.collector.requests])
        write_csv(os.path.join(run_dir, "bursts.csv"),
                  [b.to_row() for b in result.collector.bursts])
        write_csv(os.path.join(run_dir, "server_metrics.csv"), result.metric_rows)
        self.flush_summary()

    def flush_summary(self) -> None:
        rows = [r.summary_row() for r in self.results]
        write_csv(os.path.join(self.directory, "summary.csv"), rows, SUMMARY_COLUMNS)
        write_json(os.path.join(self.directory, "summary.json"), rows)

    # ---------------------------------------------------------------- charts

    def write_charts(self) -> list[str]:
        chart_dir = os.path.join(self.directory, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        written: list[str] = []
        written += self._chart_matrix(chart_dir)
        written += self._chart_ttft(chart_dir)
        written += self._chart_cache_over_time(chart_dir)
        written += self._chart_preemptions(chart_dir)
        written += self._chart_rampup(chart_dir)
        return written

    def _sweep(self) -> list[RunResult]:
        return [r for r in self.results if r.spec.kind == "sweep"]

    def _chart_matrix(self, chart_dir: str) -> list[str]:
        sweep = self._sweep()
        if not sweep:
            return []
        students = sorted({r.spec.students for r in sweep})
        contexts = sorted({r.spec.context_tokens for r in sweep})
        lookup = {(r.spec.students, r.spec.context_tokens): r for r in sweep}
        grades, notes = [], []
        for n in students:
            grade_row, note_row = [], []
            for context in contexts:
                result = lookup.get((n, context))
                if result is None:
                    grade_row.append("onbekend")
                    note_row.append("")
                    continue
                grade_row.append(result.grade.colour)
                ttft = result.aggregate.get("ttft", {}).get("p90", float("nan"))
                burst = result.aggregate.get("burst_duration", {}).get("p90", float("nan"))
                note_row.append(f"TTFT p90 {ttft:.0f}s\ninstructie {burst:.0f}s"
                                if ttft == ttft else "geen data")
            grades.append(grade_row)
            notes.append(note_row)
        path = os.path.join(chart_dir, "01_matrix.svg")
        return _draw(
            "heatmap", path, title="Stresstestmatrix: studenten tegen contextgrootte",
            subtitle="groen = bruikbaar, oranje = merkbaar traag, rood = onbruikbaar",
            row_labels=[f"{n} studenten" for n in students],
            column_labels=[f"{c // 1000}k context" for c in contexts],
            grades=grades, annotations=notes)

    def _chart_ttft(self, chart_dir: str) -> list[str]:
        sweep = self._sweep()
        if not sweep:
            return []
        contexts = sorted({r.spec.context_tokens for r in sweep})
        thresholds = dict(self.config.get("grading", {}).get("thresholds") or {})
        series_ttft, series_burst = [], []
        for context in contexts:
            subset = sorted([r for r in sweep if r.spec.context_tokens == context],
                            key=lambda r: r.spec.students)
            series_ttft.append({
                "name": f"{context // 1000}k context",
                "x": [r.spec.students for r in subset],
                "y": [r.aggregate.get("ttft", {}).get("p90", float("nan")) for r in subset],
            })
            series_burst.append({
                "name": f"{context // 1000}k context",
                "x": [r.spec.students for r in subset],
                "y": [r.aggregate.get("burst_duration", {}).get("p90", float("nan"))
                      for r in subset],
            })
        paths: list[str] = []
        path = os.path.join(chart_dir, "02_ttft_vs_studenten.svg")
        paths += _draw(
            "line_chart", path, title="p90 time-to-first-token tegen aantal studenten",
            x_label="gelijktijdige studenten", y_label="p90 TTFT (seconden)",
            series=series_ttft,
            hlines=[{"y": float(thresholds.get("ttft_p90_green_s", 20)),
                     "label": "grens groen", "colour": "#38a169"},
                    {"y": float(thresholds.get("ttft_p90_amber_s", 45)),
                     "label": "grens oranje", "colour": "#c53030"}])
        path = os.path.join(chart_dir, "03_doorlooptijd_instructie.svg")
        paths += _draw(
            "line_chart", path, title="p90 doorlooptijd van een volledige student-instructie",
            x_label="gelijktijdige studenten", y_label="p90 doorlooptijd (seconden)",
            series=series_burst,
            hlines=[{"y": float(thresholds.get("burst_p90_green_s", 90)),
                     "label": "grens groen", "colour": "#38a169"},
                    {"y": float(thresholds.get("burst_p90_amber_s", 180)),
                     "label": "grens oranje", "colour": "#c53030"}])
        return paths

    def _chart_cache_over_time(self, chart_dir: str) -> list[str]:
        candidates = [r for r in self.results if r.metric_rows]
        if not candidates:
            return []
        preferred = ([r for r in candidates if r.spec.kind == "lesson"] or
                     [r for r in candidates if r.spec.run_id == "scen_na_de_stilte"] or
                     [r for r in candidates if r.spec.kind == "sweep" and r.spec.students >= 20] or
                     candidates)
        result = preferred[0]
        rows = result.metric_rows
        series = []
        if any("kv_cache_usage" in row for row in rows):
            series.append({"name": "KV-cachebezetting (%)",
                           "x": [row["t_s"] for row in rows],
                           "y": [row.get("kv_cache_usage", float("nan")) * 100 for row in rows]})
        if any("prefix_cache_hit_rate_window" in row for row in rows):
            series.append({"name": "prefix cache hit rate (%)",
                           "x": [row["t_s"] for row in rows if "prefix_cache_hit_rate_window" in row],
                           "y": [row["prefix_cache_hit_rate_window"] * 100 for row in rows
                                 if "prefix_cache_hit_rate_window" in row]})
        if any("requests_waiting" in row for row in rows):
            series.append({"name": "wachtrijdiepte (verzoeken)",
                           "x": [row["t_s"] for row in rows],
                           "y": [row.get("requests_waiting", float("nan")) for row in rows]})
        if not series:
            return []
        path = os.path.join(chart_dir, "04_cache_en_kv_over_tijd.svg")
        return _draw("line_chart", path,
                     title=f"Cache en geheugen over de tijd -- {result.spec.label}",
                     x_label="seconden sinds start van de run",
                     y_label="procent / aantal", series=series)

    def _chart_preemptions(self, chart_dir: str) -> list[str]:
        scenarios = [r for r in self.results if r.spec.kind in ("scenario", "shared", "engine")]
        if not scenarios:
            scenarios = self.results
        if not scenarios:
            return []
        labels = [r.spec.run_id for r in scenarios]
        values = [float(r.server.get("preemptions") or 0.0) for r in scenarios]
        colours = [svgplot.GRADE_FILL.get(r.grade.colour, "#a0aec0") for r in scenarios]
        path = os.path.join(chart_dir, "05_preempties_per_scenario.svg")
        return _draw("bar_chart", path, title="Preempties per scenario",
                     labels=labels, values=values,
                     y_label="preempties tijdens de meting", colours=colours)

    def _chart_rampup(self, chart_dir: str) -> list[str]:
        ramps = [r for r in self.results if r.ramp and r.ramp.get("steps")]
        if not ramps:
            return []
        result = ramps[0]
        steps = result.ramp["steps"]
        path = os.path.join(chart_dir, "06_klifzoeker.svg")
        return _draw(
            "line_chart", path, title="Klifzoeker: waar breekt het?",
            x_label="gelijktijdige studenten", y_label="seconden",
            series=[
                {"name": "p90 TTFT", "x": [s["students"] for s in steps],
                 "y": [s["ttft_p90_s"] for s in steps]},
                {"name": "p90 doorlooptijd instructie", "x": [s["students"] for s in steps],
                 "y": [s["burst_p90_s"] for s in steps]},
            ],
            hlines=[{"y": 20.0, "label": "TTFT-grens groen", "colour": "#38a169"},
                    {"y": 45.0, "label": "TTFT-grens rood", "colour": "#c53030"}])

    # ------------------------------------------------------------ ascii view

    def matrix_text(self) -> str:
        sweep = self._sweep()
        if not sweep:
            return "(geen sweep-runs in deze resultaten)"
        students = sorted({r.spec.students for r in sweep})
        contexts = sorted({r.spec.context_tokens for r in sweep})
        lookup = {(r.spec.students, r.spec.context_tokens): r for r in sweep}
        header = "studenten \\ context".ljust(20) + "".join(
            f"{c // 1000}k".rjust(16) for c in contexts)
        lines = [header, "-" * len(header)]
        for n in students:
            cells = []
            for context in contexts:
                result = lookup.get((n, context))
                if result is None:
                    cells.append("-".rjust(16))
                    continue
                ttft = result.aggregate.get("ttft", {}).get("p90", float("nan"))
                cells.append(f"{result.grade.colour[:6]} {ttft:5.1f}s".rjust(16))
            lines.append(str(n).ljust(20) + "".join(cells))
        return "\n".join(lines)

    def print_matrix(self) -> None:
        for line in self.matrix_text().splitlines():
            colour = ("green" if "groen" in line else
                      "amber" if "oranje" in line else
                      "red" if "rood" in line else None)
            log(line, color=colour)


# --------------------------------------------------------------------------
# Analysis that answers the four questions
# --------------------------------------------------------------------------

DEFAULT_HARDWARE: dict[str, Any] = {
    "gpu_name": "onbekende GPU",
    "vram_gb": 96.0,
    "gpu_memory_utilization": 0.90,
    "model_weights_gb": 33.0,
    "alternatives": [
        {"name": "RTX PRO 5000 (72 GB)", "vram_gb": 72.0, "price_eur": 7602},
        {"name": "2x RTX 5090 (64 GB)", "vram_gb": 64.0, "price_eur": 10000},
        {"name": "RTX PRO 6000 (96 GB)", "vram_gb": 96.0, "price_eur": 37400},
    ],
}


def resolve_hardware(config: dict) -> dict[str, Any]:
    """Hardware description with defaults filled in.

    The analysis and the written report must agree on these numbers, so both
    go through here rather than reading the raw config. A missing value is
    substituted with the assumption from the budget request, and the report
    says which values it used.
    """
    hardware = dict(DEFAULT_HARDWARE)
    hardware.update({k: v for k, v in (config.get("hardware") or {}).items()
                     if v is not None})
    if not hardware.get("alternatives"):
        hardware["alternatives"] = DEFAULT_HARDWARE["alternatives"]
    return hardware


def analyse(results: Sequence[RunResult], config: dict) -> dict[str, Any]:
    hardware = resolve_hardware(config)
    vram = float(hardware["vram_gb"])
    utilisation = float(hardware["gpu_memory_utilization"])
    weights = float(hardware["model_weights_gb"])
    pool_gb = max(vram * utilisation - weights, 0.1)

    sweep = [r for r in results if r.spec.kind == "sweep"]
    findings: dict[str, Any] = {
        "kv_pool_gb": pool_gb,
        "gpu": hardware.get("gpu_name"),
        "hardware": hardware,
        "hardware_defaults_used": sorted(
            key for key in ("vram_gb", "gpu_memory_utilization", "model_weights_gb")
            if (config.get("hardware") or {}).get(key) is None),
        "runs": len(results),
    }

    # Question 1 -- does a class of twenty fit, and with how much margin?
    class_size = int(config.get("matrix", {}).get("scenarios", {}).get("class_size", 20))
    twenty = [r for r in sweep if r.spec.students == class_size]
    if twenty:
        worst = max(twenty, key=lambda r: {"groen": 0, "oranje": 1, "rood": 2}[r.grade.colour])
        peaks = [r.server.get("kv_cache_usage_peak") for r in twenty
                 if r.server.get("kv_cache_usage_peak") is not None]
        findings["class_size"] = class_size
        findings["class_grades"] = {f"{r.spec.context_tokens // 1000}k": r.grade.colour
                                    for r in twenty}
        findings["class_worst_context"] = worst.spec.context_tokens
        findings["class_worst_grade"] = worst.grade.colour
        if peaks:
            findings["kv_peak_fraction"] = max(peaks)
            findings["kv_peak_gb"] = max(peaks) * pool_gb
            findings["kv_headroom_gb"] = pool_gb - max(peaks) * pool_gb
            findings["kv_headroom_fraction"] = 1.0 - max(peaks)

    # Question 2 -- where is the cliff?
    ramp = next((r for r in results if r.ramp), None)
    if ramp and ramp.ramp:
        findings["cliff_students"] = ramp.ramp.get("max_students_ok")
        findings["cliff_broke_at"] = ramp.ramp.get("broke_at")
        findings["cliff_reasons"] = ramp.ramp.get("reasons")
    first_red = {}
    for context in sorted({r.spec.context_tokens for r in sweep}):
        subset = sorted([r for r in sweep if r.spec.context_tokens == context],
                        key=lambda r: r.spec.students)
        red = next((r.spec.students for r in subset if r.grade.colour == RED), None)
        amber = next((r.spec.students for r in subset if r.grade.colour != GREEN), None)
        first_red[f"{context // 1000}k"] = {"eerste_oranje": amber, "eerste_rood": red}
    findings["cliff_per_context"] = first_red

    # Question 3 -- would less memory do?
    alternatives = []
    peak_fraction = findings.get("kv_peak_fraction")
    for alternative in hardware["alternatives"]:
        alt_pool = float(alternative["vram_gb"]) * utilisation - weights
        entry = {"name": alternative.get("name"), "vram_gb": alternative.get("vram_gb"),
                 "price_eur": alternative.get("price_eur"),
                 "kv_pool_gb": round(alt_pool, 1)}
        if peak_fraction is not None:
            needed = peak_fraction * pool_gb
            entry["needed_kv_gb"] = round(needed, 1)
            entry["fits"] = alt_pool >= needed
            entry["margin_gb"] = round(alt_pool - needed, 1)
            entry["utilisation_on_alternative"] = (round(needed / alt_pool, 3)
                                                   if alt_pool > 0 else None)
        alternatives.append(entry)
    findings["alternatives"] = alternatives

    # Question 4 -- which engine settings matter?
    engine = [r for r in results if r.spec.kind == "engine"]
    if engine:
        findings["engine"] = [{
            "name": r.spec.run_id.replace("engine_", ""),
            "flags": r.spec.tags.get("server_flags"),
            "grade": r.grade.colour,
            "ttft_p90_s": r.aggregate.get("ttft", {}).get("p90"),
            "kv_peak": r.server.get("kv_cache_usage_peak"),
            "preemptions": r.server.get("preemptions"),
            "prefix_cache_hit_rate": r.server.get("prefix_cache_hit_rate"),
        } for r in engine]
        best = min(engine, key=lambda r: (
            {"groen": 0, "oranje": 1, "rood": 2}[r.grade.colour],
            r.aggregate.get("ttft", {}).get("p90") or 1e9))
        findings["engine_best"] = best.spec.run_id.replace("engine_", "")
        findings["engine_best_flags"] = best.spec.tags.get("server_flags")

    # The shared-project-base axis: how much does it actually save?
    shared = [r for r in results if r.spec.kind == "shared"]
    if shared:
        findings["shared"] = [{
            "shared_fraction": r.spec.shared_fraction,
            "context_tokens": r.spec.context_tokens,
            "prefix_cache_hit_rate": r.server.get("prefix_cache_hit_rate"),
            "ttft_p90_s": r.aggregate.get("ttft", {}).get("p90"),
            "kv_peak": r.server.get("kv_cache_usage_peak"),
            "grade": r.grade.colour,
        } for r in sorted(shared, key=lambda r: (r.spec.context_tokens, r.spec.shared_fraction))]

    scenarios = [r for r in results if r.spec.kind == "scenario"]
    findings["scenarios"] = [{
        "run_id": r.spec.run_id, "label": r.spec.label, "grade": r.grade.colour,
        "ttft_p90_s": r.aggregate.get("ttft", {}).get("p90"),
        "burst_p90_s": r.aggregate.get("burst_duration", {}).get("p90"),
        "preemptions": r.server.get("preemptions"),
        "prefix_cache_hit_rate": r.server.get("prefix_cache_hit_rate"),
        "reasons": r.grade.reasons,
    } for r in scenarios]

    lesson = next((r for r in results if r.spec.kind == "lesson"), None)
    if lesson:
        findings["lesson"] = {
            "grade": lesson.grade.colour,
            "ttft_p90_s": lesson.aggregate.get("ttft", {}).get("p90"),
            "burst_p90_s": lesson.aggregate.get("burst_duration", {}).get("p90"),
            "preemptions": lesson.server.get("preemptions"),
            "prefix_cache_hit_rate": lesson.server.get("prefix_cache_hit_rate"),
            "kv_peak": lesson.server.get("kv_cache_usage_peak"),
        }

    disagreements = [r.spec.run_id for r in results
                     if r.grade.colour != r.grade_brief.colour]
    findings["grading_disagreements"] = disagreements
    return findings


def write_analysis(directory: str, results: Sequence[RunResult], config: dict) -> dict:
    findings = analyse(results, config)
    write_json(os.path.join(directory, "analyse.json"), findings)
    return findings


# --------------------------------------------------------------------------
# Reading results back from disk, so charts and RESULTATEN.md can be
# regenerated without re-running anything.
# --------------------------------------------------------------------------

def load_results(directory: str) -> tuple[list[RunResult], dict, dict]:
    from .grading import Grade
    from .metrics import Collector
    from .runspec import Phase, RunSpec

    config_path = os.path.join(directory, "config.json")
    environment_path = os.path.join(directory, "environment.json")
    config = json.load(open(config_path, encoding="utf-8")) if os.path.exists(config_path) else {}
    environment = (json.load(open(environment_path, encoding="utf-8"))
                   if os.path.exists(environment_path) else {})

    runs_dir = os.path.join(directory, "runs")
    results: list[RunResult] = []
    if not os.path.isdir(runs_dir):
        return results, config, environment
    for run_id in sorted(os.listdir(runs_dir)):
        run_json = os.path.join(runs_dir, run_id, "run.json")
        if not os.path.exists(run_json):
            continue
        with open(run_json, encoding="utf-8") as handle:
            payload = json.load(handle)
        spec_data = payload["spec"]
        spec = RunSpec(
            run_id=spec_data["run_id"], label=spec_data["label"], kind=spec_data["kind"],
            students=spec_data["students"], context_tokens=spec_data["context_tokens"],
            activity=spec_data.get("activity", "normaal"),
            shared_fraction=spec_data.get("shared_fraction", 0.5),
            phases=[Phase(**p) for p in spec_data.get("phases", [])],
            arrival_window_s=spec_data.get("arrival_window_s", 60.0),
            expects_shared_prefix=spec_data.get("expects_shared_prefix", True),
            ramp=spec_data.get("ramp"), tags=spec_data.get("tags", {}),
            notes=spec_data.get("notes", ""),
        )
        grade = Grade(**payload["grade"])
        grade_brief = Grade(**payload["grade_brief_definition"])
        metric_rows = _read_csv(os.path.join(runs_dir, run_id, "server_metrics.csv"))
        results.append(RunResult(
            spec=spec, started_wall=0.0, finished_wall=0.0,
            aggregate=payload["aggregate"], server=payload["server"],
            grade=grade, grade_brief=grade_brief,
            composition=payload.get("composition", {}),
            collector=Collector(spec.run_id), metric_rows=metric_rows,
            ramp=payload.get("ramp"),
        ))
    return results, config, environment


def _read_csv(path: str) -> list[dict]:
    import csv
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    rows: list[dict] = []
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict = {}
            for key, value in row.items():
                if value == "" or value is None:
                    continue
                try:
                    converted[key] = float(value)
                except ValueError:
                    converted[key] = value
            rows.append(converted)
    return rows
