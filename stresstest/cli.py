"""Command line interface.

    python -m stresstest doctor        controleer of alles bereikbaar is
    python -m stresstest plan          toon de matrix en de geschatte rekentijd
    python -m stresstest corpus        haal het voorbeeldproject binnen
    python -m stresstest matrix        fase 1: de volledige stresstestmatrix
    python -m stresstest lesson        fase 2: een les van negentig minuten
    python -m stresstest run           een losse run met eigen parameters
    python -m stresstest report DIR    grafieken en RESULTATEN.md opnieuw maken
    python -m stresstest monitor       een draaiende server meten zonder zelf last te maken
    python -m stresstest mock          een nep-vLLM om het harnas te testen
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from typing import Any, Sequence

from . import __version__, resultaten
from .client import Endpoint, OpenAIClient
from .corpus import load_corpus
from .matrix import BUILDERS, PHASE1, build_specs, describe_plan
from .report import ResultsWriter, load_results, write_analysis
from .runner import RunEngine, RunResult
from .runspec import RunSpec, standard_phases
from .personas import work_profiles_from_config
from .tokens import build_counter
from .util import (colored, deep_merge, human_duration, iso, load_jsonc, log,
                   write_csv, write_json)
from .vllm_metrics import METRIC_ALIASES, MetricSample, MetricsSampler

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "config", "default.json")


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def load_config(path: str, overrides: Sequence[str] = ()) -> dict:
    config = load_jsonc(path)
    for override in overrides:
        if "=" not in override:
            raise SystemExit(f"--set expects key.path=value, got {override!r}")
        key, _, raw = override.partition("=")
        config = deep_merge(config, _nest(key.strip().split("."), _coerce(raw.strip())))
    return config


def _nest(keys: list[str], value: Any) -> dict:
    out: Any = value
    for key in reversed(keys):
        out = {key: out}
    return out


def _coerce(raw: str) -> Any:
    import json
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none"):
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def make_endpoint(config: dict) -> Endpoint:
    block = config.get("endpoint", {})
    return Endpoint(
        base_url=block.get("base_url", "http://127.0.0.1:8000/v1"),
        api_key=block.get("api_key") or os.environ.get("OPENAI_API_KEY"),
        model=block.get("model", "local-model"),
        timeout_s=float(block.get("timeout_s", 300)),
        connect_timeout_s=float(block.get("connect_timeout_s", 30)),
        verify_tls=bool(block.get("verify_tls", True)),
        extra_headers=block.get("extra_headers", {}) or {},
    )


def build_environment(config: dict, counter, corpus, endpoint: Endpoint) -> dict:
    return {
        "version": __version__,
        "generated": iso(),
        "python": sys.version.split()[0],
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "metrics_url": config.get("endpoint", {}).get("metrics_url"),
        "tokenizer": counter.describe(),
        "tokenizer_exact": counter.exact,
        "corpus": corpus.describe(),
        "seed": config.get("seed"),
        # The work profiles decide how heavy one agent step is, which turned
        # out to be the assumption the conclusion is most sensitive to. They
        # belong with the results, not only in the config next to them.
        "work_profiles": [w.to_dict() for w in
                          work_profiles_from_config(
                              config.get("behaviour", {}).get("work_profiles"))],
        "hardware": config.get("hardware", {}),
    }


def _largest_context(config: dict) -> int:
    """Largest prompt any run in this configuration will send.

    A run grows to its context target and then adds the model's own output on
    top, so the heaviest work profile's closing answer is the headroom that
    has to fit. With a heavy profile that is thousands of tokens, not the
    couple of hundred this used to assume.
    """
    from .matrix import BUILDERS, PHASE1
    largest = 0
    for group in PHASE1 + ["lesson"]:
        try:
            for spec in BUILDERS[group](config):
                largest = max(largest, spec.context_tokens)
        except Exception:  # noqa: BLE001 - a disabled group must not break doctor
            continue
    profiles = work_profiles_from_config(config.get("behaviour", {}).get("work_profiles"))
    headroom = max((int(w.output_tokens_final[1]) for w in profiles), default=700)
    return largest + headroom + 1024


def _round_up(value: int, step: int = 8192) -> int:
    return ((value + step - 1) // step) * step


def _number(value: Any, suffix: str = "", digits: int = 1) -> str:
    if value is None or value != value:
        return "n/b"
    return f"{value:.{digits}f}{suffix}"


def results_directory(config: dict, explicit: str | None, tag: str) -> str:
    if explicit:
        return explicit
    base = config.get("output", {}).get("directory", "results")
    return os.path.join(base, f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}")


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_plan(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    groups = args.only or PHASE1
    specs = build_specs(config, groups)
    gap = float(config.get("run_defaults", {}).get("gap_between_runs_s", 30))
    plan = describe_plan(specs, gap)
    print(f"Configuratie: {args.config}")
    print(f"Groepen:      {', '.join(groups)}")
    print(f"Runs:         {plan['runs']}")
    print(f"Rekentijd:    {human_duration(plan['total_seconds'])} "
          f"(inclusief {gap:.0f}s pauze tussen runs)")
    print()
    print(f"{'run_id':32} {'studenten':>9} {'context':>9} {'activiteit':>11} "
          f"{'gedeeld':>8} {'duur':>8}")
    print("-" * 84)
    for spec in specs:
        print(f"{spec.run_id:32} {spec.students:>9} "
              f"{spec.context_tokens // 1000:>8}k {spec.activity:>11} "
              f"{spec.shared_fraction:>7.0%} "
              f"{human_duration(spec.total_duration_s):>8}")
    print("-" * 84)
    for kind, entry in plan["by_kind"].items():
        print(f"{kind:>12}: {entry['runs']:>2} runs, "
              f"{human_duration(entry['seconds'])}")
    hourly = args.price_per_hour
    if hourly:
        print(f"\nGeschatte huurkosten: "
              f"${plan['total_seconds'] / 3600 * hourly:.2f} bij ${hourly:.2f}/uur")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    corpus = load_corpus(config.get("corpus", {}))
    description = corpus.describe()
    log(f"corpus '{description['name']}': {description['files']} bestanden, "
        f"{description['shared_files']} in het gedeelde skelet, "
        f"{description['total_characters']:,} tekens", color="bold")
    counter = _counter(config)
    log(f"tokenizer: {counter.describe()}")
    if counter.exact:
        from .tokens import calibrate_ratio
        ratio = calibrate_ratio(counter, [f.content for f in corpus.files[:40]])
        if ratio:
            log(f"gemeten verhouding op dit corpus: {ratio:.2f} tekens per token "
                f"(schatting gebruikt {config.get('tokenizer', {}).get('chars_per_token', 3.5)})")
    return 0


def _counter(config: dict):
    block = config.get("tokenizer", {})
    return build_counter(
        config.get("endpoint", {}).get("model", "local-model"),
        chars_per_token=float(block.get("chars_per_token", 3.5)),
        prefer_exact=bool(block.get("prefer_exact", True)),
        tokenizer_path=block.get("path"),
    )


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Measure characters per token against this corpus and this model.

    Only useful when an exact tokenizer is available -- the point is to make
    the fallback estimate agree with it, so that a colleague who cannot
    install `tokenizers` still gets context sizes that are right.
    """
    import random

    from .conversation import TOOLS, Session
    from .tokens import TokenCounter, best_ratio, calibrate_ratio

    config = load_config(args.config, args.set)
    exact = _counter(config)
    if not exact.exact:
        print("Geen exacte tokenizer beschikbaar; er valt niets te ijken.")
        print("Installeer `tokenizers` (pip install tokenizers) en probeer opnieuw.")
        return 1
    corpus = load_corpus(config.get("corpus", {}), quiet=True)
    print(f"tokenizer : {exact.describe()}")
    print(f"corpus    : {corpus.describe()['files']} bestanden, "
          f"{corpus.describe()['total_characters']:,} tekens")

    raw = calibrate_ratio(exact, [f.content for f in corpus.files])
    print(f"\nRuwe broncode      : {raw:.2f} tekens per token")

    targets = args.contexts or sorted({
        spec.context_tokens for spec in build_specs(config, ["sweep"])}) or [32000]

    def errors_for(ratio: float) -> list[float]:
        estimate = TokenCounter("estimate", "calibratie", ratio)
        out = []
        for target in targets:
            for fraction in (0.0, 0.9):
                session = Session(0, corpus, estimate, target, fraction, random.Random(0))
                session.reset()
                measured = (exact.count_messages(session.messages)
                            + exact.count_tools(TOOLS))
                if measured:
                    out.append((session.initial_tokens - measured) / measured)
        return out

    ratio, worst = best_ratio(errors_for)
    current = float(config.get("tokenizer", {}).get("chars_per_token", 4.6))
    current_worst = max(abs(e) for e in errors_for(current))

    print(f"Volledige berichten: {ratio:.2f} tekens per token "
          f"(grootste afwijking {worst:.1%})")
    print(f"Huidige instelling : {current:.2f} "
          f"(grootste afwijking {current_worst:.1%})")
    print()
    if abs(ratio - current) < 0.05:
        print("De huidige instelling klopt; er hoeft niets te veranderen.")
    else:
        print("Zet dit in je configuratie:")
        print(f'    "tokenizer": {{ "chars_per_token": {ratio:.2f} }}')
        print()
        print("Of eenmalig op de opdrachtregel:")
        print(f"    python3 -m stresstest matrix --set tokenizer.chars_per_token={ratio:.2f}")
    print()
    print("Let op: dit ijkt alleen de schatting die gebruikt wordt als er geen")
    print("tokenizer beschikbaar is. Zolang `tokenizers` geinstalleerd is, telt")
    print("het harnas exact en doet deze waarde er niet toe.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    endpoint = make_endpoint(config)
    metrics_url = config.get("endpoint", {}).get("metrics_url")
    ok = True

    print(f"harnas       : versie {__version__}, Python {sys.version.split()[0]}")
    counter = _counter(config)
    print(f"tokenizer    : {counter.describe()}")
    if not counter.exact:
        print("               LET OP: de contextgroottes zijn een SCHATTING.")
        print("               De contextgrootte is een as van de hele matrix, dus een")
        print("               systematische fout hierin verschuift de conclusie over")
        print("               hoeveel geheugen een klas nodig heeft.")
        try:
            import tokenizers  # noqa: F401
            installed = True
        except Exception:  # noqa: BLE001 - any import failure means "not usable"
            installed = False
        if installed:
            # The package is there, so the fallback is a wiring problem, not a
            # missing dependency: the tokenizer was looked up under
            # `endpoint.model`, which is vLLM's --served-model-name and not a
            # HuggingFace repo. Telling the operator to pip install something
            # they already have sends them down the wrong path.
            source = (config.get("tokenizer", {}).get("path")
                      or config.get("endpoint", {}).get("model"))
            print("               `tokenizers` is wel geinstalleerd, maar er is geen")
            print(f"               tokenizer gevonden onder '{source}'.")
            print("               Zet `tokenizer.path` in de config op de map van het")
            print("               model of op de repo-id, bijvoorbeeld:")
            print("                 --set tokenizer.path=Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
        else:
            print("               Los dit op met:  pip install tokenizers")
            print("               Kan dat niet, ijk dan eerst: stresstest calibrate")

    try:
        corpus = load_corpus(config.get("corpus", {}), quiet=True)
        description = corpus.describe()
        print(f"corpus       : {description['name']}, {description['files']} bestanden")
    except Exception as exc:  # noqa: BLE001
        print(f"corpus       : MISLUKT -- {exc}")
        ok = False

    async def probe() -> None:
        nonlocal ok
        client = OpenAIClient(endpoint)
        info = await client.probe()
        if info.get("ok"):
            print(f"endpoint     : {endpoint.base_url} bereikbaar, "
                  f"modellen: {', '.join(str(m) for m in info.get('models', []))}")
            if endpoint.model not in (info.get("models") or []):
                print(f"               LET OP: geconfigureerd model {endpoint.model!r} "
                      f"staat niet in die lijst")
            window = info.get("max_model_len")
            if window:
                needed = _largest_context(config)
                print(f"contextvenster: {window:,} tokens "
                      f"(zwaarste run in deze matrix vraagt ~{needed:,})")
                if needed > window:
                    ok = False
                    print(f"               TE KLEIN: runs boven {window:,} tokens "
                          f"worden geweigerd. Start vLLM met "
                          f"--max-model-len {_round_up(needed)} of haal de "
                          f"grootste contextwaarden uit de matrix.")
        else:
            print(f"endpoint     : NIET BEREIKBAAR -- {info}")
            ok = False
            return
        result = await client.chat_stream(
            [{"role": "user", "content": "Zeg alleen: ok"}], max_tokens=8,
            extra_body={"ignore_eos": True} if config.get("request", {}).get("ignore_eos", True) else None)
        if result.ok:
            print(f"testverzoek  : ok, TTFT {result.ttft_s:.2f}s, "
                  f"{result.output_tokens} tokens, "
                  f"{result.decode_tokens_per_s:.1f} tok/s")
        else:
            print(f"testverzoek  : MISLUKT -- {result.error}")
            ok = False
        sampler = MetricsSampler(metrics_url, verify_tls=endpoint.verify_tls)
        if await sampler.check():
            sample = await sampler._sample_once()
            found = ", ".join(sorted((sample.values if sample else {}).keys()))
            print(f"metrics      : {metrics_url} ok -- {found}")
        else:
            print(f"metrics      : NIET BRUIKBAAR -- {sampler.error}")
            print("               De test draait door, maar zonder preempties, "
                  "cache hit rate en KV-bezetting is de hoofdvraag niet te beantwoorden.")

    asyncio.run(probe())
    print()
    print("gereed" if ok else "er zijn problemen -- zie hierboven")
    return 0 if ok else 1


async def _execute(specs: list[RunSpec], config: dict, directory: str,
                   pause_for_engine: bool) -> list[RunResult]:
    endpoint = make_endpoint(config)
    counter = _counter(config)
    corpus = load_corpus(config.get("corpus", {}))
    client = OpenAIClient(endpoint)
    environment = build_environment(config, counter, corpus, endpoint)
    writer = ResultsWriter(directory, config, environment)
    engine = RunEngine(client=client, corpus=corpus, counter=counter, config=config)

    metrics_url = config.get("endpoint", {}).get("metrics_url")
    interval = float(config.get("run_defaults", {}).get("metrics_interval_s", 2))
    gap = float(config.get("run_defaults", {}).get("gap_between_runs_s", 30))

    log(f"resultaten -> {directory}", color="bold")
    log(f"tokenizer: {counter.describe()} | corpus: {corpus.describe()['files']} bestanden")
    if not counter.exact:
        log("LET OP: contextgroottes zijn geschat, niet exact geteld. "
            "Installeer `tokenizers` of draai eerst `stresstest calibrate`.",
            color="amber")
    total = len(specs)
    started = time.monotonic()

    for index, spec in enumerate(specs, start=1):
        if spec.kind == "engine" and pause_for_engine:
            flags = spec.tags.get("server_flags", "")
            print()
            print(colored(f"Run {spec.run_id} vereist een herstart van vLLM met:", "bold"))
            print(f"    {flags}")
            print(f"    ({spec.notes})")
            try:
                answer = input("Herstart vLLM en druk op enter, of 's' om over te slaan: ")
            except EOFError:
                answer = "s"
            if answer.strip().lower().startswith("s"):
                log(f"{spec.run_id} overgeslagen", color="amber")
                continue

        remaining = sum(s.total_duration_s + gap for s in specs[index - 1:])
        log(f"[{index}/{total}] {spec.run_id} -- {spec.label} "
            f"({human_duration(spec.total_duration_s)}, nog "
            f"{human_duration(remaining)} te gaan)", color="bold")
        sampler = MetricsSampler(metrics_url, interval_s=interval,
                                 verify_tls=endpoint.verify_tls)
        result = await engine.run(spec, sampler)
        writer.add(result)
        colour = {"groen": "green", "oranje": "amber", "rood": "red"}.get(result.grade.colour)
        aggregate = result.aggregate
        bits = [
            f"{result.grade.colour.upper():6}",
            f"p90 TTFT {_number(aggregate.get('ttft', {}).get('p90'), 's')}",
            f"p90 instructie {_number(aggregate.get('burst_duration', {}).get('p90'), 's', 0)}",
            f"{aggregate.get('requests_total', 0)} verzoeken",
            f"{aggregate.get('bursts_completed', 0)} instructies",
        ]
        if result.server.get("preemptions") is not None:
            bits.append(f"preempties {result.server['preemptions']:.0f}")
        if result.server.get("prefix_cache_hit_rate") is not None:
            bits.append(f"cache {result.server['prefix_cache_hit_rate']:.0%}")
        if result.server.get("kv_cache_usage_peak") is not None:
            bits.append(f"KV-piek {result.server['kv_cache_usage_peak']:.0%}")
        log("    " + " | ".join(bits), color=colour)
        for reason in result.grade.reasons:
            log(f"      - {reason}", color=colour)
        for warning in result.grade.warnings:
            log(f"      ! {warning}", color="amber")
        if index < total:
            await asyncio.sleep(gap)

    elapsed = time.monotonic() - started
    log(f"klaar in {human_duration(elapsed)}", color="bold")
    writer.print_matrix()
    charts = writer.write_charts()
    write_analysis(directory, writer.results, config)
    path = resultaten.write(directory, writer.results, config, environment)
    log(f"grafieken: {len(charts)} bestanden in {os.path.join(directory, 'charts')}")
    log(f"conclusie: {path}", color="bold")
    return writer.results


def cmd_matrix(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    groups = args.only or PHASE1
    specs = build_specs(config, groups)
    if args.run:
        wanted = set(args.run)
        specs = [s for s in specs if s.run_id in wanted]
        if not specs:
            raise SystemExit(f"geen runs met id in {sorted(wanted)}")
    if args.skip:
        specs = [s for s in specs if s.run_id not in set(args.skip)]
    if args.dry_run:
        args.only = groups
        return cmd_plan(args)
    directory = results_directory(config, args.out, "matrix")
    asyncio.run(_execute(specs, config, directory, pause_for_engine=not args.no_pause))
    return 0


def cmd_lesson(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    specs = build_specs(config, ["lesson"])
    if args.dry_run:
        args.only = ["lesson"]
        return cmd_plan(args)
    directory = results_directory(config, args.out, "les")
    asyncio.run(_execute(specs, config, directory, pause_for_engine=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    defaults = config.get("run_defaults", {})
    spec = RunSpec(
        run_id=args.id or f"run_s{args.students}_c{args.context // 1000}k",
        label=f"{args.students} studenten, {args.context // 1000}k context, {args.activity}",
        kind="sweep", students=args.students, context_tokens=args.context,
        activity=args.activity, shared_fraction=args.shared,
        phases=standard_phases(args.warmup if args.warmup is not None
                               else float(defaults.get("warmup_s", 90)),
                               args.measure if args.measure is not None
                               else float(defaults.get("measure_s", 300))),
        arrival_window_s=float(defaults.get("arrival_window_s", 60)),
        expects_shared_prefix=args.shared >= 0.4,
    )
    directory = results_directory(config, args.out, spec.run_id)
    asyncio.run(_execute([spec], config, directory, pause_for_engine=False))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    results, config, environment = load_results(args.directory)
    if not results:
        raise SystemExit(f"geen runs gevonden in {args.directory}")

    # Fase 1 and fase 2 land in separate directories, so neither one on its own
    # holds the whole answer. --also folds extra directories in; the hardware
    # has to match, because a report that silently mixes two cards is worse
    # than two reports that each cover half.
    for extra in getattr(args, "also", None) or []:
        more, other_config, _ = load_results(extra)
        if not more:
            raise SystemExit(f"geen runs gevonden in {extra}")
        here = (config.get("hardware") or {}).get("gpu_name")
        there = (other_config.get("hardware") or {}).get("gpu_name")
        if here and there and here.split(" (")[0] != there.split(" (")[0]:
            raise SystemExit(
                f"{extra} is gemeten op '{there}' en {args.directory} op '{here}'. "
                f"Resultaten van twee kaarten horen niet in een rapport.")
        known = {r.spec.run_id for r in results}
        added = [r for r in more if r.spec.run_id not in known]
        if len(added) < len(more):
            log(f"{extra}: {len(more) - len(added)} runs overgeslagen, "
                f"die run_id staat al in {args.directory}", color="amber")
        results += added

    if args.out:
        # Writing elsewhere leaves the measurement directories untouched: their
        # summary.csv must keep describing the runs that directory contains.
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(resultaten.render(
                results, config, environment,
                sources=[args.directory, *(getattr(args, "also", None) or [])]))
        log(f"{len(results)} runs, conclusie in {path}", color="bold")
        return 0

    writer = ResultsWriter(args.directory, config, environment)
    writer.results = results
    writer.flush_summary()
    writer.print_matrix()
    charts = writer.write_charts()
    write_analysis(args.directory, results, config)
    path = resultaten.write(args.directory, results, config, environment)
    log(f"{len(results)} runs, {len(charts)} grafieken, conclusie in {path}", color="bold")
    return 0


# --------------------------------------------------------------------------
# monitor: measuring a server that somebody else is using
# --------------------------------------------------------------------------
#
# Every other command in this harness *makes* the load it measures. `monitor`
# does not: it watches a vLLM that a real class is hammering through OpenCode
# or anything else, and writes the same `server_metrics.csv` the runs write, so
# a real lesson can be plotted on the same axes as `les_90min`.
#
# It is written for a machine that gets unplugged. Every sample is flushed to
# disk as it arrives, so a lost SSH session, a killed pod or a Ctrl-C at the
# wrong moment costs you the last two seconds and nothing more.

# A superset of the run columns: whichever metrics this vLLM does not expose
# simply stay empty, and the column list never depends on what the first sample
# happened to contain.
MONITOR_COLUMNS = (["t_s", "t_wall"] + list(METRIC_ALIASES)
                   + ["prefix_cache_hit_rate_window"])


class _MetricsCsv:
    """Append-as-you-go writer for the live time series."""

    def __init__(self, path: str) -> None:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._handle = open(path, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=MONITOR_COLUMNS,
                                      extrasaction="ignore", restval="",
                                      lineterminator="\n")
        self._writer.writeheader()
        self._handle.flush()
        self.origin: float | None = None
        self.rows = 0
        self._previous: MetricSample | None = None

    def add(self, sample: MetricSample) -> None:
        if self.origin is None:
            self.origin = sample.t_monotonic
        row: dict[str, Any] = {"t_s": round(sample.t_monotonic - self.origin, 2),
                               "t_wall": sample.t_wall}
        row.update(sample.values)
        if self._previous is not None:
            queries = (sample.values.get("prefix_cache_queries", 0.0)
                       - self._previous.values.get("prefix_cache_queries", 0.0))
            hits = (sample.values.get("prefix_cache_hits", 0.0)
                    - self._previous.values.get("prefix_cache_hits", 0.0))
            if queries > 0:
                row["prefix_cache_hit_rate_window"] = hits / queries
        self._previous = sample
        self._writer.writerow(row)
        # Flushing every row is what makes this survive a kill -9. Two seconds
        # of buffering would be cheaper and would lose the whole lesson.
        self._handle.flush()
        self.rows += 1

    def close(self) -> None:
        try:
            self._handle.close()
        except Exception:  # noqa: BLE001
            pass


def _monitor_status(sampler: MetricsSampler, pool_gb: float | None) -> str:
    if not sampler.samples:
        return "nog geen monsters"
    latest = sampler.samples[-1]
    parts = []
    running = latest.values.get("requests_running")
    waiting = latest.values.get("requests_waiting")
    if running is not None:
        parts.append(f"draaiend {running:.0f}")
    if waiting is not None:
        parts.append(f"wachtrij {waiting:.0f}")
    usage = latest.values.get("kv_cache_usage")
    if usage is not None:
        if pool_gb:
            parts.append(f"KV {usage * 100:.0f}% ({usage * pool_gb:.1f} GB)")
        else:
            parts.append(f"KV {usage * 100:.0f}%")
    first = sampler.samples[0]
    window = sampler.summarize(first.t_monotonic, latest.t_monotonic)
    rate = window.get("prefix_cache_hit_rate")
    if rate is not None:
        parts.append(f"hit rate {rate * 100:.0f}%")
    preemptions = window.get("preemptions")
    if preemptions:
        parts.append(colored(f"preempties {preemptions:.0f}", "red"))
    return " \u00b7 ".join(parts) or "geen bruikbare meetwaarden"


async def _monitor(args: argparse.Namespace, config: dict, url: str,
                   directory: str) -> int:
    from .report import resolve_hardware

    hardware = resolve_hardware(config)
    try:
        pool_gb = max(float(hardware["vram_gb"]) * float(hardware["gpu_memory_utilization"])
                      - float(hardware["model_weights_gb"]), 0.1)
    except (KeyError, TypeError, ValueError):
        pool_gb = None

    csv_path = os.path.join(directory, "server_metrics.csv")
    writer = _MetricsCsv(csv_path)
    sampler = MetricsSampler(url, interval_s=args.interval,
                             verify_tls=config.get("endpoint", {}).get("verify_tls", True),
                             on_sample=writer.add)

    if not await sampler.check():
        writer.close()
        log(f"kan {url} niet uitlezen: {sampler.error}", color="red")
        log("draait vLLM, en klopt endpoint.metrics_url? Probeer: "
            f"curl -s {url} | head", color="amber")
        return 2

    started_wall = iso()
    log(f"meten van {url} elke {args.interval:g}s")
    log(f"schrijft naar {csv_path}")
    if args.duration:
        log(f"stopt vanzelf na {human_duration(args.duration)}; "
            "Ctrl-C stopt eerder en bewaart alles")
    else:
        log("Ctrl-C stopt de meting en schrijft de samenvatting")

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        try:
            import signal as signal_module
            loop.add_signal_handler(getattr(signal_module, signal_name), stop.set)
        except (AttributeError, NotImplementedError, RuntimeError, ValueError):
            # Windows, and any loop that will not take handlers: Ctrl-C then
            # arrives as KeyboardInterrupt, which main() already catches.
            pass

    await sampler.start()
    began = time.monotonic()
    deadline = began + args.duration if args.duration else None
    try:
        while not stop.is_set():
            # Wake for the status line, but never sleep past the deadline: with
            # a status interval of half a minute, waiting for the next tick
            # would overshoot a fixed duration by up to that much.
            timeout = args.status_interval
            if deadline is not None:
                timeout = min(timeout, max(deadline - time.monotonic(), 0.0))
            try:
                await asyncio.wait_for(stop.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
            if stop.is_set():
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            log(f"[{human_duration(time.monotonic() - began)}] "
                f"{_monitor_status(sampler, pool_gb)}")
    except KeyboardInterrupt:
        pass
    finally:
        await sampler.stop()
        writer.close()

    if len(sampler.samples) < 2:
        log("te weinig monsters voor een samenvatting; de CSV staat er wel.", color="amber")
        return 1

    first, last = sampler.samples[0], sampler.samples[-1]
    summary = sampler.summarize(first.t_monotonic, last.t_monotonic)
    peak = summary.get("kv_cache_usage_peak")
    payload = {
        "kind": "monitor",
        "label": args.label,
        "harness_version": __version__,
        "metrics_url": url,
        "interval_s": args.interval,
        "started": started_wall,
        "finished": iso(),
        "duration_s": round(last.t_monotonic - first.t_monotonic, 1),
        "samples": len(sampler.samples),
        "kv_pool_gb": round(pool_gb, 2) if pool_gb else None,
        "kv_peak_gb": round(peak * pool_gb, 2) if (peak is not None and pool_gb) else None,
        "hardware": hardware,
        "server": summary,
        "metric_names": dict(last.raw_names),
    }
    write_json(os.path.join(directory, "monitor.json"), payload)

    log("")
    log(f"gemeten: {human_duration(payload['duration_s'])}, "
        f"{payload['samples']} monsters, {writer.rows} regels")
    _print_monitor_summary(summary, pool_gb)
    log("")
    log(f"tijdreeks    : {csv_path}")
    log(f"samenvatting : {os.path.join(directory, 'monitor.json')}")
    return 0


def _print_monitor_summary(summary: dict, pool_gb: float | None) -> None:
    def pct(value: Any) -> str:
        return "niet gemeten" if value is None else f"{float(value) * 100:.1f}%"

    peak = summary.get("kv_cache_usage_peak")
    gb = f" ({float(peak) * pool_gb:.1f} GB van {pool_gb:.1f} GB)" \
        if (peak is not None and pool_gb) else ""
    preemptions = summary.get("preemptions")
    print(f"  KV-bezetting      gem {pct(summary.get('kv_cache_usage_avg'))}, "
          f"piek {pct(peak)}{gb}")
    print(f"  wachtrij          gem {summary.get('queue_depth_avg', 0) or 0:.2f}, "
          f"piek {summary.get('queue_depth_peak', 0) or 0:.0f}")
    print(f"  gelijktijdig      gem {summary.get('requests_running_avg', 0) or 0:.2f}, "
          f"piek {summary.get('requests_running_peak', 0) or 0:.0f}")
    rate = summary.get("prefix_cache_hit_rate")
    print(f"  prefix cache      hit rate {pct(rate)}")
    print(f"  preempties        {0 if preemptions is None else int(preemptions)}")
    print(f"  doorvoer          {summary.get('server_prefill_tokens_per_s', 0) or 0:.0f} tok/s prefill, "
          f"{summary.get('server_decode_tokens_per_s', 0) or 0:.1f} tok/s decode")
    if preemptions:
        print("  LET OP: er is gepreempt. De cachepool liep vol; sessies moesten")
        print("          hun hele context opnieuw laten voorrekenen.")


def cmd_monitor(args: argparse.Namespace) -> int:
    config = load_config(args.config, args.set)
    url = args.url or config.get("endpoint", {}).get("metrics_url")
    if not url:
        log("geen metrics-URL: zet endpoint.metrics_url in de configuratie of "
            "geef --url mee.", color="red")
        return 2
    directory = args.out or os.path.join(
        config.get("output", {}).get("directory", "results"),
        f"{time.strftime('%Y%m%d-%H%M%S')}_monitor")
    os.makedirs(directory, exist_ok=True)
    return asyncio.run(_monitor(args, config, url, directory))


def cmd_mock(args: argparse.Namespace) -> int:
    from .mockserver import main as mock_main
    rest = list(args.rest)
    if rest and rest[0] == "--":      # argparse.REMAINDER keeps the separator
        rest = rest[1:]
    return mock_main(rest)


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stresstest",
        description="Stresstest agentic coding voor een lokaal draaiend LLM.")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("-c", "--config", default=DEFAULT_CONFIG,
                         help="pad naar het configuratiebestand")
        sub.add_argument("--set", action="append", default=[], metavar="SLEUTEL=WAARDE",
                         help="overschrijf een configuratiewaarde, bv. "
                              "--set endpoint.base_url=http://host:8000/v1")

    plan = subparsers.add_parser("plan", help="toon de matrix en de geschatte rekentijd")
    common(plan)
    plan.add_argument("--only", nargs="+", choices=list(BUILDERS),
                      help="alleen deze groepen")
    plan.add_argument("--price-per-hour", type=float, default=None,
                      help="reken de huurkosten uit tegen dit uurtarief in dollar")
    plan.set_defaults(func=cmd_plan)

    calibrate = subparsers.add_parser(
        "calibrate", help="ijk de tokenschatting tegen dit corpus en dit model")
    common(calibrate)
    calibrate.add_argument("--contexts", nargs="+", type=int,
                           help="contextgroottes om tegen te ijken "
                                "(standaard: die uit de sweep)")
    calibrate.set_defaults(func=cmd_calibrate)

    doctor = subparsers.add_parser("doctor", help="controleer endpoint, metrics, corpus")
    common(doctor)
    doctor.set_defaults(func=cmd_doctor)

    corpus = subparsers.add_parser("corpus", help="haal het voorbeeldproject binnen")
    common(corpus)
    corpus.set_defaults(func=cmd_corpus)

    matrix = subparsers.add_parser("matrix", help="fase 1: de volledige stresstestmatrix")
    common(matrix)
    matrix.add_argument("--only", nargs="+", choices=list(BUILDERS),
                        help="alleen deze groepen draaien")
    matrix.add_argument("--run", nargs="+", help="alleen deze run-ids")
    matrix.add_argument("--skip", nargs="+", help="deze run-ids overslaan")
    matrix.add_argument("--out", help="resultatenmap (standaard results/<tijdstempel>)")
    matrix.add_argument("--dry-run", action="store_true", help="toon alleen het plan")
    matrix.add_argument("--no-pause", action="store_true",
                        help="niet wachten op een handmatige vLLM-herstart bij "
                             "de engine-varianten")
    matrix.set_defaults(func=cmd_matrix, price_per_hour=None)

    lesson = subparsers.add_parser("lesson", help="fase 2: lesvalidatie van negentig minuten")
    common(lesson)
    lesson.add_argument("--out")
    lesson.add_argument("--dry-run", action="store_true")
    lesson.set_defaults(func=cmd_lesson, price_per_hour=None)

    run = subparsers.add_parser("run", help="een losse run met eigen parameters")
    common(run)
    run.add_argument("--students", type=int, default=20)
    run.add_argument("--context", type=int, default=32000)
    run.add_argument("--activity", default="normaal",
                     choices=["rustig", "normaal", "intensief"])
    run.add_argument("--shared", type=float, default=0.5,
                     help="aandeel gedeelde projectbasis, 0.0 tot 1.0")
    run.add_argument("--warmup", type=float, default=None)
    run.add_argument("--measure", type=float, default=None)
    run.add_argument("--id", help="eigen run-id")
    run.add_argument("--out")
    run.set_defaults(func=cmd_run)

    report = subparsers.add_parser("report",
                                   help="grafieken en RESULTATEN.md opnieuw maken")
    report.add_argument("directory", help="een resultatenmap")
    report.add_argument("--also", nargs="+", metavar="MAP", default=[],
                        help="extra resultatenmappen van dezelfde kaart om mee te "
                             "nemen, bijvoorbeeld de lesvalidatie naast de matrix")
    report.add_argument("--out", metavar="PAD",
                        help="schrijf RESULTATEN.md hierheen in plaats van in de "
                             "resultatenmap; de bronmappen blijven dan ongemoeid")
    report.set_defaults(func=cmd_report)

    monitor = subparsers.add_parser(
        "monitor",
        help="meet een draaiende server zonder zelf belasting te maken")
    common(monitor)
    monitor.add_argument("--out", metavar="MAP",
                         help="waar server_metrics.csv en monitor.json komen "
                              "(standaard results/<tijdstempel>_monitor)")
    monitor.add_argument("--url", metavar="URL",
                         help="metrics-endpoint; standaard endpoint.metrics_url")
    monitor.add_argument("--interval", type=float, default=2.0, metavar="SECONDEN",
                         help="hoe vaak /metrics wordt uitgelezen (standaard 2)")
    monitor.add_argument("--duration", type=float, default=None, metavar="SECONDEN",
                         help="stop vanzelf na zoveel seconden; standaard tot Ctrl-C")
    monitor.add_argument("--status-interval", type=float, default=30.0, metavar="SECONDEN",
                         help="hoe vaak een statusregel wordt getoond (standaard 30)")
    monitor.add_argument("--label", default="", metavar="TEKST",
                         help="waar deze meting over gaat, bv. 'les 3H woensdag'")
    monitor.set_defaults(func=cmd_monitor)

    mock = subparsers.add_parser("mock", help="start een nep-vLLM om het harnas te testen")
    mock.add_argument("rest", nargs=argparse.REMAINDER,
                      help="argumenten voor de mockserver, zie --help daarvan")
    mock.set_defaults(func=cmd_mock)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        log("afgebroken door gebruiker", color="amber")
        return 130
