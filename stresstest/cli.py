"""Command line interface.

    python -m stresstest doctor        controleer of alles bereikbaar is
    python -m stresstest plan          toon de matrix en de geschatte rekentijd
    python -m stresstest corpus        haal het voorbeeldproject binnen
    python -m stresstest matrix        fase 1: de volledige stresstestmatrix
    python -m stresstest lesson        fase 2: een les van negentig minuten
    python -m stresstest run           een losse run met eigen parameters
    python -m stresstest report DIR    grafieken en RESULTATEN.md opnieuw maken
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
from .tokens import build_counter
from .util import (colored, deep_merge, human_duration, iso, load_jsonc, log)
from .vllm_metrics import MetricsSampler

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
        "hardware": config.get("hardware", {}),
    }


def _largest_context(config: dict) -> int:
    """Largest prompt any run in this configuration will send.

    A run grows to its context target and then adds up to 700 output tokens
    per step, so that headroom is included.
    """
    from .matrix import BUILDERS, PHASE1
    largest = 0
    for group in PHASE1 + ["lesson"]:
        try:
            for spec in BUILDERS[group](config):
                largest = max(largest, spec.context_tokens)
        except Exception:  # noqa: BLE001 - a disabled group must not break doctor
            continue
    return largest + 1024


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
