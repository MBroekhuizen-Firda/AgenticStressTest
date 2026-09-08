"""Generating RESULTATEN.md.

Written for whoever has to judge the budget request, not for an engineer:
the four questions from the assignment, answered in plain Dutch, with the
measurements underneath so the answer can be checked.
"""

from __future__ import annotations

import os
from typing import Any, Sequence

from .grading import AMBER, GREEN, RED
from .report import analyse, resolve_hardware
from .runner import RunResult
from .util import iso

SEVERITY = {GREEN: 0, AMBER: 1, RED: 2}


def _n(value: Any, suffix: str = "", digits: int = 1) -> str:
    if value is None:
        return "niet gemeten"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "niet gemeten"
    return f"{number:.{digits}f}{suffix}"


def _pct(value: Any) -> str:
    if value is None:
        return "niet gemeten"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "niet gemeten"
    return f"{number * 100:.0f}%"


def _verdict_line(colour: str) -> str:
    return {GREEN: "**Ja.**", AMBER: "**Met kanttekeningen.**",
            RED: "**Nee.**"}.get(colour, "**Onbekend.**")


def render(results: Sequence[RunResult], config: dict, environment: dict) -> str:
    findings = analyse(results, config)
    hardware = resolve_hardware(config)
    class_size = findings.get("class_size", 20)
    lines: list[str] = []
    add = lines.append

    add("# Resultaten stresstest agentic coding")
    add("")
    add(f"Gemeten op **{hardware.get('gpu_name', 'onbekende GPU')}** "
        f"({_n(hardware.get('vram_gb'), ' GB', 0)} videogeheugen) met "
        f"**{environment.get('model', 'onbekend model')}** via vLLM.")
    add(f"Uitgevoerd op {iso()[:10]}, {findings.get('runs', 0)} runs, "
        f"seed {config.get('seed')}.")
    add("")
    add("> Dit bestand is automatisch gegenereerd uit de meetgegevens in deze map. "
        "De onderliggende getallen staan in `summary.csv`; per run staan alle "
        "afzonderlijke verzoeken in `runs/<run_id>/requests.csv`.")
    add("")

    # ---------------------------------------------------------------- kern
    add("## De korte versie")
    add("")
    cliff = findings.get("cliff_students")
    class_grade = findings.get("class_worst_grade")
    if class_grade:
        add(f"- Een klas van {class_size} studenten komt in de zwaarste geteste "
            f"contextgrootte uit op **{class_grade}**.")
    if cliff:
        broke = findings.get("cliff_broke_at")
        add(f"- De klif ligt bij **{cliff} gelijktijdige studenten**"
            + (f"; bij {broke} braken de drempels." if broke else
               " (de test liep tot het ingestelde maximum zonder te breken)."))
    if findings.get("kv_peak_fraction") is not None:
        add(f"- Piekgebruik van de KV-cache: **{_pct(findings['kv_peak_fraction'])}** "
            f"van de {_n(findings['kv_pool_gb'], ' GB')} cachepool, oftewel "
            f"**{_n(findings.get('kv_peak_gb'), ' GB')}** in gebruik en "
            f"**{_n(findings.get('kv_headroom_gb'), ' GB')}** over.")
    fitting = [a for a in findings.get("alternatives", []) if a.get("fits")]
    if fitting:
        cheapest = min(fitting, key=lambda a: a.get("price_eur") or 10 ** 9)
        add(f"- Goedkoopste geteste kaart die volgens deze meting past: "
            f"**{cheapest['name']}** ({_n(cheapest.get('price_eur'), ' euro', 0)}).")
    add("")

    # ------------------------------------------------------------ vraag 1
    add(f"## 1. Past een klas van {class_size} op "
        f"{_n(hardware.get('vram_gb'), ' GB', 0)}, en met hoeveel marge?")
    add("")
    if not findings.get("class_grades"):
        add("_Nog geen sweep-runs met deze klasgrootte in deze resultatenmap._")
    else:
        add(_verdict_line(class_grade or "onbekend"))
        add("")
        add("| Contextgrootte per student | Oordeel | p90 TTFT | p90 doorlooptijd instructie | Preempties |")
        add("|---|---|---|---|---|")
        for result in sorted([r for r in results
                              if r.spec.kind == "sweep" and r.spec.students == class_size],
                             key=lambda r: r.spec.context_tokens):
            add(f"| {result.spec.context_tokens // 1000}k | {result.grade.colour} "
                f"| {_n(result.aggregate.get('ttft', {}).get('p90'), ' s')} "
                f"| {_n(result.aggregate.get('burst_duration', {}).get('p90'), ' s', 0)} "
                f"| {_n(result.server.get('preemptions'), '', 0)} |")
        add("")
        if findings.get("kv_headroom_fraction") is not None:
            add(f"De marge in geheugen is **{_pct(findings['kv_headroom_fraction'])}** "
                f"van de cachepool: bij de zwaarste run stond "
                f"{_n(findings.get('kv_peak_gb'), ' GB')} van de "
                f"{_n(findings['kv_pool_gb'], ' GB')} vol. De pool is berekend als "
                f"{_n(hardware.get('vram_gb'), ' GB', 0)} x "
                f"{hardware.get('gpu_memory_utilization')} min "
                f"{_n(hardware.get('model_weights_gb'), ' GB')} aan modelgewichten.")
        else:
            add("De KV-cachebezetting is niet gemeten; zonder bereikbare "
                "`/metrics`-endpoint kan de geheugenmarge niet worden bepaald.")
        assumed = findings.get("hardware_defaults_used") or []
        if assumed:
            add("")
            add(f"> Let op: {', '.join(assumed)} stond niet in de configuratie. "
                f"Hiervoor is de aanname uit de aanvraag gebruikt. Vul "
                f"`hardware` in `config.json` in om dit met gemeten waarden te "
                f"vervangen.")
    add("")

    # ------------------------------------------------------------ vraag 2
    add("## 2. Waar ligt de klif?")
    add("")
    if cliff:
        reasons = findings.get("cliff_reasons") or []
        add(f"De oplopende run hield het uit tot **{cliff} gelijktijdige studenten**.")
        if reasons:
            add(f"Bij de volgende student braken: {', '.join(reasons)}.")
        add("")
    add("Per contextgrootte, het aantal studenten waarbij het oordeel omslaat:")
    add("")
    add("| Contextgrootte | Eerste oranje bij | Eerste rood bij |")
    add("|---|---|---|")
    for context, entry in (findings.get("cliff_per_context") or {}).items():
        add(f"| {context} | {entry.get('eerste_oranje') or 'niet bereikt'} "
            f"| {entry.get('eerste_rood') or 'niet bereikt'} |")
    add("")
    add("De contextgrootte is hier de belangrijkere as: een grotere context kost "
        "per student lineair meer cachegeheugen, terwijl een extra student er "
        "alleen bijkomt als er nog blokken vrij zijn.")
    add("")

    # ------------------------------------------------------------ vraag 3
    add("## 3. Zou minder videogeheugen ook volstaan?")
    add("")
    if not findings.get("alternatives") or findings.get("kv_peak_fraction") is None:
        add("_Zonder gemeten KV-cachebezetting is deze vraag niet te beantwoorden. "
            "Controleer of `/metrics` bereikbaar was tijdens de runs._")
    else:
        add(f"De zwaarste geteste klas gebruikte **{_n(findings.get('kv_peak_gb'), ' GB')}** "
            f"aan KV-cache. Diezelfde behoefte afgezet tegen de alternatieven:")
        add("")
        add("| Kaart | Videogeheugen | Cachepool | Nodig | Marge | Past? | Prijs |")
        add("|---|---|---|---|---|---|---|")
        for alternative in findings["alternatives"]:
            add(f"| {alternative['name']} | {_n(alternative.get('vram_gb'), ' GB', 0)} "
                f"| {_n(alternative.get('kv_pool_gb'), ' GB')} "
                f"| {_n(alternative.get('needed_kv_gb'), ' GB')} "
                f"| {_n(alternative.get('margin_gb'), ' GB')} "
                f"| {'ja' if alternative.get('fits') else 'nee'} "
                f"| {_n(alternative.get('price_eur'), ' euro', 0)} |")
        add("")
        add("Deze vergelijking rekent alleen met geheugen. Een kaart met minder "
            "geheugen heeft doorgaans ook minder rekenkracht en geheugenbandbreedte, "
            "wat de doorlooptijd van een instructie raakt ook als het geheugen past. "
            "Meet daarom de gekozen alternatieven na met dezelfde matrix voordat "
            "je bestelt; het harnas draait ongewijzigd tegen elk endpoint.")
    add("")

    # ------------------------------------------------------------ vraag 4
    add("## 4. Welke vLLM-instellingen zijn bepalend?")
    add("")
    if findings.get("engine"):
        add("| Variant | Vlaggen | Oordeel | p90 TTFT | KV-piek | Preempties | Cache hit rate |")
        add("|---|---|---|---|---|---|---|")
        for entry in findings["engine"]:
            add(f"| {entry['name']} | `{entry.get('flags', '')}` | {entry['grade']} "
                f"| {_n(entry.get('ttft_p90_s'), ' s')} | {_pct(entry.get('kv_peak'))} "
                f"| {_n(entry.get('preemptions'), '', 0)} "
                f"| {_pct(entry.get('prefix_cache_hit_rate'))} |")
        add("")
        add(f"Beste variant in deze meting: **{findings.get('engine_best')}** "
            f"(`{findings.get('engine_best_flags')}`).")
    else:
        add("_Er zijn geen vLLM-variantruns in deze resultatenmap. Die vereisen een "
            "herstart van de server per variant; zie `stresstest matrix --only engine`._")
    add("")

    # ----------------------------------------------------- gedeelde basis
    if findings.get("shared"):
        add("## Wat de gedeelde projectbasis oplevert")
        add("")
        add("Studenten in een klas werken aan dezelfde opdracht. Dat scheelt "
            "geheugen, omdat het gedeelde deel van hun context maar een keer in "
            "de cache staat. Gemeten:")
        add("")
        add("| Gedeelde basis | Context | Cache hit rate | p90 TTFT | KV-piek | Oordeel |")
        add("|---|---|---|---|---|---|")
        for entry in findings["shared"]:
            add(f"| {int(entry['shared_fraction'] * 100)}% "
                f"| {entry['context_tokens'] // 1000}k "
                f"| {_pct(entry.get('prefix_cache_hit_rate'))} "
                f"| {_n(entry.get('ttft_p90_s'), ' s')} "
                f"| {_pct(entry.get('kv_peak'))} | {entry['grade']} |")
        add("")

    # ------------------------------------------------------------ scenario
    if findings.get("scenarios"):
        add("## De benoemde momenten")
        add("")
        add("Deze zijn belangrijker dan het gemiddelde, want dit is wat er misgaat.")
        add("")
        add("| Moment | Oordeel | p90 TTFT | p90 doorlooptijd | Preempties | Waarom rood/oranje |")
        add("|---|---|---|---|---|---|")
        for entry in findings["scenarios"]:
            add(f"| {entry['label']} | {entry['grade']} "
                f"| {_n(entry.get('ttft_p90_s'), ' s')} "
                f"| {_n(entry.get('burst_p90_s'), ' s', 0)} "
                f"| {_n(entry.get('preemptions'), '', 0)} "
                f"| {'; '.join(entry.get('reasons') or []) or '-'} |")
        add("")

    # --------------------------------------------------------------- les
    if findings.get("lesson"):
        lesson = findings["lesson"]
        add("## Fase 2: de lesvalidatie van negentig minuten")
        add("")
        add(f"Oordeel over het volledige lesuur: **{lesson['grade']}**. "
            f"p90 TTFT {_n(lesson.get('ttft_p90_s'), ' s')}, p90 doorlooptijd van een "
            f"instructie {_n(lesson.get('burst_p90_s'), ' s', 0)}, "
            f"{_n(lesson.get('preemptions'), '', 0)} preempties, prefix cache hit rate "
            f"{_pct(lesson.get('prefix_cache_hit_rate'))}, KV-piek "
            f"{_pct(lesson.get('kv_peak'))}.")
        add("")
        add("De grafiek `charts/04_cache_en_kv_over_tijd.svg` laat zien wat er tijdens "
            "de tien minuten klassikale uitleg met de cache gebeurt, en hoe duur de "
            "eerste stap daarna is.")
        add("")

    # ------------------------------------------------------------ drempels
    add("## Hoe de kleuren zijn bepaald")
    add("")
    from .grading import DEFAULT_THRESHOLDS
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(config.get("grading", {}).get("thresholds") or {})
    add(f"- **Groen**: p90 TTFT onder {_n(thresholds.get('ttft_p90_green_s'), ' s', 0)}, "
        f"p90 doorlooptijd van een instructie onder "
        f"{_n(thresholds.get('burst_p90_green_s'), ' s', 0)}, decodesnelheid boven "
        f"{_n(thresholds.get('decode_tps_p50_green'), ' tokens/s', 0)} per stream, "
        f"geen preempties, geen afgelopen verzoeken.")
    add(f"- **Oranje**: p90 TTFT tot {_n(thresholds.get('ttft_p90_amber_s'), ' s', 0)}, "
        f"of doorlooptijd tot {_n(thresholds.get('burst_p90_amber_s'), ' s', 0)}, "
        f"of incidentele preempties.")
    add("- **Rood**: daarboven, of aanhoudende preempties, of verzoeken die aflopen.")
    add("")
    add("Twee afwijkingen ten opzichte van de oorspronkelijke opzet, allebei bewust:")
    add("")
    add("1. **De doorlooptijd van een hele instructie telt mee.** Time to first token "
        "is de eerste van drie tot vijftien modelaanroepen die de agent doet voor een "
        "enkele opdracht van de student. Een run kan een prima TTFT hebben en toch "
        "twee minuten over een instructie doen; dat is wat de student wacht.")
    add("2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die "
        "hangt af van het scenario, niet van de hardware: in de runs met 0% gedeelde "
        "basis en in de koude start is 60% per definitie onhaalbaar. Een lage hit rate "
        "waar we een hoge verwachtten staat als waarschuwing bij de run.")
    disagreements = findings.get("grading_disagreements") or []
    if disagreements:
        add("")
        add(f"Elke run is ook beoordeeld volgens de oorspronkelijke definitie uit de "
            f"opdracht (kolom `grade_brief` in `summary.csv`). De twee definities "
            f"verschillen bij {len(disagreements)} van de {findings.get('runs')} runs: "
            f"{', '.join(disagreements[:10])}"
            + (" ..." if len(disagreements) > 10 else "") + ".")
    add("")

    # ---------------------------------------------------------- beperkingen
    add("## Wat deze test niet zegt")
    add("")
    add("- Niets over de kwaliteit van de gegenereerde code. Er is capaciteit en "
        "latentie gemeten, niet of het model goede antwoorden geeft.")
    add("- Niets over andere inferentie-engines. vLLM was al gekozen.")
    add("- De simulatie gebruikt echte broncode uit een publiek voorbeeldproject en "
        "een gedragsmodel met vier persona's. Het gedrag van een echte klas wijkt af; "
        "de persona-verdeling staat daarom in `config.json` en is aan te passen.")
    if not environment.get("tokenizer_exact", True):
        add("- De contextgroottes zijn **geschat** op tekens per token, niet exact "
            "geteld. Die schatting is geijkt op dit corpus en klopt binnen enkele "
            "procenten, maar voor een harde uitspraak over geheugen: installeer "
            "het pakket `tokenizers` en meet opnieuw.")
    add("")
    add("---")
    add("")
    add(f"Gegenereerd door het stresstestharnas, versie {environment.get('version')}. "
        f"Configuratie: `config.json` naast dit bestand.")
    return "\n".join(lines) + "\n"


def write(directory: str, results: Sequence[RunResult], config: dict,
          environment: dict) -> str:
    path = os.path.join(directory, "RESULTATEN.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render(results, config, environment))
    return path
