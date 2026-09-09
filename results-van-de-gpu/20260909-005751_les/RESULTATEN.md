# Resultaten stresstest agentic coding

Gemeten op **NVIDIA RTX PRO 6000 Blackwell Server Edition (600W) (600W)** (96 GB videogeheugen) met **qwen3-coder** via vLLM.
Uitgevoerd op 2026-09-09, 1 runs, seed 20250908.

> Dit bestand is automatisch gegenereerd uit de meetgegevens in deze map. De onderliggende getallen staan in `summary.csv`; per run staan alle afzonderlijke verzoeken in `runs/<run_id>/requests.csv`.

## De korte versie


## 1. Past een klas van 20 op 96 GB, en met hoeveel marge?

_Nog geen sweep-runs met deze klasgrootte in deze resultatenmap._

## 2. Waar ligt de klif?

Per contextgrootte, het aantal studenten waarbij het oordeel omslaat:

| Contextgrootte | Eerste oranje bij | Eerste rood bij |
|---|---|---|

De contextgrootte is hier de belangrijkere as: een grotere context kost per student lineair meer cachegeheugen, terwijl een extra student er alleen bijkomt als er nog blokken vrij zijn.

## 3. Zou minder videogeheugen ook volstaan?

_Zonder gemeten KV-cachebezetting is deze vraag niet te beantwoorden. Controleer of `/metrics` bereikbaar was tijdens de runs._

## 4. Welke vLLM-instellingen zijn bepalend?

_Er zijn geen vLLM-variantruns in deze resultatenmap. Die vereisen een herstart van de server per variant; zie `stresstest matrix --only engine`._

## Fase 2: de lesvalidatie van negentig minuten

Oordeel over het volledige lesuur: **groen**. p90 TTFT 0.3 s, p90 doorlooptijd van een instructie 69 s, 0 preempties, prefix cache hit rate 96%, KV-piek 15%.

De grafiek `charts/04_cache_en_kv_over_tijd.svg` laat zien wat er tijdens de tien minuten klassikale uitleg met de cache gebeurt, en hoe duur de eerste stap daarna is.

## Hoe de kleuren zijn bepaald

- **Groen**: p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder 90 s, decodesnelheid boven 12 tokens/s per stream, geen preempties, geen afgelopen verzoeken.
- **Oranje**: p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele preempties.
- **Rood**: daarboven, of aanhoudende preempties, of verzoeken die aflopen.

Twee afwijkingen ten opzichte van de oorspronkelijke opzet, allebei bewust:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is de eerste van drie tot vijftien modelaanroepen die de agent doet voor een enkele opdracht van de student. Een run kan een prima TTFT hebben en toch twee minuten over een instructie doen; dat is wat de student wacht.
2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die hangt af van het scenario, niet van de hardware: in de runs met 0% gedeelde basis en in de koude start is 60% per definitie onhaalbaar. Een lage hit rate waar we een hoge verwachtten staat als waarschuwing bij de run.

## Wat deze test niet zegt

- Niets over de kwaliteit van de gegenereerde code. Er is capaciteit en latentie gemeten, niet of het model goede antwoorden geeft.
- Niets over andere inferentie-engines. vLLM was al gekozen.
- De simulatie gebruikt echte broncode uit een publiek voorbeeldproject en een gedragsmodel met vier persona's. Het gedrag van een echte klas wijkt af; de persona-verdeling staat daarom in `config.json` en is aan te passen.
- De contextgroottes zijn **geschat** op tekens per token, niet exact geteld. Die schatting is geijkt op dit corpus en klopt binnen enkele procenten, maar voor een harde uitspraak over geheugen: installeer het pakket `tokenizers` en meet opnieuw.

---

Gegenereerd door het stresstestharnas, versie 1.0.0. Configuratie: `config.json` naast dit bestand.
