# Resultaten stresstest agentic coding

Gemeten op **NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)** (96 GB videogeheugen) met **qwen3-coder** via vLLM.
Uitgevoerd op 2026-09-09, 38 runs, seed 20250908.

> Dit bestand is automatisch gegenereerd uit de meetgegevens in deze map. De onderliggende getallen staan in `summary.csv`; per run staan alle afzonderlijke verzoeken in `runs/<run_id>/requests.csv`.

## De korte versie

- Een klas van 20 studenten komt in de zwaarste geteste contextgrootte uit op **groen**.
- De klif ligt bij **40 gelijktijdige studenten** (de test liep tot het ingestelde maximum zonder te breken).
- Piekgebruik van de KV-cache: **29%** van de 54.9 GB cachepool, oftewel **15.7 GB** in gebruik en **39.2 GB** over.
- Goedkoopste geteste kaart die volgens deze meting past: **RTX PRO 5000 (72 GB)** (7602 euro).

## 1. Past een klas van 20 op 96 GB, en met hoeveel marge?

**Ja.**

| Contextgrootte per student | Oordeel | p90 TTFT | p90 doorlooptijd instructie | Preempties |
|---|---|---|---|---|
| 8k | groen | 0.2 s | 25 s | 0 |
| 32k | groen | 0.3 s | 36 s | 0 |
| 64k | groen | 0.4 s | 40 s | 0 |
| 100k | groen | 0.6 s | 50 s | 0 |

De marge in geheugen is **71%** van de cachepool: bij de zwaarste run stond 15.7 GB van de 54.9 GB vol. De pool is berekend als 96 GB x 0.9 min 31.1 GB aan modelgewichten.

## 2. Waar ligt de klif?

De oplopende run hield het uit tot **40 gelijktijdige studenten**.

Per contextgrootte, het aantal studenten waarbij het oordeel omslaat:

| Contextgrootte | Eerste oranje bij | Eerste rood bij |
|---|---|---|
| 8k | niet bereikt | niet bereikt |
| 32k | niet bereikt | niet bereikt |
| 64k | niet bereikt | niet bereikt |
| 100k | 30 | 30 |

De contextgrootte is hier de belangrijkere as: een grotere context kost per student lineair meer cachegeheugen, terwijl een extra student er alleen bijkomt als er nog blokken vrij zijn.

## 3. Zou minder videogeheugen ook volstaan?

De zwaarste geteste klas gebruikte **15.7 GB** aan KV-cache. Diezelfde behoefte afgezet tegen de alternatieven:

| Kaart | Videogeheugen | Cachepool | Nodig | Marge | Past? | Prijs |
|---|---|---|---|---|---|---|
| RTX PRO 5000 (72 GB) | 72 GB | 33.7 GB | 15.7 GB | 18.0 GB | ja | 7602 euro |
| 2x RTX 5090 (64 GB) | 64 GB | 26.5 GB | 15.7 GB | 10.8 GB | ja | 10000 euro |
| RTX PRO 6000 (96 GB) | 96 GB | 55.3 GB | 15.7 GB | 39.6 GB | ja | 37400 euro |

Deze vergelijking rekent alleen met geheugen. Een kaart met minder geheugen heeft doorgaans ook minder rekenkracht en geheugenbandbreedte, wat de doorlooptijd van een instructie raakt ook als het geheugen past. Meet daarom de gekozen alternatieven na met dezelfde matrix voordat je bestelt; het harnas draait ongewijzigd tegen elk endpoint.

## 4. Welke vLLM-instellingen zijn bepalend?

| Variant | Vlaggen | Oordeel | p90 TTFT | KV-piek | Preempties | Cache hit rate |
|---|---|---|---|---|---|---|
| kv_auto | `--kv-cache-dtype auto --max-num-seqs 32 --max-model-len 131072` | groen | 0.4 s | 28% | 0 | 99% |
| kv_fp8 | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072` | groen | 0.4 s | 13% | 0 | 99% |
| len_65k | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 65536` | groen | 0.4 s | 13% | 0 | 99% |
| seqs_16 | `--kv-cache-dtype fp8 --max-num-seqs 16 --max-model-len 131072` | groen | 0.4 s | 13% | 0 | 99% |
| seqs_64 | `--kv-cache-dtype fp8 --max-num-seqs 64 --max-model-len 131072` | groen | 0.4 s | 13% | 0 | 99% |

Beste variant in deze meting: **kv_auto** (`--kv-cache-dtype auto --max-num-seqs 32 --max-model-len 131072`).

## Wat de gedeelde projectbasis oplevert

Studenten in een klas werken aan dezelfde opdracht. Dat scheelt geheugen, omdat het gedeelde deel van hun context maar een keer in de cache staat. Gemeten:

| Gedeelde basis | Context | Cache hit rate | p90 TTFT | KV-piek | Oordeel |
|---|---|---|---|---|---|
| 0% | 32k | 97% | 0.2 s | 8% | groen |
| 50% | 32k | 97% | 0.3 s | 7% | groen |
| 90% | 32k | 98% | 0.3 s | 5% | groen |
| 0% | 64k | 99% | 0.4 s | 18% | groen |
| 50% | 64k | 99% | 0.4 s | 13% | groen |
| 90% | 64k | 99% | 0.4 s | 9% | groen |

## De benoemde momenten

Deze zijn belangrijker dan het gemiddelde, want dit is wat er misgaat.

| Moment | Oordeel | p90 TTFT | p90 doorlooptijd | Preempties | Waarom rood/oranje |
|---|---|---|---|---|---|
| Deadline, 20 studenten in burst | oranje | 0.3 s | 123 s | 0 | p90 doorlooptijd instructie 123s |
| Koude start, 20 studenten | groen | 0.2 s | 43 s | 0 | - |
| Koude start, 30 studenten | groen | 0.2 s | 53 s | 0 | - |
| Lange sessies, weinig studenten, zeer grote context | groen | 0.6 s | 51 s | 0 | - |
| Na de stilte: tien minuten niets, dan hervat iedereen tegelijk | groen | 0.3 s | 83 s | 0 | - |
| Worst case: 20 studenten, maximale context, geen gedeelde prefix | rood | 0.7 s | 396 s | 0 | p90 doorlooptijd instructie 396s |

## Hoe de kleuren zijn bepaald

- **Groen**: p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder 90 s, decodesnelheid boven 12 tokens/s per stream, geen preempties, geen afgelopen verzoeken.
- **Oranje**: p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele preempties.
- **Rood**: daarboven, of aanhoudende preempties, of verzoeken die aflopen.

Twee afwijkingen ten opzichte van de oorspronkelijke opzet, allebei bewust:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is de eerste van drie tot vijftien modelaanroepen die de agent doet voor een enkele opdracht van de student. Een run kan een prima TTFT hebben en toch twee minuten over een instructie doen; dat is wat de student wacht.
2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die hangt af van het scenario, niet van de hardware: in de runs met 0% gedeelde basis en in de koude start is 60% per definitie onhaalbaar. Een lage hit rate waar we een hoge verwachtten staat als waarschuwing bij de run.

Elke run is ook beoordeeld volgens de oorspronkelijke definitie uit de opdracht (kolom `grade_brief` in `summary.csv`). De twee definities verschillen bij 3 van de 38 runs: scen_deadline, scen_worst_case, sweep_s30_c100k.

## Wat deze test niet zegt

- Niets over de kwaliteit van de gegenereerde code. Er is capaciteit en latentie gemeten, niet of het model goede antwoorden geeft.
- Niets over andere inferentie-engines. vLLM was al gekozen.
- De simulatie gebruikt echte broncode uit een publiek voorbeeldproject en een gedragsmodel met vier persona's. Het gedrag van een echte klas wijkt af; de persona-verdeling staat daarom in `config.json` en is aan te passen.
- De contextgroottes zijn **geschat** op tekens per token, niet exact geteld. Die schatting is geijkt op dit corpus en klopt binnen enkele procenten, maar voor een harde uitspraak over geheugen: installeer het pakket `tokenizers` en meet opnieuw.

---

Gegenereerd door het stresstestharnas, versie 1.0.0. Configuratie: `config.json` naast dit bestand.
