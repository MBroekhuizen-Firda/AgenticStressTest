# Resultaten stresstest agentic coding

Gemeten op **NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)** (47 GB videogeheugen) met **qwen3-coder** via vLLM.
Meting van 2026-09-10: 39 runs, seed 20250908. Dit bestand is geschreven op 2026-09-10.

> Dit bestand is automatisch gegenereerd uit:
> - `results/20260910-084827_matrix` -- gemeten 2026-09-10, 38 runs, opstelling `d7954db50151`
> - `results/20260910-165600_les` -- gemeten 2026-09-10, 1 run, opstelling `d7954db50151`
>
> De onderliggende getallen staan daar in `summary.csv`; per run staan alle afzonderlijke verzoeken in `runs/<run_id>/requests.csv`.

## De korte versie

- Een klas van 20 studenten komt in de zwaarste geteste contextgrootte uit op **rood**.
- De klif ligt bij **39 gelijktijdige studenten**; tot 37 hield de oplopende run het.
- Piekgebruik van de KV-cache bij een klas van 20: **100%** van de 11.6 GB cachepool, oftewel **11.5 GB** in gebruik en **0.0 GB** over.
- Hoogste piek over alle runs: **11.6 GB** (100%), in `les_90min`. Dat is het getal waartegen een kleinere kaart hieronder wordt afgemeten.
- Goedkoopste geteste kaart die de zwaarste gemeten run nog aankan: **RTX PRO 5000 (72 GB)** (7602 euro) — op geheugen alleen; over de rekenkracht van die kaart zegt deze meting niets.

## 1. Past een klas van 20 op 47 GB, en met hoeveel marge?

**Nee.**

| Contextgrootte per student | Oordeel | p90 TTFT | p90 doorlooptijd instructie | Preempties |
|---|---|---|---|---|
| 8k | oranje | 0.7 s | 122 s | 0 |
| 32k | oranje | 0.9 s | 172 s | 0 |
| 64k | rood | 4.0 s | 246 s | 0 |
| 100k | rood | 114.9 s | 417 s | 0 |

De marge in geheugen is **0%** van de cachepool: bij de zwaarste run stond 11.5 GB van de 11.6 GB vol. De pool is berekend als 47 GB x 0.9 min 31.1 GB aan modelgewichten.

Dat getal geldt voor de sweep-runs met 20 studenten. Over *alle* runs samen ligt de piek hoger: 11.6 GB (100%) in `les_90min` — Lesvalidatie 90 minuten, 20 studenten. Vraag 3 rekent met die hogere piek.

## 2. Waar ligt de klif?

De oplopende run brak bij **39 gelijktijdige studenten** bij 32k context; tot 37 hield hij het.
Wat er brak: p90 TTFT 50.7s > 45s.
De run stapte met 2 studenten tegelijk, dus de grens ligt ergens tussen 37 en 39. Zet `matrix.rampup.step_students` op 1 om hem preciezer te zoeken.

Per contextgrootte, het aantal studenten waarbij het oordeel omslaat:

| Contextgrootte | Eerste oranje bij | Eerste rood bij |
|---|---|---|
| 8k | 10 | 30 |
| 32k | 10 | 30 |
| 64k | 10 | 20 |
| 100k | 5 | 10 |

De contextgrootte is hier de belangrijkere as: een grotere context kost per student lineair meer cachegeheugen, terwijl een extra student er alleen bijkomt als er nog blokken vrij zijn.

## 3. Zou minder videogeheugen ook volstaan?

Er zijn twee getallen in omloop, en ze geven een ander antwoord. Een klas van 20 in de sweep kwam niet hoger dan **11.5 GB**. De zwaarste run uit de hele meting — `les_90min`, Lesvalidatie 90 minuten, 20 studenten — vroeg **11.6 GB**. De kolom **Past?** hieronder oordeelt op dat tweede getal: een kaart die de zwaarste gemeten belasting niet aankan, kun je niet aanbevelen omdat het gemiddelde er wel op past.

| Kaart | Videogeheugen | Cachepool | Nodig (klas) | Nodig (zwaarste run) | Marge | Past? | Prijs |
|---|---|---|---|---|---|---|---|
| RTX PRO 5000 (72 GB) | 72 GB | 33.7 GB | 11.5 GB | 11.6 GB | 22.1 GB | ja | 7602 euro |
| 2x RTX 5090 (64 GB) | 64 GB | 26.5 GB | 11.5 GB | 11.6 GB | 14.9 GB | ja | 10000 euro |
| RTX PRO 6000 (96 GB) | 96 GB | 55.3 GB | 11.5 GB | 11.6 GB | 43.7 GB | ja | 37400 euro |

Deze vergelijking rekent alleen met geheugen. Een kaart met minder geheugen heeft doorgaans ook minder rekenkracht en geheugenbandbreedte, wat de doorlooptijd van een instructie raakt ook als het geheugen past. Meet daarom de gekozen alternatieven na met dezelfde matrix voordat je bestelt; het harnas draait ongewijzigd tegen elk endpoint.

## 4. Welke vLLM-instellingen zijn bepalend?

| Variant | Vlaggen | Oordeel | p90 TTFT | KV-piek | Preempties | Cache hit rate |
|---|---|---|---|---|---|---|
| kv_auto | `--kv-cache-dtype auto --max-num-seqs 32 --max-model-len 131072` | rood | 95.1 s | 99% | 0 | 46% |
| kv_fp8 | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072` | rood | 2.1 s | 100% | 0 | 95% |
| len_65k | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 65536` | rood | 3.1 s | 100% | 0 | 94% |
| seqs_16 | `--kv-cache-dtype fp8 --max-num-seqs 16 --max-model-len 131072` | rood | 3.7 s | 99% | 0 | 94% |
| seqs_64 | `--kv-cache-dtype fp8 --max-num-seqs 64 --max-model-len 131072` | rood | 4.2 s | 100% | 0 | 93% |

Beste variant in deze meting: **kv_fp8** (`--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072`).

De keuze is niet op p90 TTFT gemaakt. Alle varianten zitten daar ver onder de groen-grens, en een verschil kleiner dan 1.0 s — vijf procent van die grens — is ruis, geen signaal. Wat wél uiteenloopt is de cachebezetting, en daarop is gerangschikt.

## Wat de gedeelde projectbasis oplevert

Studenten in een klas werken aan dezelfde opdracht. Dat scheelt geheugen, omdat het gedeelde deel van hun context maar een keer in de cache staat. Gemeten:

| Gedeelde basis | Context | Cache hit rate | p90 TTFT | KV-piek | Oordeel |
|---|---|---|---|---|---|
| 0% | 32k | 92% | 1.1 s | 83% | rood |
| 50% | 32k | 94% | 0.8 s | 60% | oranje |
| 90% | 32k | 95% | 1.7 s | 49% | oranje |
| 0% | 64k | 4% | 70.1 s | 100% | rood |
| 50% | 64k | 94% | 3.4 s | 100% | rood |
| 90% | 64k | 98% | 1.2 s | 72% | rood |

## Uit wie de klas bestond

Twee assen, los van elkaar geloot. Een *persona* is een tempo: hoe lang iemand nadenkt, hoeveel stappen een instructie kost, of hij afhaakt. Een *werkprofiel* is een gewicht: hoe groot de codebase is, hoeveel bestanden de agent per keer leest, hoeveel het model per stap schrijft.

De aandelen hieronder zijn de loting die voor **elke** run in deze meting gold, met seed `20250908`. Vingerafdruk van dit gedragsmodel: `0389659f7efb` -- twee metingen met hetzelfde getal zijn met dezelfde persona's, werkprofielen en seed gedaan, en dus onderling te vergelijken. (De vingerafdruk boven aan dit bestand is een andere: die dekt de hele opstelling, corpus en tokenizer inbegrepen.)

De kolom *geloot* komt uit `les_90min`: de run met 20 studenten op activiteit normaal. Andere runs in deze meting hebben minder of meer studenten en dus een eigen loting.

| Persona | Aandeel | Geloot |
|---|---|---|
| gemiddelde | 45% | 9 |
| worstelaar | 25% | 5 |
| afhaker | 15% | 3 |
| doorpakker | 15% | 3 |

| Werkprofiel | Aandeel | Geloot |
|---|---|---|
| klein | 50% | 10 |
| middel | 30% | 6 |
| doorspitten | 20% | 4 |

Het werkprofiel bepaalt hoeveel werk één modelaanroep is, en dat is het getal waar de doorlooptijd het gevoeligst voor is. De gebruikte waarden staan in `environment.json` naast dit bestand.

## Is de context groot genoeg?

Een sessie die vol raakt moet ruimte maken: het oudste deel van de werkhistorie gaat eruit, en de volgende agentstap moet dat deel opnieuw laten voorrekenen. Hoe vaak dat gebeurde, per contextgrootte:

| Contextgrootte | Compacties per 100 stappen | Compacties totaal | Hoogste context bereikt |
|---|---|---|---|
| 8k | 12.5 | 74 | 52879 (661% van het doel) |
| 32k | 3.8 | 17 | 57235 (179% van het doel) |
| 64k | 1.6 | 5 | 63323 (99% van het doel) |
| 100k | 0.0 | 0 | 85584 (86% van het doel) |

Niet de broncode vult het venster maar de gesprekshistorie: gelezen bestanden, diffs, testuitvoer. Een grotere context koopt dus minder compacties, en betaalt daarvoor in cachegeheugen en doorlooptijd — beide staan in vraag 1 en vraag 2.

> Let op: hoe snel een venster vol loopt hangt volledig af van hoeveel elke agentstap toevoegt. Het gedragsmodel van dit harnas gaat uit van kleine stappen (een gelezen bestand uit dit corpus is een paar honderd tokens). Een agent die complete testlogs of grote bestanden in de context kiepert, vult hetzelfde venster een orde van grootte sneller. Deze tabel geldt dus voor dit gedragsmodel, niet voor agentic coding in het algemeen.

## De benoemde momenten

Deze zijn belangrijker dan het gemiddelde, want dit is wat er misgaat.

| Moment | Oordeel | p90 TTFT | p90 doorlooptijd | Preempties | Waarom rood/oranje |
|---|---|---|---|---|---|
| Deadline, 20 studenten in burst | rood | 9.8 s | 408 s | 0 | p90 doorlooptijd instructie 408s |
| Koude start, 20 studenten | oranje | 0.6 s | 139 s | 0 | p90 doorlooptijd instructie 139s |
| Koude start, 30 studenten | rood | 0.6 s | 214 s | 0 | p90 doorlooptijd instructie 214s |
| Lange sessies, weinig studenten, zeer grote context | rood | 2.3 s | 220 s | 0 | p90 doorlooptijd instructie 220s |
| Na de stilte: tien minuten niets, dan hervat iedereen tegelijk | rood | 3.3 s | 254 s | 0 | p90 doorlooptijd instructie 254s |
| Worst case: 20 studenten, maximale context, geen gedeelde prefix | rood | niet gemeten | 428 s | 0 | p90 TTFT niet gemeten; p90 doorlooptijd instructie 428s; decodesnelheid niet gemeten; 100.0% van de verzoeken faalde |

## Fase 2: de lesvalidatie van negentig minuten

Oordeel over het volledige lesuur: **rood**. p90 TTFT 4.8 s, p90 doorlooptijd van een instructie 314 s, 4 preempties, prefix cache hit rate 89%, KV-piek 100%.

Contextdruk over het hele lesuur: 146 compacties, 6.5 per 100 agentstappen, hoogste bereikte context 73738 tokens.

De grafiek `04_cache_en_kv_over_tijd.svg`, in de `charts`-map van de resultatenmap van de lesvalidatie, laat zien wat er tijdens de tien minuten klassikale uitleg met de cache gebeurt, en hoe duur de eerste stap daarna is.

## Hoe de kleuren zijn bepaald

- **Groen**: p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder 90 s, decodesnelheid boven 12 tokens/s per stream, geen preempties, geen afgelopen verzoeken.
- **Oranje**: p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele preempties.
- **Rood**: daarboven, of aanhoudende preempties, of verzoeken die aflopen.

Twee afwijkingen ten opzichte van de oorspronkelijke opzet, allebei bewust:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is de eerste van drie tot vijftien modelaanroepen die de agent doet voor een enkele opdracht van de student. Een run kan een prima TTFT hebben en toch twee minuten over een instructie doen; dat is wat de student wacht.
2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die hangt af van het scenario, niet van de hardware: in de runs met 0% gedeelde basis en in de koude start is 60% per definitie onhaalbaar. Een lage hit rate waar we een hoge verwachtten staat als waarschuwing bij de run.

Elke run is ook beoordeeld volgens de oorspronkelijke definitie uit de opdracht (kolom `grade_brief` in `summary.csv`). De twee definities verschillen bij 27 van de 39 runs: act_intensief_s10, act_intensief_s20, act_rustig_s20, engine_kv_fp8, engine_len_65k, engine_seqs_16, engine_seqs_64, scen_deadline, scen_koudestart_s20, scen_koudestart_s30 ....

## Wat deze test niet zegt

- Niets over de kwaliteit van de gegenereerde code. Er is capaciteit en latentie gemeten, niet of het model goede antwoorden geeft.
- Niets over andere inferentie-engines. vLLM was al gekozen.
- De simulatie gebruikt echte broncode uit een publiek voorbeeldproject en een gedragsmodel met vier persona's. Het gedrag van een echte klas wijkt af; de persona-verdeling staat daarom in `config.json` en is aan te passen.

---

Gegenereerd door het stresstestharnas, versie 1.0.0. Configuratie: `config.json` naast dit bestand.
