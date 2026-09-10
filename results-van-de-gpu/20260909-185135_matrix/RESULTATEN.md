# Resultaten stresstest agentic coding

Gemeten op **NVIDIA RTX PRO 6000 Blackwell Server Edition (600W)** (96 GB videogeheugen) met **qwen3-coder** via vLLM.
Uitgevoerd op 2026-09-10, 38 runs, seed 20250908.

> Dit bestand is automatisch gegenereerd uit de meetgegevens in deze map. De onderliggende getallen staan in `summary.csv`; per run staan alle afzonderlijke verzoeken in `runs/<run_id>/requests.csv`.

## De korte versie

- Een klas van 20 studenten komt in de zwaarste geteste contextgrootte uit op **oranje**.
- **Geen klif gevonden op de studenten-as.** De oplopende run liep tot het ingestelde plafond van 40 studenten en was daar nog groen — dat plafond is een instelling, geen gemeten grens.
- Piekgebruik van de KV-cache bij een klas van 20: **35%** van de 54.9 GB cachepool, oftewel **19.1 GB** in gebruik en **35.8 GB** over.
- Hoogste piek over alle runs: **46.2 GB** (84%), in `scen_worst_case`. Dat is het getal waartegen een kleinere kaart hieronder wordt afgemeten.
- Goedkoopste geteste kaart die de zwaarste gemeten run nog aankan: **RTX PRO 6000 (96 GB)** (37400 euro) — op geheugen alleen; over de rekenkracht van die kaart zegt deze meting niets.

## 1. Past een klas van 20 op 96 GB, en met hoeveel marge?

**Met kanttekeningen.**

| Contextgrootte per student | Oordeel | p90 TTFT | p90 doorlooptijd instructie | Preempties |
|---|---|---|---|---|
| 8k | groen | 0.4 s | 65 s | 0 |
| 32k | groen | 0.6 s | 76 s | 0 |
| 64k | oranje | 0.8 s | 102 s | 0 |
| 100k | oranje | 0.9 s | 164 s | 0 |

De marge in geheugen is **65%** van de cachepool: bij de zwaarste run stond 19.1 GB van de 54.9 GB vol. De pool is berekend als 96 GB x 0.9 min 31.1 GB aan modelgewichten.

Dat getal geldt voor de sweep-runs met 20 studenten. Over *alle* runs samen ligt de piek hoger: 46.2 GB (84%) in `scen_worst_case` — Worst case: 20 studenten, maximale context, geen gedeelde prefix. Vraag 3 rekent met die hogere piek.

## 2. Waar ligt de klif?

**Op de studenten-as is geen klif gevonden.** De oplopende run liep bij 32k context door tot het ingestelde plafond van 40 studenten en was daar nog groen. Die 40 is dus de bovengrens van de test (`matrix.rampup.max_students`), niet een gemeten grens: verhoog hem om verder te zoeken.

Per contextgrootte, het aantal studenten waarbij het oordeel omslaat:

| Contextgrootte | Eerste oranje bij | Eerste rood bij |
|---|---|---|
| 8k | niet bereikt | niet bereikt |
| 32k | 30 | niet bereikt |
| 64k | 10 | 30 |
| 100k | 10 | 30 |

De contextgrootte is hier de belangrijkere as: een grotere context kost per student lineair meer cachegeheugen, terwijl een extra student er alleen bijkomt als er nog blokken vrij zijn.

## 3. Zou minder videogeheugen ook volstaan?

Er zijn twee getallen in omloop, en ze geven een ander antwoord. Een klas van 20 in de sweep kwam niet hoger dan **19.1 GB**. De zwaarste run uit de hele meting — `scen_worst_case`, Worst case: 20 studenten, maximale context, geen gedeelde prefix — vroeg **46.2 GB**. De kolom **Past?** hieronder oordeelt op dat tweede getal: een kaart die de zwaarste gemeten belasting niet aankan, kun je niet aanbevelen omdat het gemiddelde er wel op past.

| Kaart | Videogeheugen | Cachepool | Nodig (klas) | Nodig (zwaarste run) | Marge | Past? | Prijs |
|---|---|---|---|---|---|---|---|
| RTX PRO 5000 (72 GB) | 72 GB | 33.7 GB | 19.1 GB | 46.2 GB | -12.5 GB | **nee** | 7602 euro |
| 2x RTX 5090 (64 GB) | 64 GB | 26.5 GB | 19.1 GB | 46.2 GB | -19.7 GB | **nee** | 10000 euro |
| RTX PRO 6000 (96 GB) | 96 GB | 55.3 GB | 19.1 GB | 46.2 GB | 9.1 GB | ja | 37400 euro |

Welke runs er op welke kaart niet passen:

- **RTX PRO 5000 (72 GB)**: 1 van 38 runs — `scen_worst_case`.
- **2x RTX 5090 (64 GB)**: 2 van 38 runs — `scen_worst_case`, `sweep_s30_c100k`.

Of die runs binnen bereik horen te vallen, is een keuze en geen meting: de worst case is met opzet extreem en komt bij een klas die aan dezelfde opdracht werkt niet voor. Maar hij staat wel in de opdracht, dus wie hem meerekent koopt een andere kaart dan wie hem weglaat. Zet die keuze expliciet op papier.

Deze vergelijking rekent alleen met geheugen. Een kaart met minder geheugen heeft doorgaans ook minder rekenkracht en geheugenbandbreedte, wat de doorlooptijd van een instructie raakt ook als het geheugen past. Meet daarom de gekozen alternatieven na met dezelfde matrix voordat je bestelt; het harnas draait ongewijzigd tegen elk endpoint.

## 4. Welke vLLM-instellingen zijn bepalend?

| Variant | Vlaggen | Oordeel | p90 TTFT | KV-piek | Preempties | Cache hit rate |
|---|---|---|---|---|---|---|
| kv_auto | `--kv-cache-dtype auto --max-num-seqs 32 --max-model-len 131072` | oranje | 0.7 s | 42% | 0 | 97% |
| kv_fp8 | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072` | oranje | 0.7 s | 19% | 0 | 97% |
| len_65k | `--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 65536` | oranje | 0.7 s | 19% | 0 | 97% |
| seqs_16 | `--kv-cache-dtype fp8 --max-num-seqs 16 --max-model-len 131072` | oranje | 0.7 s | 20% | 0 | 97% |
| seqs_64 | `--kv-cache-dtype fp8 --max-num-seqs 64 --max-model-len 131072` | oranje | 0.7 s | 19% | 0 | 97% |

Beste variant in deze meting: **kv_fp8** (`--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072`).

De keuze is niet op p90 TTFT gemaakt. Alle varianten zitten daar ver onder de groen-grens, en een verschil kleiner dan 1.0 s — vijf procent van die grens — is ruis, geen signaal. Wat wél uiteenloopt is de cachebezetting, en daarop is gerangschikt.

Binnen de meetruis gelijkwaardig: `kv_fp8`, `len_65k`, `seqs_16`, `seqs_64`. De vlaggen die deze varianten onderscheiden doen op deze kaart dus niets — wat niet wil zeggen dat ze op een kleinere kaart niets doen, want daar komt de geheugendruk wel in de buurt van de grens.

## Wat de gedeelde projectbasis oplevert

Studenten in een klas werken aan dezelfde opdracht. Dat scheelt geheugen, omdat het gedeelde deel van hun context maar een keer in de cache staat. Gemeten:

| Gedeelde basis | Context | Cache hit rate | p90 TTFT | KV-piek | Oordeel |
|---|---|---|---|---|---|
| 0% | 32k | 93% | 0.6 s | 17% | groen |
| 50% | 32k | 94% | 0.6 s | 15% | groen |
| 90% | 32k | 95% | 0.7 s | 13% | groen |
| 0% | 64k | 96% | 0.7 s | 29% | oranje |
| 50% | 64k | 97% | 0.8 s | 19% | oranje |
| 90% | 64k | 98% | 0.8 s | 17% | oranje |

## Uit wie de klas bestond

Twee assen, los van elkaar geloot. Een *persona* is een tempo: hoe lang iemand nadenkt, hoeveel stappen een instructie kost, of hij afhaakt. Een *werkprofiel* is een gewicht: hoe groot de codebase is, hoeveel bestanden de agent per keer leest, hoeveel het model per stap schrijft.

| Persona | Studenten | Werkprofiel | Studenten |
|---|---|---|---|
| gemiddelde | 5 | klein | 5 |
| doorpakker | 2 | middel | 3 |
| worstelaar | 2 | doorspitten | 2 |
| afhaker | 1 |  |  |

Het werkprofiel bepaalt hoeveel werk één modelaanroep is, en dat is het getal waar de doorlooptijd het gevoeligst voor is. De gebruikte waarden staan in `environment.json` naast dit bestand.

## Is de context groot genoeg?

Een sessie die vol raakt moet ruimte maken: het oudste deel van de werkhistorie gaat eruit, en de volgende agentstap moet dat deel opnieuw laten voorrekenen. Hoe vaak dat gebeurde, per contextgrootte:

| Contextgrootte | Compacties per 100 stappen | Compacties totaal | Hoogste context bereikt |
|---|---|---|---|
| 8k | 12.1 | 94 | 53045 (663% van het doel) |
| 32k | 5.3 | 36 | 57853 (181% van het doel) |
| 64k | 2.5 | 13 | 67571 (106% van het doel) |
| 100k | 0.6 | 3 | 90261 (90% van het doel) |

Niet de broncode vult het venster maar de gesprekshistorie: gelezen bestanden, diffs, testuitvoer. Een grotere context koopt dus minder compacties, en betaalt daarvoor in cachegeheugen en doorlooptijd — beide staan in vraag 1 en vraag 2.

> Let op: hoe snel een venster vol loopt hangt volledig af van hoeveel elke agentstap toevoegt. Het gedragsmodel van dit harnas gaat uit van kleine stappen (een gelezen bestand uit dit corpus is een paar honderd tokens). Een agent die complete testlogs of grote bestanden in de context kiepert, vult hetzelfde venster een orde van grootte sneller. Deze tabel geldt dus voor dit gedragsmodel, niet voor agentic coding in het algemeen.

## De benoemde momenten

Deze zijn belangrijker dan het gemiddelde, want dit is wat er misgaat.

| Moment | Oordeel | p90 TTFT | p90 doorlooptijd | Preempties | Waarom rood/oranje |
|---|---|---|---|---|---|
| Deadline, 20 studenten in burst | rood | 0.7 s | 285 s | 0 | p90 doorlooptijd instructie 285s |
| Koude start, 20 studenten | groen | 0.3 s | 87 s | 0 | - |
| Koude start, 30 studenten | oranje | 0.3 s | 144 s | 0 | p90 doorlooptijd instructie 144s |
| Lange sessies, weinig studenten, zeer grote context | oranje | 1.1 s | 146 s | 0 | p90 doorlooptijd instructie 146s |
| Na de stilte: tien minuten niets, dan hervat iedereen tegelijk | oranje | 0.9 s | 166 s | 0 | p90 doorlooptijd instructie 166s |
| Worst case: 20 studenten, maximale context, geen gedeelde prefix | rood | 1.0 s | 401 s | 0 | p90 doorlooptijd instructie 401s |

## Hoe de kleuren zijn bepaald

- **Groen**: p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder 90 s, decodesnelheid boven 12 tokens/s per stream, geen preempties, geen afgelopen verzoeken.
- **Oranje**: p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele preempties.
- **Rood**: daarboven, of aanhoudende preempties, of verzoeken die aflopen.

Twee afwijkingen ten opzichte van de oorspronkelijke opzet, allebei bewust:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is de eerste van drie tot vijftien modelaanroepen die de agent doet voor een enkele opdracht van de student. Een run kan een prima TTFT hebben en toch twee minuten over een instructie doen; dat is wat de student wacht.
2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die hangt af van het scenario, niet van de hardware: in de runs met 0% gedeelde basis en in de koude start is 60% per definitie onhaalbaar. Een lage hit rate waar we een hoge verwachtten staat als waarschuwing bij de run.

Elke run is ook beoordeeld volgens de oorspronkelijke definitie uit de opdracht (kolom `grade_brief` in `summary.csv`). De twee definities verschillen bij 22 van de 38 runs: act_intensief_s20, engine_kv_auto, engine_kv_fp8, engine_len_65k, engine_seqs_16, engine_seqs_64, rampup, scen_deadline, scen_koudestart_s30, scen_lange_sessies ....

## Wat deze test niet zegt

- Niets over de kwaliteit van de gegenereerde code. Er is capaciteit en latentie gemeten, niet of het model goede antwoorden geeft.
- Niets over andere inferentie-engines. vLLM was al gekozen.
- De simulatie gebruikt echte broncode uit een publiek voorbeeldproject en een gedragsmodel met vier persona's. Het gedrag van een echte klas wijkt af; de persona-verdeling staat daarom in `config.json` en is aan te passen.

---

Gegenereerd door het stresstestharnas, versie 1.0.0. Configuratie: `config.json` naast dit bestand.
