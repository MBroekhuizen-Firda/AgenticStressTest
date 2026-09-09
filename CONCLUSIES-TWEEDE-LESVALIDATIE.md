# Tweede lesvalidatie: hetzelfde lesuur, een eerlijker klas

Gelezen uit `results-van-de-gpu/20260909-125855_les/` (één run van negentig
minuten, 3.891 verzoeken), afgezet tegen de eerste lesvalidatie in
`results-van-de-gpu/20260909-005751_les/` (5.202 verzoeken). Alles hieronder is
terug te rekenen uit `runs/les_90min/requests.csv`, `bursts.csv` en
`server_metrics.csv` in die twee mappen.

Dit is de controlemeting die
[OPZET-VERVOLGMETING.md §2](OPZET-VERVOLGMETING.md#2-één-variabele-tegelijk--dit-is-nu-het-grootste-risico)
vraagt: het nieuwe gedragsmodel nog één keer op dezelfde kaart, zodat een
latere meting op andere hardware ergens tegen af te zetten is. Dezelfde
RTX PRO 6000, dezelfde vLLM-vlaggen, dezelfde seed, dezelfde lesopzet van
negentig minuten met twintig studenten op 32k context en de helft gedeeld.
Alleen het model van de klas is anders.

**De matrix is níét opnieuw gedraaid.** De map
`results-van-de-gpu/20260908-180708_matrix/` in deze push is op de regeleindes
en één toegevoegd veld in `analyse.json` na byte-voor-byte gelijk aan wat er al
stond. Alles in `RESULTATEN.md` dat uit de matrix komt — de sweep, de klifzoeker,
de engine-varianten, de kaartvergelijking — is dus nog steeds gemeten onder het
oude gedragsmodel. Zie [punt 6](#6-wat-er-nu-ontbreekt).

---

## De korte versie

1. **Het lesuur is niet meer groen, maar oranje.** De p90-doorlooptijd van één
   instructie gaat van 69 s naar 144 s; de groen-grens ligt op 90 s, de
   rood-grens op 180 s. Geen enkel verzoek mislukte, er was geen preemptie, en
   de KV-cache kwam niet boven 27 %.
2. **De kaart is niet langzamer geworden, de klas is zwaarder geworden.** Bij
   gelijke gelijktijdigheid decodeert de server even snel als in de eerste
   lesvalidatie — 113 tegen 116 tokens/s bij één verzoek, 85,8 tegen 85,8 bij
   twee. Het verschil zit in wat er gevraagd wordt, niet in wat de kaart levert.
3. **Vier studenten veroorzaken bijna een derde van de belasting.** De vier op
   het profiel `doorspitten` doen 12 % van de agentstappen en verbruiken 29 %
   van de rekentijd. Hun p90-doorlooptijd is 238 s — op zichzelf rood.
4. **Geheugen blijft de verkeerde zorg.** De piek ging van 8,5 GB naar 14,6 GB
   van de 54,9 GB pool. Ruim drie keer zoveel als dit zou de kaart nog aankunnen;
   de wachtrij liep wél op tot negen verzoeken, en dat is rekenkracht.
5. **Conclusie 1 van [CONCLUSIES.md](CONCLUSIES.md) is hiermee achterhaald, en
   conclusie 2 juist bevestigd.** "Een klas van twintig past, en niet nipt" gold
   voor een klas die te licht gemodelleerd was. Dat rekenkracht en niet geheugen
   de beperkende factor is, staat er nu twee keer zo scherp op.

---

## 1. Wat er veranderd is, en wat niet

Hetzelfde: kaart, model, vLLM-vlaggen (`--kv-cache-dtype fp8 --max-num-seqs 32
--max-model-len 131072 --enable-prefix-caching`), seed 20250908, twintig
studenten, 32k doelcontext, 50 % gedeelde projectbasis, dezelfde vijf fasen van
samen 5.400 seconden, dezelfde persona-verdeling (9 gemiddelde, 5 worstelaars,
3 doorpakkers, 3 afhakers).

Anders zijn drie dingen:

| | Eerste lesvalidatie | Tweede lesvalidatie |
|---|---|---|
| Gedragsmodel | één as: persona | twee assen: persona × werkprofiel |
| Corpus | `web`, 187 bestanden, 332k tekens | `web` + `unity`, 559 bestanden, 2,02M tekens |
| Contexttelling | schatting, 4,65 tekens/token | exact, tokenizer van het model |
| Attention-backend | niet vastgelegd (`null`) | `TRITON_ATTN` |

De werkprofielen zijn het zwaartepunt. Een persona bepaalt *tempo* — hoe lang
iemand nadenkt, hoeveel stappen een instructie kost, of hij afhaakt. Een
werkprofiel bepaalt *gewicht* — op welke codebase iemand werkt, hoeveel
bestanden de agent per keer leest, hoeveel het model per stap schrijft. De twee
worden onafhankelijk geloot en dan gepaard, zodat de zware codebase niet
automatisch bij de doorpakkers terechtkomt. De klas kreeg 10 × `klein`,
6 × `middel`, 4 × `doorspitten`.

De contexttelling bleek achteraf de kleinste van de drie wijzigingen. De eigen
telling van het harnas ligt in beide runs onder wat vLLM terugrapporteert —
6,0 % in de eerste run, 5,5 % in de tweede. De schatting van 4,65 tekens per
token zat er dus maar een half procentpunt naast; het restant van 5,5 % is de
opmaak van het chatsjabloon, die aan clientzijde niet wordt meegeteld en dus
ook met een exacte tokenizer blijft staan. **Alle geheugenuitspraken uit de
eerste meting blijven daarmee overeind** — die stonden al op de door de server
gerapporteerde aantallen.

## 2. Het oordeel kantelt naar oranje

| | Eerste | Tweede |
|---|---|---|
| Oordeel (streng) | groen | **oranje** |
| Oordeel volgens de opdracht | groen | groen |
| p90 doorlooptijd instructie | 69,0 s | **144,4 s** |
| p50 doorlooptijd instructie | 30,6 s | 51,0 s |
| p99 doorlooptijd instructie | 116 s | 268 s |
| p90 TTFT | 0,33 s | 0,75 s |
| p99 TTFT | 1,53 s | 4,03 s |
| decodesnelheid per stream (p50) | 53,4 tok/s | 41,0 tok/s |
| Verzoeken / instructies | 5.202 / 618 | 3.891 / 483 |
| Mislukte verzoeken | 0 | 0 |
| Preempties | 0 | 0 |
| Prefix cache hit rate | 96,5 % | 94,0 % |
| KV-piek | 8,5 GB (15,4 %) | 14,6 GB (26,6 %) |
| Wachtrijpiek | 2 verzoeken | 9 verzoeken |
| Serverdoorvoer (decode) | 230 tok/s | 256 tok/s |

Eén ding valt daarbij op: de server levert *méér* tokens per seconde dan in de
eerste run (256 tegen 230) en is toch trager per student. Dat is precies het
beeld van een verzadigde kaart. De doorvoer zit tegen zijn plafond, en wie er
een student bij zet krijgt geen extra doorvoer maar een kleiner deel van
dezelfde taart.

De rekensom is na te lopen. Een instructie is een reeks agentstappen en verder
niets: in beide runs bestaat de mediane instructie voor 100 % van zijn
wachttijd uit modelaanroepen, er zit geen denkpauze in. Het aantal stappen per
instructie is ook niet veranderd — gemiddeld 8,4 tegen 8,1, uit dezelfde
persona-verdeling. Wat veranderde is de stap zelf:

| | Eerste | Tweede | Verandering |
|---|---|---|---|
| Outputtokens per stap (mediaan) | 172 | 217 | +26 % |
| Decodesnelheid per stream (p50) | 53,4 tok/s | 41,0 tok/s | −23 % |
| Gemiddelde staptijd binnen een instructie (p50) | 4,00 s | 6,66 s | +67 % |

Die +67 % op de stap is precies de +67 % op de instructie: 30,6 s wordt 51,0 s.
In de staart is het erger — de p90 van de gemiddelde staptijd binnen een
instructie gaat van 6,2 s naar 17,8 s.

Ongeveer een kwart van de vertraging komt dus doordat het model per stap meer
schrijft, en de rest doordat elke stap zijn snelheid met meer gelijktijdige
streams moet delen. Die twee versterken elkaar: langere stappen betekenen dat
studenten elkaar vaker overlappen, en meer overlap maakt de stappen weer langer.
De meettijd waarin vijf of meer verzoeken tegelijk liepen ging van 43 % naar
66 %; de tijd met acht of meer van 11 % naar 38 %.

### Waar in het lesuur het misgaat

| Fase | Duur | Intensiteit | p50 instructie | p90 instructie | KV-piek | Wachtrijpiek |
|---|---|---|---|---|---|---|
| opstartpiek | 8 min | 1,5 | 50 s | 129 s | 8,7 GB | 0 |
| opbouw | 27 min | 1,0 | 44 s | 96 s | 11,1 GB | 1 |
| klassikale uitleg | 10 min | — | — | — | 5,3 GB | 0 |
| **eindpiek** | **35 min** | **1,6** | **70 s** | **179 s** | **14,6 GB** | **9** |
| afbouw | 10 min | 0,6 | 20 s | 49 s | 9,2 GB | 0 |

De eindpiek is het probleem, en hij komt tot op één seconde van de rood-grens
van 180 s. Vijfendertig minuten lang duurt een op de tien instructies bijna drie
minuten. In de rustige afbouw is dezelfde klas op dezelfde kaart ruim groen.

Ter vergelijking stond de eindpiek in de eerste lesvalidatie op 90,5 s — ook
daar de zwaarste fase, maar precies op de groen-grens.

### Na de tien minuten stilte

De hervatting na de klassikale uitleg blijft goedkoop: in de eerste twee minuten
na de stilte is de mediane TTFT 0,21 s, gelijk aan de eerste run. De p90 loopt
op van 0,40 s naar 1,31 s en de duurste eerste stap kostte 9,98 s tegen 2,02 s.
Dat is meer, maar het is nog altijd een factor twee onder de groen-grens van
20 s: de prefix cache overleeft tien minuten niets-doen ruimschoots.

## 3. Wie de vertraging veroorzaakt

Het werkprofiel verklaart het oordeel bijna alleen. Per profiel, over alle 484
instructies van de meetfasen:

| Werkprofiel | Studenten | Instructies | p50 | p90 | langste | instructies met compactie | p90 context |
|---|---|---|---|---|---|---|---|
| `klein` (web) | 10 | 289 | 42 s | 100 s | 193 s | 38 % | 32.138 tok |
| `middel` (web) | 6 | 119 | 61 s | 159 s | 340 s | 64 % | 35.926 tok |
| `doorspitten` (unity) | 4 | 76 | 107 s | **238 s** | 315 s | 86 % | 63.318 tok |

Op zichzelf beoordeeld is `klein` oranje (net boven 90 s), `middel` oranje en
`doorspitten` rood. Er is dus geen deel van deze klas dat nog groen is — ook de
tien studenten op de kleinste opdracht niet.

Wat die vier studenten kosten, staat los van hoe vaak ze iets vragen:

| Werkprofiel | Studenten | Aandeel agentstappen | Aandeel rekentijd | Outputtokens per stap |
|---|---|---|---|---|
| `klein` | 10 (50 %) | 64 % | 43 % | 211 |
| `middel` | 6 (30 %) | 24 % | 28 % | 392 |
| `doorspitten` | 4 (20 %) | 12 % | **29 %** | 758 |

Een student op de Unity-codebase kost ongeveer 1,7 keer zoveel rekentijd als een
student op de kleine webopdracht, terwijl hij minder vaak iets vraagt. Dat is
het getal om te onthouden bij het inrichten van een les: **de zwaarte van de
opdracht weegt zwaarder dan het aantal studenten.** Twintig studenten die
allemaal aan een klein project werken zijn iets heel anders dan twintig
studenten van wie er vier in een grote codebase graven.

## 4. De backend is niet de verklaring

Deze run draaide op `--attention-backend TRITON_ATTN`, omdat FlashInfer op deze
pod niet wilde bouwen (zie commit b0f4af6 en de toelichting in `scripts/pod.sh`).
De eerste lesvalidatie draaide vóór die uitwijkroute bestond en legde geen
backend vast — `"attention_backend": null`. Er zit dus een tweede, ongecontroleerde
verandering tussen de twee runs, en het rapport zou niets waard zijn als dat
niet werd nagegaan.

Het is na te gaan, en het antwoord is geruststellend. De decodesnelheid per
stream is per niveau van gelijktijdigheid vrijwel identiek:

| Gelijktijdige verzoeken | Eerste run (p50) | Tweede run (p50) |
|---|---|---|
| 1 | 113,2 tok/s | 115,9 tok/s |
| 2 | 85,8 tok/s | 85,8 tok/s |
| 3 | 69,4 tok/s | 71,2 tok/s |
| 4 | 61,9 tok/s | 59,1 tok/s |
| 5 | 52,4 tok/s | 51,1 tok/s |
| 6 | 50,7 tok/s | 47,0 tok/s |
| 8 | 45,7 tok/s | 39,1 tok/s |

Tot en met vijf gelijktijdige verzoeken lopen de curves over elkaar heen; de
mediane TTFT is in beide runs 0,208 s. Vanaf zes loopt de tweede run 7 tot 15 %
achter, en dat is te verwachten zonder de backend erbij te halen: die verzoeken
dragen grotere contexten mee, en decoderen wordt duurder naarmate de KV-cache
per sequentie langer is.

**De kaart doet dus hetzelfde als de vorige keer. Wat verschoof is waar de klas
op die curve zit.** In de eerste run lag het zwaartepunt op drie tot zeven
gelijktijdige verzoeken, in de tweede op acht en meer.

Dat neemt niet weg dat de backend voor een volgende vergelijking wél
vastgelegd moet worden. Pin hem expliciet — `pod.sh` accepteert een
`--attention-backend` — zodat de volgende meting geen tweede onbekende meesleept.

## 5. Contextdruk: het getal dat er eerst niet was

Nieuw in deze run is dat compactie geteld wordt. Een sessie die over zijn
doelcontext heen groeit, gooit op de grens van een instructie het oudste stuk
werkgeschiedenis weg en houdt de kop en de laatste beurten. De gedeelde
projectbasis blijft daarbij staan, anders zou hij geen prefix meer zijn.

Over het lesuur: **250 compacties, 6,4 per 100 agentstappen, hoogst bereikte
context 81.631 tokens** — meer dan het dubbele van de 32k waar de sessies op
gemikt zijn.

Dat laatste getal verdient uitleg, want het ziet eruit als een fout en is het
niet. Compactie gebeurt alleen op een instructiegrens, nooit middenin. Een
student op `doorspitten` die in één instructie vijf Unity-bestanden laat lezen,
groeit binnen die instructie ver over zijn doel heen en wordt pas bij de
volgende opdracht teruggesnoeid. Vandaar ook dat 86 % van hun instructies met
een compactie begint tegen 38 % bij `klein`.

De prijs staat in de cijfers hierboven: compactie kost de cache-ingang van alles
achter de bewaarde kop, dus de eerste stap na een compactie moet opnieuw
voorgerekend worden. Dat is precies waar de daling van de prefix cache hit rate
vandaan komt — 96,5 % naar 94,0 % — en waar de hogere p99-TTFT van 4,0 s zit.
Het is geen storing; het is wat een echte agent op een volle context ook doet.

## 6. Wat er nu ontbreekt

Deze push bevat één nieuwe run. Wat er níét in zit, en wat de conclusies
daardoor open laat:

- **De matrix onder het nieuwe gedragsmodel.** De sweep over context en aantal
  studenten, de klifzoeker, de engine-varianten en de kaartvergelijking in
  `RESULTATEN.md` staan nog allemaal op de oude, te lichte klas. De zin "een klas
  van 20 komt in de zwaarste geteste contextgrootte uit op groen" in
  `RESULTATEN.md` gaat over die oude runs; de lesvalidatie eronder zegt nu
  oranje. Dat is geen tegenspraak in de meting maar twee gedragsmodellen in
  hetzelfde bestand, en dat moet niet zo blijven.
- **Waar de grens nu ligt op de studenten-as.** Onder het oude model was er tot
  veertig studenten geen klif. Onder het nieuwe model is twintig al oranje. Waar
  vijftien of tien uitkomt, is niet gemeten — en dat is de vraag die een
  opleiding het eerst stelt.
- **Of de eindpiek representatief is.** De intensiteit van 1,6 in die fase is een
  aanname uit `config.json`, geen waarneming aan een echte klas. Het oordeel over
  dit hele lesuur hangt aan die ene fase: zonder de eindpiek is de run groen.

De goedkoopste manier om het eerste en tweede gat te dichten is dezelfde matrix
opnieuw op deze kaart, ongeveer € 13 aan huur volgens
[OPZET-VERVOLGMETING.md](OPZET-VERVOLGMETING.md). Doe dat vóór de meting op twee
RTX 5090's, niet erna: anders is die meting opnieuw met niets te vergelijken.

## 7. Wat dit met de aankoopvraag doet

De aanvraag gaat over 96 GB tegen 72 GB of twee 5090's. Deze run verschuift daar
één ding in, en het is niet het geheugen.

- **Op geheugen verandert er niets wezenlijks.** De piek van dit lesuur is
  14,6 GB van de 54,9 GB pool. Dat past ruim in de 33,7 GB van een RTX PRO 5000
  en ook in de 26,5 GB van twee 5090's. Er is nog steeds geen gemeten scenario
  waarin een klas van twintig aan een normale les 96 GB nodig heeft.
- **Op rekenkracht wordt het argument sterker in plaats van zwakker.** Het
  oranje van deze run is er een van doorlooptijd, gehaald op de snelste kaart uit
  de vergelijking, met de cachepool voor driekwart leeg en de wachtrij op negen.
  Een kaart met minder rekenkracht en minder bandbreedte doet dit niet beter.
  Dat maakt de meting op twee 5090's niet minder nodig maar meer: de kans dat die
  opstelling onder dit gedragsmodel rood kleurt, is nu aanzienlijk.
- **De eerlijke samenvatting blijft dezelfde als in
  [CONCLUSIES.md](CONCLUSIES.md), met één woord anders.** Deze meting weerlegt de
  onderbouwing van € 37.400 op *geheugen*gronden. Ze onderbouwt hem niet op
  rekenkrachtgronden — daarvoor moet je weten wat het alternatief doet, en dat is
  nog steeds niet gemeten. Wat ze wél laat zien is dat er op de rekenkracht-as
  minder marge is dan de eerste meting suggereerde.

---

## Wat deze meting niet zegt

- **Niets over andere hardware.** Eén kaart, één run.
- **Niets over de kwaliteit van de gegenereerde code.** Gemeten zijn capaciteit
  en latentie.
- **Niets over een klas die kleiner is dan twintig, of over andere
  contextgroottes onder dit gedragsmodel.** Alleen het lesuur is opnieuw
  gedraaid.
- **Niets hards over de verdeling van de werkprofielen.** 50/30/20 is een keuze
  in `config.json`, geen waarneming. Wie denkt dat zijn klas meer of minder in
  grote codebases werkt, verschuift daarmee het oordeel — punt 3 laat zien hoe
  gevoelig dat is.
- **Eén run.** Er is geen herhaling, dus geen spreiding tussen runs bekend. Het
  verschil met de eerste lesvalidatie is groot genoeg om niet aan toeval te
  liggen, maar de afstand tot de rood-grens (144 s tegen 180 s) is dat niet
  vanzelfsprekend.
