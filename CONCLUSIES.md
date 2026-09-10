# Conclusies uit de meting op de RTX PRO 6000 Blackwell

Gelezen uit `results-van-de-gpu/20260909-185135_matrix/` (38 runs, 8.361
verzoeken) en `results-van-de-gpu/20260909-125855_les/` (de lesvalidatie van
negentig minuten, 3.891 verzoeken). Samen 39 runs, 12.252 verzoeken en 1.774
instructies. Alles hieronder is terug te rekenen uit `summary.csv` en de
`bursts.csv`, `requests.csv` en `server_metrics.csv` per run.

Dit bestand is de tekstversie van [`RAPPORT-RTX-PRO-6000.html`](RAPPORT-RTX-PRO-6000.html)
en staat naast de automatisch gegenereerde [`RESULTATEN.md`](RESULTATEN.md).

> **Waarom hier maar één meetsessie in staat.** Er zijn oudere metingen op deze
> kaart — de matrix van 8 september en de eerste lesvalidatie — en die tellen
> niet meer mee. Sinds die runs is de belasting veranderd: het corpus ging van
> 187 naar 559 bestanden (332k naar 2,02M tekens), de contexttelling ging van
> een schatting naar de tokenizer van het model, en het gedragsmodel kreeg een
> tweede as (werkprofiel naast persona). Dezelfde kaart met een andere belasting
> levert geen vergelijking op, alleen verwarring. De lesvalidatie van 9 september
> en de matrix van 9 op 10 september draaien op exact dezelfde belasting en
> vormen samen één meting.

---

## De korte versie

1. **De klas past ruim in het geheugen en niet ruim in de rekenkracht.** Een klas
   van twintig op 32k context vraagt 8,0 GB van de 54,9 GB cachepool. Het lesuur
   van negentig minuten komt niettemin op **oranje** uit, op doorlooptijd: p90
   144 s tegen een groen-grens van 90 s. In geen van de 39 runs is er ook maar
   één preemptie of één mislukt verzoek geweest.
2. **Hoe hard de klas duwt weegt zwaarder dan hoe groot hij is.** Twintig
   studenten op 32k met dezelfde opdrachten leveren 65 s bij een rustige groep,
   76 s bij een normale, 120 s bij een gedreven groep en 285 s in de
   deadlineburst. Groen tot rood zonder dat er één student bijkomt.
3. **De contextgrootte is de tweede as, en die kost wél geheugen.** Bij twintig
   studenten loopt de p90-doorlooptijd van 65 s op 8k naar 164 s op 100k, en de
   cachepiek van 4,4 naar 19,1 GB. De twee assen versterken elkaar: bij 8k kost
   dertig studenten in plaats van vijf 71 % extra doorlooptijd, bij 100k 444 %.
4. **Op de studenten-as is geen klif maar een helling — en de klifzoeker meldt
   hem te laat.** Nagerekend ligt de omslag van groen naar oranje rond de twintig
   studenten; de live-uitslag van het harnas ("niets gebroken tot 40") is een
   meetartefact. Zie [vraag 2](#2-waar-ligt-de-grens-op-de-studenten-as).
5. **De 96 GB wordt in geen enkel realistisch scenario gebruikt.** Het lesuur zat
   op 14,6 GB, de zwaarste realistische run op 19,1 GB. Alleen de opzettelijk
   extreme worst case (46,2 GB) en dertig studenten op 100k (28,2 GB) passen niet
   op de goedkoopste opstelling.
6. **De volgende meting is belangrijker dan deze.** Over de rekenkracht van de
   alternatieven — de as die volgens punt 1 tot en met 3 bepalend is — zegt deze
   meting niets. Dezelfde matrix op twee RTX 5090's kost ongeveer € 12 en is het
   experiment dat het aankoopbesluit werkelijk beslist. Zie
   [OPZET-VERVOLGMETING.md](OPZET-VERVOLGMETING.md).

---

## 1. Past een klas van twintig?

Dat hangt af van wat die klas doet. Op de sweep — zeven minuten normaal tempo,
de helft van de projectbasis gedeeld — wel:

| Run | Studenten | Context | Oordeel | p90 TTFT | p90 doorlooptijd | KV-piek | Preempties |
|---|---|---|---|---|---|---|---|
| sweep | 20 | 8k | groen | 0,40 s | 65 s | 4,4 GB (8 %) | 0 |
| sweep | 20 | 32k | groen | 0,65 s | 76 s | 8,0 GB (15 %) | 0 |
| sweep | 20 | 64k | oranje | 0,77 s | 102 s | 10,4 GB (19 %) | 0 |
| sweep | 20 | 100k | oranje | 0,93 s | 164 s | 19,1 GB (35 %) | 0 |
| **Lesvalidatie 90 min** | **20** | **32k** | **oranje** | **0,75 s** | **144 s** | **14,6 GB (27 %)** | **0** |

De lesvalidatie is het zwaarstwegende getal in dit hele rapport: negentig minuten
aaneengesloten, met aankomstpatroon, een opstartpiek, tien minuten klassikale
uitleg, een eindpiek van vijfendertig minuten en een afbouw — in plaats van een
meetvenster van vijf minuten op één intensiteit. Diezelfde klas op diezelfde
context is in de sweep groen (76 s) en over een heel lesuur oranje (144 s). Het
verschil zit volledig in de fasen: de eindpiek alleen komt op 179 s uit, één
seconde onder de rood-grens.

| Fase van het lesuur | Duur | Intensiteit | p50 instructie | p90 instructie |
|---|---|---|---|---|
| opstartpiek | 8 min | 1,5 | 50 s | 129 s |
| opbouw | 27 min | 1,0 | 45 s | 96 s |
| klassikale uitleg | 10 min | — | — | — |
| **eindpiek** | **35 min** | **1,6** | **70 s** | **179 s** |
| afbouw | 10 min | 0,6 | 20 s | 49 s |

**Wie de vertraging veroorzaakt.** Het werkprofiel verklaart het oordeel bijna
alleen. Over de 483 instructies van de meetfasen:

| Werkprofiel | Studenten | Instructies | p50 | p90 | Aandeel stappen | Aandeel rekentijd | Outputtokens per stap | Met compactie |
|---|---|---|---|---|---|---|---|---|
| `klein` (web) | 10 | 288 | 42 s | 100 s | 64 % | 43 % | 211 | 38 % |
| `middel` (web) | 6 | 119 | 61 s | 159 s | 24 % | 28 % | 392 | 64 % |
| `doorspitten` (unity) | 4 | 76 | 107 s | **238 s** | 12 % | **29 %** | 758 | 86 % |

Op zichzelf beoordeeld is `klein` oranje (net boven 90 s), `middel` oranje en
`doorspitten` rood. Er is geen deel van deze klas dat nog groen is — ook de tien
studenten op de kleinste opdracht niet. Een student op de Unity-codebase kost
ongeveer 1,7 keer zoveel rekentijd als een student op de kleine webopdracht,
terwijl hij mínder vaak iets vraagt. **De zwaarte van de opdracht weegt zwaarder
dan het aantal studenten.**

## 2. Waar ligt de grens op de studenten-as?

De klifzoeker begint met vijf studenten en zet er elke twee minuten twee bij, tot
het plafond van veertig. Hij stopt zodra de p90-doorlooptijd boven 180 s komt, de
p90-TTFT boven 45 s of het foutpercentage boven 2 %. Dat gebeurde niet: de run
haalde het plafond.

**Die uitslag is te rooskleurig, en dat is na te rekenen.** De klifzoeker
beoordeelt elk venster van twee minuten op de instructies die op dat moment *af*
zijn. Instructies die nog lopen tellen niet mee — en dat zijn precies de trage.
Vanaf 23 studenten valt daardoor bijna de helft van de instructies buiten de
beoordeling: 87 van de 183. Zet je alles wat in hetzelfde venster begón naast
elkaar, dan ziet het er anders uit:

| Studenten | Instructies | p50 | p90 (alles) | p90 (wat de klifzoeker zag) | Boven 90 s | Oordeel |
|---|---|---|---|---|---|---|
| 5 – 11 | 27 | 12,1 s | 30,2 s | 28,0 s | 0 % | groen |
| 13 – 19 | 53 | 18,4 s | 34,4 s | 32,5 s | 0 % | groen |
| 21 – 27 | 63 | 29,4 s | **92,0 s** | 47,1 s | 13 % | oranje |
| 29 – 35 | 74 | 49,5 s | 125,2 s | 73,4 s | 22 % | oranje |
| 37 – 40 | 63 | 46,7 s | 114,3 s | 74,2 s | 17 % | oranje |

**Op 32k context en normaal tempo ligt de omslag van groen naar oranje rond de
twintig studenten.** Daarna wordt het niet zozeer erger als wel grilliger: van 29
tot 40 studenten blijft de p90 tussen 114 en 125 s hangen zonder ooit rood te
worden. De servermetingen laten zien waar de knik zit — bij 23 studenten meer dan
verdubbelt de cachebezetting van 6,7 % naar 14,7 % en loopt het aantal
gelijktijdig lopende verzoeken van vijf à zes naar tien. Daarboven verdeelt vLLM
dezelfde doorvoer over meer streams; de wachtrij komt niet verder dan twee
verzoeken.

Er is dus geen klif in de zin van een punt waarop het omvalt. Er is een helling,
en de kaart glijdt er langzaam vanaf. Voor een opleiding is dat het gunstigere
geval: een klas die te groot wordt levert langzaam ongemak op, geen storing.

> De automatisch gegenereerde [`RESULTATEN.md`](RESULTATEN.md) herhaalt de
> ongecorrigeerde uitslag ("de oplopende run liep tot het ingestelde plafond van
> 40 studenten en was daar nog groen"), omdat het harnas zelf nog niet
> gerepareerd is. Lees die zin met de tabel hierboven ernaast.


**Waarom rekenkracht en niet geheugen.** De kaart raakt in totale doorvoer niet
verzadigd, maar hij schaalt langzamer dan de klas:

| Studenten (32k) | Server totaal (decode) | Per stream (p50) | p90 instructie |
|---|---|---|---|
| 5 | 84 tok/s | 113 tok/s | 49 s |
| 10 | 153 tok/s | 72 tok/s | 76 s |
| 20 | 255 tok/s | 60 tok/s | 76 s |
| 30 | 360 tok/s | 47 tok/s | 131 s |

Zes keer zoveel studenten leveren ruim vier keer zoveel tokens per seconde, en
dus houdt elke student er minder aan over. De KV-cache komt in deze hele kolom
niet boven 9,3 GB van de 54,9 GB.

## 3. Hoe hard de klas duwt

Dit is het belangrijkste resultaat van de matrix, en het staat in geen enkele
tabel van de opdracht. Zeven runs, allemaal twintig studenten op 32k context met
de helft gedeeld, dezelfde kaart, dezelfde vlaggen en exact dezelfde verdeling
van werkprofielen (10 × `klein`, 6 × `middel`, 4 × `doorspitten`):

| Belasting | p90 instructie | Oordeel |
|---|---|---|
| rustige klas (`act_rustig_s20`) | 65 s | groen |
| normale klas (`sweep_s20_c32k`) | 76 s | groen |
| gedreven klas (`act_intensief_s20`) | 120 s | oranje |
| heel het lesuur (`les_90min`) | 144 s | oranje |
| hervatten na tien minuten stilte (`scen_na_de_stilte`) | 166 s | oranje |
| eindpiek van het lesuur (`les_90min`, fase 4) | 179 s | oranje |
| deadline (`scen_deadline`) | 285 s | **rood** |

Dat gebeurt in de matrix op twee manieren. De eerste drie runs verschillen in de
*samenstelling* van de klas: het activiteitsniveau verschuift de personaverdeling
van 5 % doorpakkers en 30 % afhakers (rustig) via 15/15 (normaal) naar 40 %
doorpakkers en 3 % afhakers (intensief). De laatste vier verschillen in de
*fase-intensiteit*: een vermenigvuldiger die de denktijd van iedereen deelt, 1,0
in een gewone fase en 2,0 in de deadlineburst.

**De praktische betekenis: de vraag "past een klas van twintig?" heeft geen
antwoord zonder de vraag "wat doet die klas?"** Een werkcollege waarin studenten
om beurten iets vragen zit ruim in het groen. Hetzelfde lokaal in het laatste half
uur voor de inlevering zit in het rood, met dezelfde hardware en dezelfde opdracht.

## 4. Zou minder videogeheugen ook volstaan?

Zevenendertig van de negenendertig runs blijven onder de helft van de cachepool;
dertig blijven onder een kwart.

| Kaart | Cachepool | Nodig (lesuur) | Nodig (klas van 20, 100k) | Nodig (worst case) | Past? | Prijs |
|---|---|---|---|---|---|---|
| 2× RTX 5090 (64 GB) | 26,5 GB | 14,6 GB | 19,1 GB | 46,2 GB | 2 runs niet | ca. € 10.000 |
| RTX PRO 5000 (72 GB) | 33,7 GB | 14,6 GB | 19,1 GB | 46,2 GB | 1 run niet | € 7.602 |
| RTX PRO 6000 (96 GB) | 54,9 GB | 14,6 GB | 19,1 GB | 46,2 GB | alle 39 | € 37.400 |

De twee runs die niet op de goedkoopste opstelling passen zijn `scen_worst_case`
(46,2 GB) en `sweep_s30_c100k` (28,2 GB). De worst case is twintig studenten,
maximale context, geen enkele gedeelde projectbasis — met opzet extreem, want bij
een klas die aan dezelfde opdracht werkt kómt die 0 % niet voor, maar het is wel
het scenario dat de opdracht zelf als grens benoemt. `sweep_s30_c100k` is in deze
meting al rood op doorlooptijd, dus of je hem op geheugen wilt kunnen bedienen is
een aparte vraag. Wie beide meerekent koopt geen 72 GB; wie ze weglaat heeft aan
de kleinste pool genoeg, met het lesuur op 14,6 GB dat er bijna twee keer in past.

**Belangrijker dan die keuze:** deze tabel gaat alléén over geheugen, en volgens
vraag 1 tot en met 3 is geheugen niet de beperkende factor. De goedkopere kaarten
hebben minder rekenkracht en minder geheugenbandbreedte, en dat raakt precies de
doorlooptijd die het oordeel bepaalt. De twee 5090's hebben er nog een risico
bij: geen NVLink, dus het verkeer tussen de kaarten loopt over PCIe, en bij een
MoE-model als dit merk je dat pas onder gelijktijdige belasting.

**De eerlijke conclusie is niet "koop de goedkope kaart", maar: deze meting
weerlegt de onderbouwing van € 37.400 op geheugengronden, tenzij de worst case
meetelt, en het alternatief is nog niet gemeten.** Het oranje van dit lesuur is
gehaald op de snelste kaart uit de vergelijking, met de cachepool voor driekwart
leeg. Een kaart met minder rekenkracht doet dit niet beter. Dat maakt de meting op
twee 5090's niet minder nodig maar méér: de kans dat die opstelling onder dit
gedragsmodel rood kleurt, is aanzienlijk.

## 5. Welke vLLM-instellingen zijn bepalend?

Eén instelling doet er echt toe, de rest niet. Alle varianten bij twintig
studenten op 64k context; vermeld is alleen de vlag die afwijkt van
`--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 131072`.

| Variant | Oordeel | p90 TTFT | p90 instructie | KV-piek | Cache hit rate |
|---|---|---|---|---|---|
| `--kv-cache-dtype auto` | oranje | 0,68 s | 166 s | **23,2 GB (42 %)** | 97,0 % |
| `--kv-cache-dtype fp8` | oranje | 0,72 s | 99 s | 10,5 GB (19 %) | 96,9 % |
| `--max-model-len 65536` | oranje | 0,67 s | 99 s | 10,6 GB (19 %) | 96,9 % |
| `--max-num-seqs 16` | oranje | 0,71 s | 101 s | 10,8 GB (20 %) | 96,9 % |
| `--max-num-seqs 64` | oranje | 0,66 s | 99 s | 10,7 GB (19 %) | 96,9 % |

**FP8-KV-cache brengt het cachegebruik terug tot minder dan de helft.** 10,5 GB
tegen 23,2 GB bij verder identieke instellingen, dezelfde hit rate, hetzelfde
oordeel. Dat `auto` ook 166 s duurt in plaats van 99 is één meting en geen bewezen
patroon — maar er is geen reden om fp8 níét te gebruiken.

**`--max-num-seqs` en `--max-model-len` doen in dit bereik niets.** Zestien,
tweeëndertig en vierenzestig sequenties geven 99 tot 101 seconden en 10,5 tot
10,8 GB; dat is ruis. Dat komt doordat de belasting nergens in de buurt van de
geheugengrens kwam. Op een pool van 26,5 GB kan dit heel anders liggen — sla
`--only engine` daar dus niet over.

**Aanbevolen combinatie:** `--kv-cache-dtype fp8 --max-num-seqs 32
--max-model-len 131072 --enable-prefix-caching`.

## Wat de gedeelde projectbasis oplevert: geheugen, geen snelheid

| Gedeeld | Context | Cache hit rate | p90 TTFT | KV-piek | Oordeel |
|---|---|---|---|---|---|
| 0 % | 32k | 92,6 % | 0,58 s | 9,1 GB | groen |
| 50 % | 32k | 93,9 % | 0,60 s | 8,0 GB | groen |
| 90 % | 32k | 95,5 % | 0,68 s | 6,9 GB | groen |
| 0 % | 64k | 96,0 % | 0,70 s | 15,8 GB | oranje |
| 50 % | 64k | 96,9 % | 0,78 s | 10,6 GB | oranje |
| 90 % | 64k | 97,7 % | 0,78 s | 9,3 GB | oranje |

Van niets delen naar bijna alles delen scheelt op 64k ruim veertig procent
cachegeheugen, en op het oordeel niets. Dat is logisch: het oordeel valt op
doorlooptijd, en de gedeelde prefix bespaart prefill, niet decode. De hit rate was
toch al hoog, en die komt niet van het delen tússen studenten maar van hergebruik
*binnen* één sessie — elke agentstap stuurt de vorige conversatie opnieuw mee.
Delen koopt dus marge bij het kiezen van een kaart, geen snelheid bij het
inrichten van een les.

## De benoemde momenten

| Moment | Studenten | Context | Oordeel | p90 TTFT | p90 instructie | KV-piek | Wachtrij |
|---|---|---|---|---|---|---|---|
| Koude start | 20 | 12k | groen | 0,32 s | 87 s | 4,6 GB | 0 |
| Koude start | 30 | 12k | oranje | 0,34 s | 144 s | 9,1 GB | 0 |
| Lange sessies, grote context | 6 | 110k | oranje | 1,09 s | 146 s | 12,0 GB | 0 |
| Na de stilte, iedereen hervat | 20 | 32k | oranje | 0,94 s | 166 s | 10,9 GB | 8 |
| Deadline, iedereen in burst | 20 | 32k | **rood** | 0,72 s | 285 s | 15,4 GB | 3 |
| Worst case, niets gedeeld | 20 | 100k | **rood** | 0,96 s | **401 s** | **46,2 GB** | 9 |

De hervatting na tien minuten stilte is niet als extreem scenario bedoeld: die
test de cache, en de cache doorstaat hem moeiteloos (hit rate 93,8 %, p90 TTFT
0,94 s). Dat hij toch op 166 s uitkomt en de langste wachtrij van alle 32k-runs
oplevert, komt puur doordat twintig studenten tegelijk hervatten. Het is
gelijktijdigheid, niet een koude cache — hetzelfde beeld als de klassikale uitleg
in het lesuur: goedkoop om uit te komen, duur om samen weer in te stappen.

De worst case is de enige run in de hele meting die het geheugen echt aanspreekt.

## Contextdruk: wat een vol venster kost

Een sessie die over zijn doelcontext heen groeit, gooit op de grens van een
instructie het oudste stuk werkgeschiedenis weg en houdt de kop en de laatste
beurten. Wat dat kost is de cache-ingang van alles achter de bewaarde kop: de
eerste stap daarna moet opnieuw voorgerekend worden.

| Doelcontext | Compacties per 100 stappen | Compacties | Hoogst bereikte context | Van het doel |
|---|---|---|---|---|
| 8k | **12,1** | 94 | 53.045 | **663 %** |
| 32k | 5,3 | 36 | 57.853 | 181 % |
| 64k | 2,5 | 13 | 67.571 | 106 % |
| 100k | 0,6 | 3 | 90.261 | 90 % |

Die laatste kolom ziet eruit als een fout en is het niet. Compactie gebeurt alleen
op een instructiegrens, nooit middenin. Een student die in één instructie vijf
Unity-bestanden laat lezen, groeit binnen die instructie ver over zijn doel heen
en wordt pas bij de volgende opdracht teruggesnoeid. In het lesuur gebeurde dat
250 keer, 6,4 keer per honderd agentstappen, met een hoogst bereikte context van
81.631 tokens tegen een doel van 32k. Daar zit ook de daling van de prefix cache
hit rate — 94,0 % in het lesuur tegen 96,9 % in de rustiger matrixruns — en de
hogere p99-TTFT van 4,0 s.

**Een krappe context is niet gratis.** 8k geeft de beste doorlooptijden van het
hele raster, maar tegen 12 compacties per honderd stappen: werk dat opnieuw
gedaan moet worden en dat in een echte agent ook informatie kost. 32k is in deze
meting het punt waar beide nog meevallen.

> Hoe snel een venster vol loopt hangt volledig af van hoeveel elke agentstap
> toevoegt. Dit gedragsmodel gaat uit van kleine stappen — een gelezen bestand uit
> dit corpus is een paar honderd tokens. Een agent die complete testlogs in de
> context kiepert, vult hetzelfde venster een orde van grootte sneller en
> verschuift deze hele tabel.

## Waarom de TTFT-getallen zo mooi zijn (en waar je ze niet op moet vertrouwen)

De gemeten p90-TTFT komt in geen enkele van de 39 runs boven 1,1 s uit, tegen een
groen-grens van twintig seconden. Volgens de oorspronkelijke definitie uit de
opdracht — die alleen naar time to first token kijkt — is **élke run groen**, de
worst case met instructies van bijna zeven minuten incluis. Volgens de
aangescherpte definitie is achttien keer oranje en vier keer rood. Wie op TTFT
stuurt, meet op deze schaal niets.

Dat getal is echt, maar het meet iets anders dan je denkt: de prompt zit al in de
prefix cache (87 tot 99 % hit rate), dus er valt bijna niets voor te rekenen. De
prefill zelf haalt bijna 18.000 tokens/s.

De ene plek waar de norm van twintig seconden wél in gevaar komt is de *eerste*
aanroep van een sessie met een koude cache, en die valt in de warmloopfase die
bewust niet meetelt:

| Run | Warmloopfase, p90 TTFT | Meetfase, p90 TTFT |
|---|---|---|
| `scen_worst_case` (20 studenten, 100k, niets gedeeld) | **141 s** | 0,96 s |
| `scen_lange_sessies` (6 studenten, 110k) | **24 s** | 1,09 s |

Dat is te herleiden: twintig studenten die tegelijk voor het eerst een context van
tienduizenden tokens openen, is meer dan een miljoen tokens puur voorrekenen
voordat de laatste student iets ziet. Voor een capaciteitsmeting is het
verdedigbaar om die fase weg te laten, maar het betekent wel dat er een gat zit:
**de koude start is alleen op 12k context beoordeeld** (daar is hij groen, 0,32 s
bij twintig studenten). Als het beeld is dat de klas 's ochtends tegelijk aan een
groot project begint, is dat het scenario dat nog gemeten moet worden.

## Wat er in het harnas nog niet klopt

Twee dingen kwamen bij het nalopen van deze meting boven en zijn nog niet
gerepareerd:

1. **De klifzoeker beoordeelt op de instructies die af zijn.** `_run_ramp` in
   `stresstest/runner.py` filtert `state.collector.bursts` op `start_t` binnen het
   venster, maar de collector bevat alleen instructies die op dat moment voltooid
   zijn. Onder belasting zijn dat precies de snelle, dus de live-uitslag is
   systematisch te optimistisch — bij deze run een factor twee vanaf 23 studenten.
   Twee mogelijke reparaties: het venster één stap laten achterlopen, of
   nog-lopende instructies meetellen met hun tot dan toe verstreken tijd als
   ondergrens.
2. **`ACTIVITY_INTENSITY` wordt nergens gelezen.** `stresstest/personas.py`
   definieert een denktijdfactor voor `rustig`, `normaal` en `intensief` (0,7 /
   1,0 / 1,6), maar geen enkele aanroeper gebruikt hem: de activiteitsniveaus
   verschuiven alleen de personaverdeling. Het verschil tussen 65 en 120 seconden
   in [vraag 3](#3-hoe-hard-de-klas-duwt) komt dus volledig uit die verdeling.
   Aanzetten maakt het effect groter, niet kleiner — maar zolang het dode code is,
   moet de tabel gelezen worden als "andere klas", niet als "hoger tempo".

## De correcties die eerder gemaakt zijn

Bij de eerste analyseronde trok de automatisch gegenereerde `RESULTATEN.md` op
drie punten een conclusie die uit de meetgegevens niet volgde. Alle drie zaten in
de analysecode, niet in de meting, en alle drie zijn gerepareerd — de cijfers
hierboven staan al op de gerepareerde versie.

1. **De alternatievenvergelijking rekende met de verkeerde piek**: de hoogste piek
   uit de sweep-runs in plaats van over álle runs. De analyse toont nu beide
   getallen en noemt per kaart wélke runs er niet op passen.
2. **`kv_auto` werd "beste variant" genoemd op ruis** — een TTFT-verschil van
   zestig milliseconden, terwijl het cachegebruik ruim twee keer zo hoog was. De
   rangschikking negeert nu TTFT-verschillen kleiner dan vijf procent van de
   groen-grens en kiest op cachebezetting.
3. **"De klif ligt bij 40 studenten" was het testplafond.** De analyse
   onderscheidt nu of de oplopende run brak of zijn eigen bovengrens raakte. Dat
   het antwoord daarmee nog niet klopt, staat hierboven onder
   [Wat er in het harnas nog niet klopt](#wat-er-in-het-harnas-nog-niet-klopt).

Bij het nalopen van de opzet kwamen daar drie dingen uit die de meting zelf
raakten, alle drie inmiddels gerepareerd: de tokenizer viel stil terug op schatten
omdat `pod.sh` `endpoint.model` op de `--served-model-name` zette; herlezen van
resultaten wiste `duration_s`; en de kaartnaam kreeg er per fase een "(600W)" bij.

Welke kaart de vervolgmeting moet krijgen en wat je daarvoor aanpast, staat in
[OPZET-VERVOLGMETING.md](OPZET-VERVOLGMETING.md).

## Wat deze meting niet zegt

- **Niets over andere hardware.** Eén kaart. De kaartvergelijking rekent alleen
  met geheugen; over de rekenkracht van de alternatieven — de bepalende factor —
  is niets gemeten.
- **Niets over de kwaliteit van de gegenereerde code.** Gemeten zijn capaciteit en
  latentie.
- **Niets over een koude start met grote context.** Alleen op 12k beoordeeld.
- **Niets hards over de verdeling van de werkprofielen.** 50/30/20 is een keuze in
  `config.json`, geen waarneming aan een echte klas. Vraag 1 laat zien hoe gevoelig
  het oordeel daarvoor is: vier van de twintig studenten dragen 29 % van de
  rekentijd.
- **Niets hards over de intensiteiten.** Dat de eindpiek van een lesuur op 1,6 zit
  en een deadline op 2,0 zijn aannames. Zonder de eindpiek is het lesuur groen.
- **Eén run per cel.** Er is geen herhaling, dus geen spreiding tussen runs
  bekend. De grote verschillen (65 tegen 285 s over de intensiteitsladder) zijn te
  groot om toeval te zijn; de kleine (99 tegen 101 s tussen engine-varianten) niet,
  en die zijn hierboven ook als ruis behandeld.
