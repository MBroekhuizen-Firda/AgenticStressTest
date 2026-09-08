# Resultaten stresstest agentic coding

> **Status: nog niet gemeten.**
>
> Dit bestand wordt automatisch gegenereerd zodra de test op echte hardware
> heeft gedraaid. Het harnas schrijft dan een ingevulde versie in de
> resultatenmap:
>
> ```
> results/<tijdstempel>_matrix/RESULTATEN.md
> ```
>
> Kopieer die over dit bestand heen als de meting definitief is. Tot dat moment
> beschrijft deze pagina alleen wat er straks komt te staan, en waarop de
> huidige aanvraag berust.

---

## Waarom dit bestand er nog leeg bij ligt

De aanvraag voor € 37.400 rust op schattingen, niet op metingen. Dit harnas is
gebouwd om die metingen te doen voor ongeveer € 20 aan gehuurde rekentijd. De
code is compleet en getest, maar er is nog geen GPU aan te pas gekomen: er staat
hier dus nog geen enkel gemeten getal, en er hoort er ook geen te staan tot de
test echt gedraaid heeft.

Zie [`README.md`](README.md), hoofdstuk 3 tot en met 5, voor hoe je dat doet.

## Wat er straks in staat

De gegenereerde versie beantwoordt de vier vragen uit de opdracht, in deze
volgorde en in gewone taal, geschreven voor wie de aanvraag moet beoordelen:

**De korte versie** — drie tot vijf regels: hoeveel studenten er passen, waar de
klif ligt, hoeveel geheugen er werkelijk in gebruik was, en wat de goedkoopste
geteste kaart is die volgens de meting voldoet.

**1. Past een klas van twintig op 96 GB, en met hoeveel marge?** Een tabel met
per contextgrootte het oordeel, de p90 time-to-first-token, de p90 doorlooptijd
van een volledige student-instructie en het aantal preempties. Daaronder de
marge in geheugen, uitgedrukt in gigabytes en in procenten van de cachepool.

**2. Waar ligt de klif?** Het maximale aantal gelijktijdige studenten uit de
oplopende run, met wat er brak bij de student daarboven. Plus per
contextgrootte het aantal studenten waarbij het oordeel omslaat naar oranje en
naar rood.

**3. Zou 72 GB of 64 GB ook volstaan?** De gemeten geheugenbehoefte afgezet
tegen de RTX PRO 5000 (72 GB, € 7.602) en twee RTX 5090's (64 GB, circa
€ 10.000), met per kaart of het past en hoeveel marge er overblijft. Met de
uitdrukkelijke kanttekening dat dit alleen over geheugen gaat: een kaart met
minder geheugen heeft doorgaans ook minder rekenkracht, wat de doorlooptijd
raakt ook als het geheugen past. Vandaar het advies om het alternatief dat
overblijft na te meten met dezelfde matrix voordat er besteld wordt.

**4. Welke vLLM-instellingen zijn bepalend?** Een vergelijking van
`--kv-cache-dtype fp8` tegen `auto`, van drie waarden voor `--max-num-seqs` en
van een beperkt tegen een ruim `--max-model-len`, met de aanbevolen combinatie.

Daarnaast: wat de gedeelde projectbasis oplevert (0 %, 50 % en 90 % gedeeld,
naast elkaar), hoe de vijf benoemde momenten uitpakken — koude start, deadline,
lange sessies, na de stilte, worst case — en het oordeel over de lesvalidatie
van negentig minuten.

## Waarop de huidige aanvraag berust

Dit zijn **schattingen**, geen metingen. Ze staan hier zodat straks te zien is
waar de meting van afweek.

| | Aanname in de aanvraag |
|---|---|
| Klasgrootte | 20 gelijktijdige studenten |
| Model | Qwen3-Coder-30B-A3B in FP8, circa 31 GB aan gewichten |
| Kaart | NVIDIA RTX PRO 6000 Blackwell, 96 GB |
| Bedrag | € 37.400 |
| Onderbouwing | schatting, niet gemeten |

Bij `--gpu-memory-utilization 0.90` blijft van 96 GB ongeveer 53 GB over voor de
KV-cache nadat de gewichten en de overhead eraf zijn. Hoeveel daarvan een klas
van twintig werkelijk nodig heeft, hangt vrijwel volledig af van twee dingen:
de contextlengte per student, en hoeveel van die context gedeeld is doordat de
klas aan dezelfde opdracht werkt. Precies die twee staan als assen in de matrix.

## Hoe de kleuren bepaald worden

- **Groen** — p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder
  90 s, decodesnelheid boven 12 tokens/s per stream, geen preempties, geen
  afgelopen verzoeken.
- **Oranje** — p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele
  preempties.
- **Rood** — daarboven, of aanhoudende preempties, of verzoeken die aflopen.

De grens van 20 seconden komt uit de functionele eisen van Firda en is
overgenomen. Twee dingen wijken af van de oorspronkelijke opzet, allebei bewust:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is
   de eerste van drie tot vijftien modelaanroepen die de assistent doet voor één
   opdracht van de student. Een run kan een prima TTFT hebben en toch twee
   minuten over één instructie doen; dat laatste is wat de student wacht. Ook de
   decodesnelheid telt mee: 20 seconden tot het eerste token gevolgd door vier
   tokens per seconde is geen groene ervaring.
2. **De prefix cache hit rate is gerapporteerd maar is geen groen-eis.** Die
   hangt af van het scenario, niet van de hardware: in de runs met 0 % gedeelde
   basis en in de koude start is 60 % per definitie onhaalbaar, en die runs
   afkeuren op dat getal zou het echte signaal verbergen. Een lage hit rate
   waar we een hoge verwachtten komt als waarschuwing bij de run te staan.

Preempties blijven een harde afwijzing. Elke preemptie betekent dat een sessie
uit de cache is gegooid en dat de volgende agentstap van die student de complete
context opnieuw moet doorrekenen — tientallen keren zo duur, precies op het
moment dat de student verder wil.

Om deze aanpassing controleerbaar te maken krijgt elke run **twee** oordelen:
één volgens bovenstaande definitie en één volgens de oorspronkelijke definitie
uit de opdracht. Beide staan in `summary.csv`, en de gegenereerde
`RESULTATEN.md` benoemt bij welke runs ze uiteenlopen.

## Wat deze test niet zegt

- Niets over de kwaliteit van de gegenereerde code. Er wordt capaciteit en
  latentie gemeten, niet of het model goede antwoorden geeft.
- Niets over andere inferentie-engines. vLLM is al gekozen.
- Het is een simulatie van een klas, met echte broncode als context en een
  gedragsmodel van vier persona's. Een echte klas wijkt af; daarom staat de
  persona-verdeling in `config/default.json` en is die aan te passen.
