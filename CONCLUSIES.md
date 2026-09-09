# Conclusies uit de meting op de RTX PRO 6000 Blackwell

Gelezen uit `results-van-de-gpu/20260908-180708_matrix/` (38 runs, 12.865
verzoeken) en `results-van-de-gpu/20260909-005751_les/` (de lesvalidatie van
negentig minuten). Alles hieronder is terug te rekenen uit `summary.csv` en de
`requests.csv` per run.

Dit bestand staat naast de automatisch gegenereerde `RESULTATEN.md`. De punten
waarop die tekst een andere conclusie trok dan de meetgegevens dragen, staan
onder [De correcties die hierop gemaakt zijn](#de-correcties-die-hierop-gemaakt-zijn);
ze zijn inmiddels in de analysecode verwerkt.

---

## De korte versie

1. **Een klas van twintig past, en niet nipt.** De lesvalidatie van negentig
   minuten is groen: 5.202 verzoeken, geen enkele mislukt, geen preemptie,
   p90-doorlooptijd van een instructie 69 s, en 8,5 GB van de 54,9 GB
   cachepool in gebruik. Ook alle vier de contextgroottes met twintig
   studenten zijn groen.
2. **Geheugen is niet de beperkende factor. Rekenkracht is dat.** Bij veertig
   studenten stond de cachepool op 15,6 % vol en was alles nog groen. De twee
   rode runs zijn rood op *doorlooptijd*, niet op geheugen. In geen van de 38
   runs is er ook maar één preemptie geweest.
3. **De 96 GB wordt in geen enkel realistisch scenario gebruikt.** De zwaarste
   realistische run zat op 15,7 GB, de lesvalidatie op 8,5 GB. Er is geen
   gemeten grond om 96 GB nodig te hebben voor een klas van twintig.
4. **Maar "72 GB volstaat dus ook" volgt er niet uit.** Tegen de
   *werkelijke* piek over alle runs (43,4 GB in de worst case) past een kaart
   van 72 GB níét. En over de rekenkracht van de goedkopere kaarten — de factor
   die volgens punt 2 bepalend is — zegt deze meting niets. Zie
   [vraag 3](#3-zou-minder-videogeheugen-ook-volstaan).
5. **De volgende meting is belangrijker dan deze.** De RTX PRO 5000 is niet te
   huren op RunPod, maar het andere alternatief uit de aanvraag wél: dezelfde
   matrix op twee RTX 5090's kost ongeveer € 12 en is het experiment dat het
   aankoopbesluit werkelijk beslist. Zie
   [OPZET-VERVOLGMETING.md](OPZET-VERVOLGMETING.md).

---

## 1. Past een klas van twintig?

Ja, met ruime marge.

| Run | Studenten | Context | Oordeel | p90 TTFT | p90 doorlooptijd | KV-piek | Preempties |
|---|---|---|---|---|---|---|---|
| Lesvalidatie 90 min | 20 | 32k | groen | 0,3 s | 69 s | 8,5 GB (15 %) | 0 |
| sweep | 20 | 8k | groen | 0,2 s | 25 s | 1,2 GB (2 %) | 0 |
| sweep | 20 | 32k | groen | 0,3 s | 36 s | 3,5 GB (6 %) | 0 |
| sweep | 20 | 64k | groen | 0,4 s | 40 s | 6,9 GB (13 %) | 0 |
| sweep | 20 | 100k | groen | 0,6 s | 50 s | 15,7 GB (29 %) | 0 |

De lesvalidatie is het zwaarstwegende getal in dit hele rapport: negentig
minuten aaneengesloten, met aankomstpatroon, klassikale uitleg en hervatting,
in plaats van een meetvenster van vijf minuten. Die run vertoont geen drift —
de KV-bezetting loopt niet op en de doorlooptijd blijft stabiel.

**Wat dit niet zegt:** de contextgroottes zijn *geschat* op tekens per token
(`tokenizer_exact: false` in `environment.json`), niet exact geteld. En een
"100k-run" groeit naar 100k toe; in het meetvenster van 300 s kwamen de
prompts tot ongeveer 75k tokens. De 100k-kolom is dus eerder optimistisch dan
pessimistisch.

## 2. Waar ligt de klif?

**Niet bij het aantal studenten — daar is hij niet gevonden.** De oplopende run
liep tot veertig studenten en was daar nog groen, met de cachepool op 15,6 %.
Hij stopte omdat `rampup.max_students` op 40 staat, niet omdat er iets brak.
De gegenereerde `RESULTATEN.md` schrijft "de klif ligt bij 40 studenten"; dat
is de bovengrens van de test, niet een gemeten grens.

**Wel bij de contextlengte.** Bij 100k slaat het oordeel om tussen twintig en
dertig studenten:

| Context | 5 studenten | 10 | 20 | 30 |
|---|---|---|---|---|
| 8k | groen | groen | groen | groen |
| 32k | groen | groen | groen | groen |
| 64k | groen | groen | groen | groen |
| 100k | groen | groen | groen | **rood** (197 s) |

Waarom rekenkracht en niet geheugen: de gezamenlijke decodesnelheid van de
server zit bij 32k rond de 200–300 tokens/s en loopt vanaf ongeveer twintig
studenten nauwelijks meer op. Per student daalt de snelheid dan navenant —
van 114 tokens/s bij vijf studenten naar 54 bij veertig. De extra student
krijgt geen extra doorvoer, hij deelt dezelfde taart.

| Studenten (32k) | Server totaal | Per student (p50) |
|---|---|---|
| 5 | 73 tok/s | 114 tok/s |
| 10 | 131 tok/s | 109 tok/s |
| 20 | 199 tok/s | 75 tok/s |
| 30 | 266 tok/s | 58 tok/s |
| 40 | 227 tok/s | 54 tok/s |

## 3. Zou minder videogeheugen ook volstaan?

Dit is de vraag waar de aanvraag om draait, en het antwoord hangt volledig af
van welke piek je maatgevend vindt.

| Kaart | Cachepool | Past bij 15,7 GB (zwaarste sweep) | Past bij 43,4 GB (worst case) |
|---|---|---|---|
| RTX PRO 5000 (72 GB) — € 7.602 | 33,7 GB | ja, 18,0 GB over | **nee** |
| 2× RTX 5090 (64 GB) — ca. € 10.000 | 26,5 GB | ja, 10,8 GB over | **nee**, en ook de 100k-run met 30 studenten (27,3 GB) past niet |
| RTX PRO 6000 (96 GB) — € 37.400 | 55,3 GB | ja, 39,6 GB over | ja, 11,9 GB over |

De worst case is: twintig studenten, maximale context, geen enkele gedeelde
projectbasis. Dat scenario is met opzet extreem — bij een klas die aan dezelfde
opdracht werkt kómt die 0 % niet voor — maar het is wel het scenario dat de
opdracht zelf als grens benoemt. Wie hem meerekent, koopt geen 72 GB. Wie hem
buiten beschouwing laat, heeft aan 72 GB genoeg met achttien gigabyte over.

**Belangrijker dan die keuze:** deze tabel gaat alléén over geheugen, en
volgens conclusie 2 is geheugen niet de beperkende factor. De goedkopere
kaarten hebben minder rekenkracht en minder geheugenbandbreedte, en dat raakt
precies de doorlooptijd die in deze meting de kleur bepaalt. De twee 5090's
hebben er nog een risico bij: geen NVLink, dus het verkeer tussen de kaarten
loopt over PCIe, en bij een MoE-model als dit merk je dat pas onder
gelijktijdige belasting.

**Daarom is de eerlijke conclusie niet "koop de goedkope kaart", maar: deze
meting weerlegt de onderbouwing van € 37.400 op capaciteitsgronden, en het
alternatief is nog niet gemeten.** De RTX PRO 5000 blijkt niet te huren bij
RunPod; twee RTX 5090's — het andere alternatief uit de aanvraag — wel, voor
ongeveer € 12 aan rekentijd. Het harnas draait ongewijzigd tegen elk endpoint.
Met 96 GB en 64 GB gemeten wordt 72 GB bovendien een tussenwaarde in plaats van
een gok, want de KV-cache schaalt lineair met het aantal tokens.

## 4. Welke vLLM-instellingen zijn bepalend?

Eén instelling doet er echt toe, de rest niet.

| Variant | Oordeel | p90 TTFT | KV-piek | Cache hit rate |
|---|---|---|---|---|
| `--kv-cache-dtype auto` | groen | 0,35 s | 15,4 GB (28,1 %) | 99 % |
| `--kv-cache-dtype fp8` | groen | 0,41 s | **6,9 GB (12,6 %)** | 99 % |
| `--max-num-seqs 16` | groen | 0,42 s | 6,9 GB (12,7 %) | 99 % |
| `--max-num-seqs 64` | groen | 0,41 s | 6,9 GB (12,7 %) | 99 % |
| `--max-model-len 65536` | groen | 0,42 s | 6,9 GB (12,7 %) | 99 % |

**FP8-KV-cache halveert het cachegebruik, gratis.** 12,6 % tegen 28,1 % bij
verder identieke instellingen, hetzelfde oordeel, dezelfde hit rate. Het
verschil in TTFT is 60 milliseconden — dat is ruis, geen signaal.

**`--max-num-seqs` en `--max-model-len` doen in dit bereik niets.** Zestien,
tweeëndertig en vierenzestig sequenties geven dezelfde getallen tot in de derde
decimaal. Dat komt doordat de belasting nergens in de buurt van de
geheugengrens kwam; op een kleinere kaart kan dit heel anders liggen. Voor deze
kaart: laat `--max-num-seqs 32` staan en besteed er geen tijd meer aan.

**Aanbevolen combinatie:** `--kv-cache-dtype fp8 --max-num-seqs 32
--max-model-len 131072 --enable-prefix-caching`.

## Wat de gedeelde projectbasis oplevert: geheugen, geen snelheid

| Gedeeld | Context | KV-piek | p90 doorlooptijd | Cache hit rate |
|---|---|---|---|---|
| 0 % | 64k | 9,7 GB | 43,2 s | 99,0 % |
| 50 % | 64k | 6,9 GB | 41,3 s | 98,9 % |
| 90 % | 64k | 5,1 GB | 41,8 s | 99,2 % |

Negentig procent gedeelde basis scheelt bijna de helft van het cachegeheugen
en vrijwel niets aan wachttijd. De reden staat in de laatste kolom: de hit rate
was toch al 99 %, en die komt niet van het delen tússen studenten maar van
hergebruik *binnen* één sessie — elke agentstap stuurt de vorige conversatie
opnieuw mee. Delen koopt dus marge, geen snelheid.

## De benoemde momenten

| Moment | Oordeel | p90 doorlooptijd |
|---|---|---|
| Koude start, 20 studenten | groen | 43 s |
| Koude start, 30 studenten | groen | 53 s |
| Lange sessies, weinig studenten, zeer grote context | groen | 51 s |
| Na de stilte: tien minuten niets, dan hervat iedereen | groen | 83 s |
| Deadline: twintig studenten tegelijk in burst | **oranje** | 123 s |
| Worst case: 20 studenten, maximale context, niets gedeeld | **rood** | 396 s |

Het deadline-moment is het enige realistische scenario dat niet groen is: op
een deadline duurt één instructie ruim twee minuten. Dat is werkbaar maar
merkbaar, en het is precies het moment waarop studenten het minst geduld
hebben. Het is ook een van de drie runs waar het strengere oordeel afwijkt van
het oordeel volgens de oorspronkelijke opdracht — die noemt hem groen, want de
TTFT is 0,35 s.

## Waarom de TTFT-getallen zo mooi zijn (en waar je ze niet op moet vertrouwen)

De gemeten p90-TTFT ligt tussen 0,17 s en 0,67 s, dertig tot honderd keer onder
de eis van twintig seconden. Dat getal is echt, maar het meet iets anders dan
je denkt: in 91 tot 99 % van de gevallen zat de context al in de prefix cache,
dus er viel bijna niets voor te rekenen.

De ene plek waar de norm van twintig seconden wél in gevaar komt, is de
*eerste* aanroep van een sessie met een koude cache. In de warmloopfase van de
worst case — twintig studenten die tegelijk voor het eerst een context van
zeventigduizend tokens openen — was de p90-TTFT van die eerste stap **140
seconden**. Dat is te herleiden: de server haalt ongeveer 20.000 tokens/s aan
prefill, en 20 × 70k tokens is 1,4 miljoen tokens, oftewel zeventig seconden
puur voorrekenen voordat de laatste student iets ziet.

Die warmloopfase telt bewust niet mee in de cijfers, en dat is verdedigbaar
voor een capaciteitsmeting. Maar het betekent wel dat er een gat in de meting
zit: **de koude start is alleen op 12k context beoordeeld** (daar is hij ruim
groen, 0,4 s), en nooit op 64k of 100k. Als het beeld is dat de klas 's ochtends
tegelijk aan een groot project begint, is dat het scenario dat nog gemeten moet
worden.

Wat de student wél voelt is de doorlooptijd van een hele instructie, en dat is
het getal dat in dit hele rapport de kleur bepaalt. Eén instructie is drie tot
vijftien modelaanroepen; een run kan een TTFT van 0,3 s hebben en toch twee
minuten over één opdracht doen. Bij drie van de 38 runs lopen het strenge en
het oorspronkelijke oordeel daardoor uiteen, en steeds in dezelfde richting:
de oorspronkelijke definitie is te optimistisch.

---

## De correcties die hierop gemaakt zijn

De automatisch gegenereerde `RESULTATEN.md` trok op drie punten een conclusie
die uit dezelfde meetgegevens niet volgde. Alle drie zaten in de analysecode,
niet in de meting, en alle drie zijn nu gerepareerd; `RESULTATEN.md` in de
hoofdmap is opnieuw gegenereerd en dekt beide fases.

**1. De alternatievenvergelijking rekende met de verkeerde piek.** Het rapport
zette de kaarten af tegen 15,7 GB — de hoogste piek uit de *sweep*-runs — en
concludeerde dat alle drie de kaarten passen. De analyse kijkt nu naar de
hoogste piek over álle runs (43,4 GB, de worst case), toont beide getallen
naast elkaar, en noemt per kaart wélke runs er niet op passen. De conclusie
draait daarmee om: op geheugen alleen haalt alleen de RTX PRO 6000 de zwaarste
gemeten run.

**2. `kv_auto` werd "beste variant" genoemd op ruis.** De keuze viel op
p90-TTFT: 0,352 s tegen 0,414 s, terwijl het cachegebruik 28,1 % tegen 12,6 %
was. De rangschikking negeert nu TTFT-verschillen kleiner dan vijf procent van
de groen-grens en kiest op cachebezetting; dat wijst `kv_fp8` aan. Het rapport
noemt er ook bij welke varianten binnen de meetruis gelijkwaardig zijn, zodat
niemand een rangorde leest in een gelijkspel.

**3. "De klif ligt bij 40 studenten" was het testplafond.** De analyse
onderscheidt nu of de oplopende run brak of zijn eigen bovengrens raakte, en
zegt in het tweede geval dat er geen klif gevonden is.

Bij het nalopen van de opzet kwamen daar nog drie dingen uit die de meting zelf
raakten:

- **De tokenizer viel stil terug op schatten.** `pod.sh` zette `endpoint.model`
  op de `--served-model-name`, en daar bestaat geen tokenizer onder — dus werden
  de contextgroottes geschat terwijl `tokenizers` gewoon geïnstalleerd was.
  Gerepareerd, en `doctor` wijst nu het echte probleem aan.
- **Herlezen van resultaten wiste de duur van elke run.** `duration_s` werd bij
  opnieuw genereren 0,0 omdat de tijdstempels niet werden teruggelezen. Nu wel;
  de echte duren staan weer in `summary.csv`.
- **De kaartnaam kreeg er per fase een "(600W)" bij** — vandaar
  "... (600W) (600W)" in de lesmap.

Welke kaart de vervolgmeting moet krijgen en wat je daarvoor aanpast, staat in
[OPZET-VERVOLGMETING.md](OPZET-VERVOLGMETING.md).

## Wat deze meting niet zegt

- Niets over de kwaliteit van de gegenereerde code. Gemeten zijn capaciteit en
  latentie.
- Niets over de rekenkracht van de alternatieve kaarten — en dat is volgens
  conclusie 2 juist de bepalende factor.
- Niets over een koude start met grote context. Alleen op 12k gemeten.
- De contextgroottes zijn geschat, niet geteld. Voor een harde uitspraak over
  geheugen: `tokenizers` installeren en opnieuw meten.
- Op één na duurden alle runs vijf minuten meetvenster. De lesvalidatie van
  negentig minuten is de enige langlopende run, en die bleef stabiel — maar het
  is één run.
