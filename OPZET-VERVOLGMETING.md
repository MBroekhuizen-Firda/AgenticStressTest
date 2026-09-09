# De vervolgmeting: welke kaart, en wat je aanpast

De meting op de RTX PRO 6000 laat zien dat de 96 GB nergens nodig was, maar
niet dat een goedkopere kaart volstaat: geheugen bleek niet de beperkende
factor, en over de rekenkracht van de kleinere kaarten zegt die meting niets.
Dit bestand is de checklist voor de meting die dat wél beantwoordt.

## De RTX PRO 5000 is niet te huren

RunPod heeft hem niet in het aanbod. Wat er wél staat, en wat ervan bruikbaar
is voor deze vraag:

| Kaart | Geheugen | Prijs | Bruikbaar? |
|---|---|---|---|
| **2× RTX 5090** | 2 × 32 GB | $ 0,99/uur per kaart | **Ja — dit is het alternatief van € 10.000 uit de aanvraag** |
| RTX PRO 6000 MIG 48 GB | 48 GB | $ 1,09/uur | Ja, als goedkope ondergrens: zelfde architectuur, kleinere pool |
| RTX PRO 4500 (×2) | 2 × 32 GB | $ 0,72/uur per kaart | Ja, ook Blackwell, maar niet de kaart uit de aanvraag |
| L40S / RTX 6000 Ada | 48 GB | $ 1,09 / $ 0,84 | Kan, maar Ada in plaats van Blackwell: dan meet je twee verschillen tegelijk |
| A40, A6000, A100 | 48–80 GB | $ 0,49–1,59 | **Nee.** Ampere heeft geen FP8-KV-cache; je zou het model of `--kv-cache-dtype` moeten veranderen en dan vergelijk je niets meer |
| H100, H200, B200 | 80–288 GB | $ 2,89–7,89 | Niet relevant: duurder en groter dan wat je wilt weten |

**Doe de meting op twee RTX 5090's.** Niet als vervanging van de 72 GB-vraag,
maar omdat het een vraag beantwoordt die al op tafel ligt: de aanvraag noemt
"twee RTX 5090's, circa € 10.000" als alternatief, en die is nooit gemeten. Het
is bovendien dezelfde generatie als de kaart in de aanvraag, dus je meet het
effect van minder geheugen en minder rekenkracht, niet van een andere
architectuur.

### En het antwoord over 72 GB dan?

Dat wordt rekenwerk in plaats van huurwerk, en dat kan hier omdat de meting de
geheugenbehoefte per student geeft. Met 96 GB gemeten en 64 GB erbij liggen er
twee punten op de geheugenas; 72 GB ligt daartussen, en de KV-cache schaalt
lineair met het aantal tokens. Wat je níét kunt interpoleren is rekenkracht —
maar de 5090-meting geeft daar wel de richting van, want ze zijn van dezelfde
generatie.

Zoek je toch een echte 72 GB-meting, dan moet dat bij een andere verhuurder
(Vast.ai, Lambda, DataCrunch en Hyperstack wisselen sterk in aanbod). Het
harnas draait ongewijzigd tegen elk endpoint.

---

## Draaien op twee RTX 5090's

`scripts/pod.sh` kan dit al; het staat in zijn eigen `--help`:

```bash
TENSOR_PARALLEL=2 VRAM_GB=64 scripts/pod.sh all
```

Dat zet `--tensor-parallel-size 2` op de vLLM-start en geeft het harnas 64 GB
als pool door. De rest van de serververlaggen blijft identiek aan de vorige
meting:

```
--max-model-len 131072 --max-num-seqs 32
--gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --enable-prefix-caching
```

`--max-model-len 131072` blijft haalbaar. De KV-cache kost op dit model met FP8
ongeveer 45 kB per token — afgeleid uit de worst-case-run (43,4 GB bij circa
1,0 miljoen tokens aan onderscheiden context), en dat komt overeen met wat de
architectuur voorspelt (48 lagen × 4 KV-heads × 128 × 2). Eén sequentie van
131k tokens is dus zo'n 5,9 GB, en dat past in de 26,5 GB cachepool. De server
start; hij zal alleen eerder gaan preempten, en dat is precies wat je meet.

**Let op drie dingen bij deze opstelling**, alle drie al benoemd in
[README hoofdstuk 7](README.md#7-herhalen-op-andere-hardware):

- Consumentenkaarten hebben geen NVLink; het verkeer tussen de kaarten loopt
  over PCIe. Bij een MoE-model als dit is dat merkbaar, en het effect zie je
  pas onder gelijktijdige belasting — dus precies in deze test.
- De 64 GB is geen pool van 64 GB. Elke kaart draagt een deel van de gewichten,
  en het bruikbare cachegeheugen is minder dan de optelsom.
- Twee kaarten in één machine vragen om moederbord, voeding en behuizing die
  dat aankunnen. Dat hoort in de kostenvergelijking, niet alleen de kaartprijs.

---

## Wat je moet regelen, op welke kaart je ook meet

### 1. De tokenizer — dit ging de vorige keer mis

De meting op de RTX PRO 6000 heeft geschatte contextgroottes, niet getelde
(`"tokenizer_exact": false` in `environment.json`), terwijl `tokenizers` wél
geïnstalleerd was en `prefer_exact` op `true` stond. De oorzaak: `pod.sh` zet
`endpoint.model` op de `--served-model-name` (`qwen3-coder`), en het harnas
zocht de tokenizer onder díé naam. Dat is geen HuggingFace-repo, dus viel het
stilletjes terug op de schatting van 4,65 tekens per token — op een meting die
volledig over contextgrootte gaat.

Dat is nu gerepareerd: `pod.sh` geeft `tokenizer.path` mee, wijzend naar de
snapshot die toch al gedownload is, en `stresstest doctor` zegt voortaan dát er
een verkeerd pad staat in plaats van "installeer `tokenizers`" wanneer het
pakket er al is.

**Controleer na de eerste run** dat `environment.json` `"tokenizer_exact": true`
zegt. Zo niet: `stresstest doctor` draaien, die wijst nu het echte probleem aan.

Dit maakt de contextgroottes een paar procent anders dan in de vorige meting.
Dat is de goede kant op — geteld is beter dan geschat — maar noteer het, want de
"64k-kolom" van de twee metingen is dan niet exact dezelfde belasting. Wil je
een strikte A/B in plaats van een betere meting, draai dan met
`--set tokenizer.prefer_exact=false`; dan blijven beide runs op de schatting.

### 2. Het corpus moet hetzelfde zijn

Het harnas kloont de voorbeeldprojecten met `git clone --depth 1` op HEAD, dus
zonder vaste versie. Verandert er iets bovenstrooms, dan meet je een ander
corpus en verschuift de contextsamenstelling.

Controleer in `environment.json` van de nieuwe run dat het corpus overeenkomt:

| | Meting RTX PRO 6000 |
|---|---|
| `files` | 187 |
| `shared_files` | 75 |
| `private_files` | 112 |
| `total_characters` | 331940 |

Wijkt het af, kopieer dan `corpus_cache/` van de vorige pod mee, of zet
`corpus.local_path` op die map.

### 3. `endpoint.timeout_s` blijft op 300

Niet ophogen. Op een kleinere kaart is een verzoek dat de tijdslimiet raakt een
waarschijnlijke uitkomst, geen storing; het harnas telt hem als fout en dat
kleurt de run rood. Dat ís het antwoord. Wie de limiet ophoogt om de run "af te
laten lopen", meet een klas die niemand accepteert.

### 4. De klifzoeker stapt nu met twee tegelijk

`matrix.rampup.step_students` staat op `2`. Dat halveert de duurste run van de
matrix — van ruim 76 naar 40 minuten — ten koste van de nauwkeurigheid: het
antwoord is dan op twee studenten na goed, en het rapport zegt dat er zelf bij.

Op de RTX PRO 6000 kostte die run 76 minuten om níéts te vinden (hij liep tot
het plafond van 40). Op een kleinere kaart zal hij wél breken, en dan is twee
studenten resolutie ruim genoeg. Wil je de grens exact weten, draai hem daarna
nog eens met `--set matrix.rampup.step_students=1` en
`--set matrix.rampup.start_students=<een paar onder de gevonden grens>`.

### 5. De engine-varianten worden nu wél belangrijk

Op de RTX PRO 6000 gaven `--max-num-seqs 16`, `32` en `64` en
`--max-model-len 65536` tegen `131072` identieke getallen tot in de derde
decimaal, omdat de geheugendruk nergens in de buurt van de grens kwam. Op een
pool van 26,5 GB komt hij daar wel. Sla `--only engine` dus niet over om tijd
te besparen; dit is de opstelling waarop die vlaggen iets doen.

### 6. `hardware.alternatives` blijft staan zoals hij is

Dan vergelijkt het rapport opnieuw dezelfde drie kaarten, nu vanaf de andere
kant van de schaal.

---

## Wat je mag verwachten

De cachepool is 64 × 0,90 − 31,1 = **26,5 GB**, tegen 54,9 GB op de RTX PRO
6000. Dat is 48 % zoveel onderscheiden context.

Afgezet tegen wat elke run op de grote kaart aan cache vroeg:

| Run | Nodig | Past in 26,5 GB? |
|---|---|---|
| Lesvalidatie 90 min (20 studenten, 32k, 50 % gedeeld) | 8,5 GB | ja, ruim |
| Sweep 20 studenten, 100k, 50 % gedeeld | 15,7 GB | ja |
| Sweep 30 studenten, 100k, 50 % gedeeld | 27,3 GB | **nee**, net niet |
| Worst case (20 studenten, 100k, niets gedeeld) | 43,4 GB | **nee** |

Per student gemeten op de grote kaart: 0,35 GB bij 64k met de helft gedeeld,
0,79 GB bij 100k met de helft gedeeld, 2,17 GB bij 100k zonder iets gedeeld.
Deel dat in de 26,5 GB en je krijgt de bovengrens die geheugen alleen stelt:
ongeveer 75 studenten, 33 studenten, en **12** studenten. Die laatste ligt onder
de klas van twintig, dus verwacht bij de worst case preempties, en daarmee rood.

**Dat is geen mislukking van de test, dat is de uitkomst.** De vraag is of dat
scenario binnen bereik hoort te vallen; zie de kanttekening bij vraag 3 in
[RESULTATEN.md](RESULTATEN.md).

Wat deze meting moet uitwijzen en de vorige niet kon: hoe de doorlooptijd van
een instructie zich houdt. Op de RTX PRO 6000 zat de gezamenlijke
decodesnelheid rond 200–300 tokens/s en werd de klif door rekenkracht bepaald,
niet door geheugen. Twee 5090's over PCIe schuiven die klif naar links, en met
hoeveel is precies wat er nog niet gemeten is.

### Optioneel: de ondergrens voor $ 1,09/uur

Wil je er goedkoop een derde punt bij, draai dan dezelfde matrix op de
**RTX PRO 6000 MIG 48 GB**. Zelfde architectuur als de kaart in de aanvraag,
maar een pool van 48 × 0,90 − 31,1 = **12,1 GB**. Daar past de sweep met twintig
studenten op 100k (15,7 GB) al niet meer in. Reken op veel rood — en dat is de
waarde ervan: met 96, 64 en 48 GB gemeten ligt de hele geheugenas vast en is
72 GB een tussenwaarde in plaats van een gok.

Let op: een MIG-partitie krijgt niet alleen minder geheugen maar ook een deel
van de rekenkernen. Het is dus geen zuiver geheugenexperiment, en dat hoort bij
de uitkomst vermeld.

---

## Draaien

Zet eerst een remote klaar die mag pushen — `pod.sh` stopt de pod pas als de
resultaten in de repo staan:

```bash
git remote set-url origin https://<token>@github.com/<eigenaar>/<repo>.git

scripts/pod.sh doctor                              # endpoint, metrics, tokenizer, corpus
TENSOR_PARALLEL=2 VRAM_GB=64 scripts/pod.sh all --deadman auto
```

Aan het eind schrijft het script zelf één rapport over beide fases, commit het
met de meetmappen naar `resultaten/<kaart>-<tijdstempel>`, en zet de pod tien
minuten later uit. Mislukt de push, dan blijft de machine draaien en blijft de
doodsklok staan; dan haal je ze met `scp` op zoals in
[README hoofdstuk 3](README.md#de-resultaten-ophalen).

Het gecombineerde rapport kun je ook zelf maken, zonder GPU:

```bash
python3 -m stresstest report results/<matrixmap> \
  --also results/<lesmap> --out RESULTATEN.md
```

Kosten: de matrix is in tijd begrensd, niet in werk, dus de doorlooptijd blijft
ongeveer gelijk aan de vorige keer — min de 36 minuten die de klifzoeker nu
korter is, dus ruwweg 6 uur 40. Bij $ 1,98/uur voor twee 5090's is dat ongeveer
$ 13, oftewel **€ 12**. Zie [README hoofdstuk 8](README.md#8-kostenraming).
