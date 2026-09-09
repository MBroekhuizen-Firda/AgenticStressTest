# Stresstest agentic coding — hoeveel studenten past er op één GPU?

Dit is een meetharnas. Het beantwoordt één vraag: **hoeveel gelijktijdige
mbo-studenten kan één GPU bedienen die een lokaal codeermodel draait, voordat de
ervaring onacceptabel wordt, en hoeveel videogeheugen is daar werkelijk voor
nodig?**

Er ligt een aanvraag voor € 37.400 voor een server met een NVIDIA RTX PRO 6000
Blackwell (96 GB) die een klas van twintig studenten moet bedienen met
Qwen3-Coder-30B-A3B in FP8 via vLLM. Die aanvraag rust op schattingen. Voor
ongeveer € 20 aan gehuurde rekentijd kun je het meten in plaats van schatten.

Deze README is geschreven voor een collega die dit zelfstandig moet kunnen
uitvoeren, zonder ervaring met GPU-verhuur of vLLM.

---

## Inhoud

1. [Wat deze test beantwoordt en waarom](#1-wat-deze-test-beantwoordt-en-waarom)
2. [Eerst gratis uitproberen](#2-eerst-gratis-uitproberen)
3. [Een GPU huren](#3-een-gpu-huren)
4. [Opzetten: van kale instance naar draaiende vLLM](#4-opzetten-van-kale-instance-naar-draaiende-vllm)
5. [De test draaien](#5-de-test-draaien)
6. [De resultaten lezen](#6-de-resultaten-lezen)
7. [Herhalen op andere hardware](#7-herhalen-op-andere-hardware)
8. [Kostenraming](#8-kostenraming)
9. [Bekende valkuilen](#9-bekende-valkuilen)
10. [Hoe het harnas werkt](#10-hoe-het-harnas-werkt)

---

## 1. Wat deze test beantwoordt en waarom

Agentic coding is iets anders dan een chatbot. Een student typt één instructie
("bouw een inlogformulier") en de assistent doet daarop drie tot vijftien
modelaanroepen vlak achter elkaar: bestand lezen, bewerken, test draaien,
uitvoer lezen, fout herstellen. Daartussen zit vrijwel geen pauze. Daarna is de
student minuten bezig met lezen en nadenken.

Dat heeft twee gevolgen die de gebruikelijke benchmarks niet meten.

Ten eerste: de belasting komt in **golven**. De piek ontstaat niet doordat
iedereen constant bezig is, maar doordat meerdere studenten toevallig tegelijk
in zo'n reeks aanroepen zitten. Een test met twintig bots die onafgebroken
verzoeken afvuren meet iets anders dan een klas.

Ten tweede: elke aanroep binnen zo'n reeks stuurt de **hele voorgaande
context** opnieuw mee — de systeemprompt, de gelezen bestanden, de testuitvoer.
Bij een project van gemiddelde omvang is dat al snel 30.000 tot 60.000 tokens
per student. Die contexten staan in het videogeheugen van de GPU, in de
zogeheten KV-cache. Dáár zit de grens, niet in rekenkracht.

Wat het geheugen redt: studenten in één klas werken aan **dezelfde opdracht**.
Ze delen dus dezelfde systeemprompt en grotendeels hetzelfde projectskelet. Dat
gedeelde begin staat maar één keer in de cache — dat heet prefix caching, en het
is het belangrijkste getal in deze hele test.

Wat het geheugen breekt: als de cache vol raakt, gooit vLLM sessies eruit. Dat
heet een *preemptie*. De volgende agentstap van die student moet dan de complete
context opnieuw doorrekenen, wat tientallen keren zo duur is. Dat is de
faalmodus waar we naar zoeken: niet "het wordt langzamer", maar "het klapt om".

Na afloop kun je zeggen:

1. Past een klas van twintig op 96 GB, en met hoeveel marge?
2. Waar ligt de klif — bij hoeveel studenten, of bij welke contextlengte?
3. Zou 72 GB (RTX PRO 5000, € 7.602) of 64 GB (twee RTX 5090's, circa € 10.000)
   ook volstaan?
4. Welke vLLM-instellingen zijn bepalend, en wat is de goede configuratie?

De test loopt in twee fasen. **Fase 1** is de stresstestmatrix: korte,
gecontroleerde runs die één variabele tegelijk bewegen, zodat je de klif snel en
goedkoop vindt. **Fase 2** is één doorloop van negentig minuten met het
volledige lesritme. Fase 1 bepaalt de capaciteit, fase 2 controleert of die
capaciteit ook een prettig lesuur oplevert. Sla fase 2 niet over: de koude start
waarbij twintig studenten tegelijk beginnen, en wat er met de cache gebeurt
tijdens tien minuten klassikale uitleg, zijn alleen daar zichtbaar.

---

## 2. Eerst gratis uitproberen

Doe dit voordat je een GPU huurt. Het harnas heeft een ingebouwde nep-vLLM die
het protocol en de meetgegevens nabootst, inclusief prefix caching, een
wachtrij en preempties. Daarmee controleer je op je eigen laptop of alles werkt.

```bash
git clone <deze repository>
cd AgenticStressTest

# Python 3.9 of nieuwer (ontwikkeld en getest op 3.11).
# Er zijn geen verplichte pakketten.
python3 --version

# Haal de voorbeeldprojecten binnen die als context dienen (~1 minuut)
python3 -m stresstest corpus -c config/smoke.json

# Start de nep-server in een tweede terminal
python3 -m stresstest mock -- --port 8000

# Controleer of alles bereikbaar is
python3 -m stresstest doctor -c config/smoke.json

# Draai de verkorte matrix (ongeveer 25 minuten)
python3 -m stresstest matrix -c config/smoke.json
```

Wil je in plaats daarvan meteen oefenen met het commando dat je straks op de
gehuurde kaart gebruikt, dan doet dit hetzelfde via de wrapper — inclusief de
nep-server, de controles en de conclusie:

```bash
scripts/pod.sh all --mock --skip-lesson -c config/smoke.json
```

Je vindt de resultaten in `results/<tijdstempel>_matrix/`, met `RESULTATEN.md`
als samenvatting. De getallen zeggen niets over echte hardware — de nep-server
is een simulatie — maar als dit werkt, werkt het straks ook op de gehuurde GPU.

### Installeer `tokenizers`

Dit is de enige aanbeveling in dit document die je echt niet moet overslaan.

```bash
pip install tokenizers    # haalt bij de eerste run tokenizer.json van Hugging Face
pip install matplotlib    # optioneel: PNG-grafieken naast de SVG's
```

Zonder `tokenizers` moet het harnas de contextgrootte schatten op tekens per
token, en de contextgrootte is een van de twee assen van de hele matrix. Een
systematische fout daarin verschuift de conclusie over hoeveel geheugen een klas
nodig heeft — precies de vraag die je stelt.

Hoe groot die fout kan zijn: de veelgehoorde vuistregel van 3,5 tekens per token
komt uit Engels proza. Op broncode klopt hij niet. Gemeten met de tokenizer van
Qwen3-Coder op het corpus van deze test:

| | tekens per token |
|---|---|
| PHP | 4,80 |
| Python | 4,80 |
| JavaScript | 4,69 |
| Markdown | 4,29 |
| Volledige agentberichten, inclusief JSON en opmaak | 4,65 |

Met 3,5 bouwde het harnas een "32k"-context die er in werkelijkheid 17,7k bevatte:
een run van 32.000 tokens die er 17.672 verstuurde, 33 % ernaast. De standaard
staat nu op 4,65, gemeten. Maar die waarde hangt af van het corpus — de
docstring-rijke Python van dit harnas zelf komt op 4,25 uit — dus als je zonder
`tokenizers` moet werken, ijk hem dan eerst tegen je eigen corpus:

```bash
python3 -m stresstest calibrate
```

Dat commando heeft eenmalig wél een tokenizer nodig; daarna kun je de gemeten
waarde in `config/default.json` zetten en draait de rest zonder.

Draai je tegen de mockserver of tegen een klein model, zet dan
`tokenizer.path` op het model dat je straks écht gaat draaien. Dan tel je ook
tijdens het uitproberen met dezelfde liniaal:

```json
"tokenizer": { "path": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8" }
```
 Met
`tokenizers` geïnstalleerd telt het harnas exact en doet de waarde er niet toe.
Zowel `doctor` als elke run zegt welke van de twee is gebruikt, en
`RESULTATEN.md` vermeldt het als er geschat is.

Zonder deze pakketten draait alles door — dat is een bewuste ontwerpkeuze — maar
je meet dan met een liniaal die je zelf niet hebt nagemeten.

---

## 3. Een GPU huren

### Waar

Een RTX PRO 6000 Blackwell per uur huren kan onder meer bij:

| Aanbieder | Bijzonderheden |
|---|---|
| **RunPod** | Eenvoudigste voor wie dit nooit eerder deed. Kies een "GPU Pod", niet "Serverless". Kant-en-klare PyTorch- en vLLM-images. |
| **Vast.ai** | Marktplaats, vaak het goedkoopst. Let op de betrouwbaarheidsscore van de aanbieder en op de uploadsnelheid — je downloadt 31 GB. |
| **Hyperstack, DataCrunch, Lambda** | Reguliere clouds met uurtarieven; iets duurder, iets voorspelbaarder. |
| **Shadeform, Prime Intellect** | Aggregators die over meerdere aanbieders zoeken. Handig als een kaart schaars is. |

Reken op **$ 2,31 tot $ 3,36 per uur**. Beschikbaarheid en prijs van deze kaart
wisselen per week; controleer het actuele aanbod voordat je plant.

Deze kaart is schaars. Kom je er niet aan, dan is dat geen reden om de test af
te blazen: draai de matrix op een goedkopere kaart met bekend geheugen (een
A100 40 GB of een RTX 4090 24 GB) en vergelijk die met de RTX PRO 6000 zodra die
beschikbaar is. Het harnas is er juist op gebouwd om dezelfde test op
verschillende hardware te draaien.

### Wat je kiest

- **GPU:** 1× RTX PRO 6000 Blackwell (96 GB). Niet de "Max-Q"-variant als je kunt
  kiezen; die is trager afgeregeld. De kaartnaam in de winkel zegt dat niet
  altijd — bij RunPod is `RTX PRO 6000` de Server Edition en `RTX PRO 6000 WK`
  de Workstation Edition, allebei op vol vermogen. Wat je werkelijk hebt zie je
  pas op de machine: `nvidia-smi --query-gpu=name,power.default_limit
  --format=csv`. 600 W is de gewone uitvoering, 300 W de Max-Q. `scripts/pod.sh`
  leest dat vanzelf uit, waarschuwt als het naar Max-Q ruikt, en zet het
  wattage in `RESULTATEN.md` — want "op welke kaart is dit gemeten" is de eerste
  vraag die iemand over deze cijfers stelt.
- **Schijf:** minimaal **150 GB**, en let bij RunPod op het onderscheid: de
  *container disk* wordt gewist zodra de pod stopt, het *network volume* op
  `/workspace` niet. Kloon de repository naar `/workspace` en houd het model
  daar, anders ben je bij een stop je resultaten kwijt. Ruim genomen: 80 GB
  container (vLLM en zijn CUDA-wheels) en 100 GB volume (31 GB model plus
  resultaten). Dat kost samen ongeveer twee cent per uur. Het model is 31 GB, maar de Hugging
  Face-cache, pip-pakketten en CUDA-bibliotheken vullen de rest sneller dan je
  denkt. Een volle schijf halverwege de download is zonde van het uurtarief.
- **Image:** een recente PyTorch- of vLLM-image met **CUDA 12.8 of nieuwer**.
  Blackwell heeft compute capability 12.0 (`sm_120`); oudere CUDA-versies en
  oudere vLLM-builds hebben daar geen kernels voor en starten niet.
- **Poort:** stel 8000 open, of gebruik SSH-poortdoorsturing (zie hieronder).

### Hoe je hem weer uitzet — lees dit nu, niet straks

**Een vergeten instance tikt dag en nacht door.** Drie dagen vergeten is
ongeveer € 200 aan niets. Dit is verreweg de duurste fout die je met deze test
kunt maken.

Doe dit meteen bij het aanmaken:

1. Zet een wekker of agenda-afspraak op het moment dat je verwacht klaar te
   zijn, plus een uur.
2. Zoek vóór je begint op waar de knop **Terminate** of **Destroy** staat. Let
   op het verschil met **Stop**/**Pause**: bij de meeste aanbieders blijft je
   opslag doortellen als je alleen stopt, en bij sommige de hele instance.
3. Controleer na afloop in het overzicht dat de instance echt weg is en dat het
   verbruik niet meer oploopt.
4. Stel bij de aanbieder een uitgavenlimiet of waarschuwing in als dat kan.

### De resultaten ophalen

`scripts/pod.sh all` pusht ze zelf naar de repo en stopt de pod pas als dat
gelukt is. Daar zijn twee sleutels voor nodig, en het is makkelijk ze te
verwarren: **GitHub** geeft de pod het recht om te pushen, **RunPod** geeft hem
het recht om zichzelf uit te zetten.

#### 1. Een GitHub-token om te kunnen pushen

Een gehuurde pod heeft geen toegang tot je repo. Maak een *fine-grained*
personal access token — die is per repo af te bakenen, in tegenstelling tot de
klassieke:

1. GitHub → *Settings* → *Developer settings* → *Personal access tokens* →
   *Fine-grained tokens* → **Generate new token**.
2. **Repository access**: *Only select repositories*, en kies alleen deze repo.
3. **Permissions** → *Repository permissions* → **Contents: Read and write**.
   Meer is niet nodig; laat de rest op *No access*.
4. **Expiration**: kort. De meting duurt een dag, dus zet hem op zeven dagen.

Zet hem op de pod als remote, zonder dat hij in je shell-geschiedenis komt:

```bash
read -rsp 'GitHub-token: ' GH_TOKEN && echo
git remote set-url origin \
  "https://x-access-token:$GH_TOKEN@github.com/<eigenaar>/<repo>.git"
```

Twee dingen om te weten. De token staat daarna in platte tekst in
`.git/config` op de pod; dat is te overzien op een machine die je binnen een
dag termineert, maar het is wel een reden om hem daarna in te trekken in plaats
van te bewaren. En een token met schrijfrechten op één repo is het minimum dat
werkt — geef hem geen organisatiebrede rechten omdat het sneller klikt.

Liever geen token in een URL? Een **deploy key** doet hetzelfde met SSH: maak op
de pod een sleutel (`ssh-keygen -t ed25519`), plak de publieke helft onder
*Settings* → *Deploy keys* van de repo met *Allow write access* aan, en gebruik
de `git@github.com:` remote. Die sleutel geldt per definitie voor één repo.

Controleer voor je begint of het werkt — een mislukte push merk je liever nu
dan na zeven uur meten:

```bash
git ls-remote origin >/dev/null && echo "push-toegang in orde"
```

#### 2. Een RunPod-API-sleutel om de pod te stoppen

`pod.sh` stopt de pod met `runpodctl stop pod $RUNPOD_POD_ID`. Lukt dat niet,
dan valt hij terug op `poweroff` — en dat stopt de *container*, niet
noodzakelijk de pod: de opslag tikt dan door. Voor de doodsklok en het
automatisch afsluiten wil je dus dat `runpodctl` werkt.

1. RunPod → *Settings* → *API Keys* → **Create API Key**, met schrijfrechten op
   pods.
2. Op de pod:

```bash
read -rsp 'RunPod API key: ' RUNPOD_KEY && echo
runpodctl config --apiKey "$RUNPOD_KEY"
```

`RUNPOD_POD_ID` zet RunPod zelf al in de omgeving van de pod; controleer met
`echo $RUNPOD_POD_ID`. Is die leeg, dan kan `pod.sh` de pod niet bij naam
stoppen en moet je het zelf doen in het dashboard.

Sla beide sleutels op als *Secret* in je RunPod-template als je vaker meet; dan
staan ze bij de start al in de omgeving en hoef je ze niet per pod te plakken.
Trek ze in zodra de meetreeks klaar is.

#### Waar de resultaten belanden

De resultaten belanden op een eigen branch, `resultaten/<kaart>-<tijdstempel>`,
of op de branch die je met `--branch` meegeeft. `results/` staat in
`.gitignore` — dat is voor lokale proefdraaien; deze meting wordt bewust met
`git add -f` toegevoegd.

Mislukt de push, dan blijft de pod draaien en blijft de doodsklok staan: de
meting bestaat dan nog maar op één plek en de machine mag niet zomaar
verdwijnen. Wil je helemaal niet pushen, gebruik dan `--no-push` — en haal ze
dan zelf op, want dan stopt de pod ook niet uit zichzelf.

Handmatig ophalen kan altijd nog. Doe dat vóórdat je afsluit. De resultatenmap is klein (enkele megabytes); de
ruwe verzoekregels zijn het waardevolst, want daarmee kun je later andere
vragen beantwoorden zonder opnieuw te huren. `scripts/pod.sh all` pakt aan het
eind alles in als `/workspace/stresstest-results-<datum>.tar.gz`; dat ene
bestand kopiëren is sneller dan de map recursief.

```bash
# Op je eigen machine:
scp -P <poort> -i <private-sleutel> \
    root@<ip>:/workspace/stresstest-results-*.tar.gz .

# of de hele map:
scp -P <poort> -i <private-sleutel> -r \
    root@<ip>:/workspace/AgenticStressTest/results ./results-van-de-gpu
```

Het poortnummer staat bij RunPod in het dashboard onder *Connect*, bij **SSH
over exposed TCP**: een hoog nummer, nooit 22, en het verandert bij elke nieuwe
pod. De SSH-regel daarboven loopt via de proxy en draagt geen scp of sftp.
Vraagt hij om een wachtwoord, dan mist de pod je sleutel — RunPod installeert
die alleen bij het opstarten. Beide valkuilen staan uitgewerkt in
[VALKUILEN.md](VALKUILEN.md#de-resultaten-van-de-pod-halen), met twee uitwegen
die geen scp nodig hebben.

Controleer dat de kopie compleet is voordat je termineert; hetzelfde aantal
bestanden aan beide kanten:

```bash
ssh -p <poort> -i <private-sleutel> root@<ip> \
    "find /workspace/AgenticStressTest/results -type f | wc -l"
find ./results-van-de-gpu -type f | wc -l
```

---

## 4. Opzetten: van kale instance naar draaiende vLLM

De hele opzet duurt ongeveer 45 minuten, waarvan het grootste deel de download
van het model is.

### 4.1 Verbinden en controleren

```bash
ssh root@<ip>            # of het commando dat de aanbieder toont; bij RunPod
                         # hoort daar -p <poort> en -i <private-sleutel> bij

nvidia-smi
```

`nvidia-smi` moet de kaart tonen met 97.887 MiB geheugen en een driverversie van
**570 of hoger**. Zie je de kaart niet, of is de driver ouder, kies dan een
andere image; drivers zelf installeren op een gehuurde instance is zelden de
moeite waard.

### 4.2 vLLM installeren

Als de image vLLM al bevat (`vllm --version` werkt), sla deze stap over.

```bash
python3 -m venv /opt/vllm-env
source /opt/vllm-env/bin/activate
pip install --upgrade pip

# Een recente vLLM: eerdere versies hebben geen Blackwell-kernels.
pip install "vllm>=0.10.0"

vllm --version
```

### 4.3 Het model binnenhalen

Ongeveer **31 GB**. Op een snelle verbinding tien minuten, op een trage een uur.
Doe dit apart en niet impliciet bij het starten van de server, zodat je een
mislukte download niet met een mislukte serverstart verwart.

```bash
pip install "huggingface_hub[cli]"

# Een Hugging Face-token is voor dit model niet nodig, maar wel als je
# later een model met toegangsvoorwaarden gebruikt:
# huggingface-cli login

hf download Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8

# Bij een oudere huggingface_hub heet het commando nog:
# huggingface-cli download Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
```

Controleer daarna dat de bestanden er echt staan:

```bash
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8
```

Zit je op ongeveer 31 GB, dan is het goed. Zit je op 2 GB, dan zijn alleen de
metadata gedownload en is de download afgebroken.

### 4.4 vLLM starten

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

De eerste start duurt enkele minuten (het model wordt in het geheugen geladen en
CUDA-graphs worden opgebouwd). De server is klaar als er
`Application startup complete` staat.

> **Waarom hier 131072 en niet 65536?** De matrix bevat een kolom van 100k
> tokens en een scenario van 110k, en een verzoek dat boven `--max-model-len`
> uitkomt wordt door vLLM geweigerd. Voor de *meting* moet het venster dus ruim
> staan. Voor de uiteindelijke *opstelling in de les* wil je hem juist zo klein
> mogelijk hebben — dat is precies wat de variant `engine_len_65k` uitzoekt.
> `python3 -m stresstest doctor` vergelijkt het ingestelde venster met de
> zwaarste run in je matrix en waarschuwt als het te klein staat.

**De vlaggen die ertoe doen, en waarom:**

| Vlag | Waarom |
|---|---|
| `--max-model-len` | Het contextvenster. Tijdens de meting ruim (131072), omdat de matrix tot 110k gaat. Daarna **bewust beperken**: een groter venster kost cachegeheugen en verlaagt het aantal sessies dat tegelijk past, dus je betaalt in aantal studenten voor contextlengte die niemand gebruikt. 64k is ruim voor een mbo-project. Zet het niet op het maximum "voor de zekerheid". |
| `--max-num-seqs 32` | Hoeveel sequenties vLLM tegelijk in behandeling neemt. De standaard staat hoog (256 of 1024). Te hoog betekent dat vLLM meer sessies toelaat dan er geheugen is, waarna het gaat preempten — precies de faalmodus die we meten. Te laag betekent een lange wachtrij. Dit is een van de dingen die de test uitzoekt. |
| `--gpu-memory-utilization 0.90` | Welk deel van het videogeheugen vLLM mag gebruiken. Hoger geeft meer cache maar minder speling; boven 0.95 loop je tegen out-of-memory aan zodra er iets anders op de kaart draait. Gebruik dezelfde waarde in `config/default.json` onder `hardware`, anders klopt de omrekening naar gigabytes niet. |
| `--kv-cache-dtype fp8` | Slaat de cache op in 8 bits in plaats van 16. Ruwweg een halvering van het geheugen per token, dus ongeveer een verdubbeling van het aantal sessies dat past. De matrix meet of dat ten koste gaat van iets. |
| `--enable-prefix-caching` | Hergebruik van het gedeelde begin van contexten. In vLLM V1 staat dit standaard aan; expliciet meegeven maakt duidelijk dat het bewust zo is. **Zonder dit is de hele meting zinloos.** |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Laat het model echte tool calls produceren, zoals een agent doet. Werkt de parser niet, laat beide vlaggen dan weg: het harnas schakelt zelf terug naar een tekstvariant met dezelfde berichtstructuur. |

Controleer in een tweede terminal:

```bash
curl -s localhost:8000/v1/models
curl -s localhost:8000/metrics | grep -c '^vllm:'
```

Het tweede commando moet een getal boven de twintig geven. Geeft het 0, dan
staat de Prometheus-endpoint uit en kun je preempties, cache hit rate en
KV-bezetting niet meten — dan meet je het verkeerde.

### 4.5 Het harnas draaien: op de instance of ernaast?

**Op de instance zelf** is het eenvoudigst en meet je zuiver de server, zonder
netwerkvertraging ertussen. Aanbevolen.

```bash
git clone <deze repository> /workspace/AgenticStressTest
cd /workspace/AgenticStressTest
python3 -m stresstest corpus
```

**Vanaf je eigen machine** kan ook, via een SSH-tunnel. Alle gemeten tijden
bevatten dan wel de reistijd naar de instance:

```bash
ssh -N -L 8000:localhost:8000 root@<ip>
# en in config/default.json blijft base_url http://127.0.0.1:8000/v1
```

### 4.6 Alles in één commando

Hoofdstuk 4 en 5 met de hand doorlopen kan, en als je wilt begrijpen wat er
gebeurt is dat de beste manier. Wil je het gewoon laten draaien, dan doet
`scripts/pod.sh` alles: installeren, het model ophalen, vLLM starten, fase 1,
de engine-varianten mét de bijbehorende herstarts, fase 2, en de conclusie.

```bash
git clone <deze repository> /workspace/AgenticStressTest
cd /workspace/AgenticStressTest

tmux new -s test                      # zodat een wegvallende SSH-verbinding niets breekt
scripts/pod.sh all --deadman auto
```

Dat is de hele test, ongeveer zeven uur. Meekijken kan later met:

```bash
scripts/pod.sh log        # de laatste regels van het nieuwste logbestand
scripts/pod.sh log -f     # meelopen; ctrl-C stopt het kijken, niet de test
scripts/pod.sh status     # draait vLLM, en staat de doodsklok aan
```

Gebruik `scripts/pod.sh log` en niet `tail -20 .../pod-*.log`: zodra een tweede
run een tweede logbestand achterlaat, weigert `tail` die verkorte vorm
(*option used in invalid context*).

**Wat het je uit handen neemt, en waarom dat de moeite is:**

| | |
|---|---|
| **De engine-varianten** | Die vereisen elk een herstart van vLLM met andere vlaggen. Het script herstart de server zelf en draait daarna precies die ene run. Met de hand is dit het stuk waar `--no-pause` verleidelijk is, en dat maakt die vijf runs betekenisloos. |
| **Server en harnas gelijk houden** | `hardware.vram_gb`, `gpu_memory_utilization`, `model_weights_gb` en de modelnaam worden afgeleid uit `nvidia-smi` en uit het model op schijf, en meegegeven aan elke aanroep. Staat de omrekening naar gigabytes scheef, dan is het antwoord op vraag 3 scheef. |
| **Hervatten** | Runs die al op schijf staan worden overgeslagen. Valt je verbinding weg of loopt de pod vast, dan draai je hetzelfde commando opnieuw en gaat het verder waar het was. |
| **De controle vooraf** | Schijfruimte, driverversie, `tokenizers`, en of `/metrics` de drie reeksen levert waar de hoofdvraag op hangt: preempties, prefix-cache en KV-bezetting. Ontbreekt er een, dan stopt het script in plaats van zes uur het verkeerde te meten. |
| **De tool-call-parser** | Start vLLM niet op met `--tool-call-parser qwen3_coder`, dan probeert het script het nog één keer zonder, zoals hoofdstuk 4.4 beschrijft. |
| **De vergeten instance** | `--deadman auto` zet een wekker op de geschatte duur plus anderhalf uur; daarna wordt de pod *gestopt* — niet getermineerd, dus `/workspace` en je resultaten blijven staan. Afzetten met `scripts/pod.sh disarm`. |

**De losse commando's**, als je het toch stap voor stap wilt:

```bash
scripts/pod.sh setup                  # installeren, model ophalen, corpus ophalen
scripts/pod.sh serve                  # vLLM starten op de basisinstelling
scripts/pod.sh doctor                 # de controle uit hoofdstuk 5.2
scripts/pod.sh plan --price-per-hour 3.36
scripts/pod.sh group rampup           # één groep: rampup, sweep, scenarios, shared, activity, engine
scripts/pod.sh lesson                 # alleen fase 2
scripts/pod.sh report results/<map>   # grafieken en conclusie opnieuw, zonder GPU
scripts/pod.sh stop                   # de lopende run en vLLM stoppen
```

Er kan er maar één tegelijk draaien. Start je per ongeluk een tweede keer —
na een verbroken SSH-sessie weet je vaak niet meer of de vorige nog loopt —
dan weigert het script dat en noemt het de pid van de lopende run: twee runs
delen anders dezelfde GPU, dezelfde poort en dezelfde resultaatmap, en aan de
getallen achteraf is niet te zien dat het gebeurd is. `scripts/pod.sh status`
zegt of er iets loopt, `scripts/pod.sh stop` neemt over. Het slot hangt aan
het proces, dus een afgebroken run of een gestopte pod laat niets achter dat
je eerst moet opruimen.

**Andere hardware** gaat via omgevingsvariabelen; de rest blijft gelijk, zoals
[hoofdstuk 7](#7-herhalen-op-andere-hardware) vraagt:

```bash
VRAM_GB=72 scripts/pod.sh all                          # RTX PRO 5000
TENSOR_PARALLEL=2 VRAM_GB=64 scripts/pod.sh all        # twee RTX 5090's

MODEL=Qwen/Qwen2.5-Coder-7B-Instruct MAX_MODEL_LEN=32768 \
  VRAM_GB=24 scripts/pod.sh all --skip-engine          # goedkoop uitproberen
```

**Eerst droog oefenen, op je eigen laptop, zonder GPU en zonder kosten:**

```bash
scripts/pod.sh all --mock --skip-lesson -c config/smoke.json
```

Dat draait dezelfde wrapper tegen de ingebouwde nep-vLLM uit
[hoofdstuk 2](#2-eerst-gratis-uitproberen). De getallen zeggen niets over echte
hardware, maar je ziet precies wat er straks op de gehuurde kaart gebeurt.

**Twee dingen doet het bewust niet.**

Het kiest niet zelf de beste configuratie voor fase 2. Fase 1 levert een oordeel
op waar een mens naar moet kijken; standaard draait de lesvalidatie op de
basisinstelling. Komt er uit fase 1 iets anders, draai fase 2 dan over:

```bash
scripts/pod.sh lesson --flags "--kv-cache-dtype fp8 --max-num-seqs 32 --max-model-len 65536"
```

En het termineert je instance niet. `--deadman` en `--shutdown` *stoppen* de pod:
het GPU-tarief loopt dan niet meer, je resultaten blijven staan. Het echte
opruimen doe je zelf, nadat je de resultaten hebt opgehaald — zie
[hoofdstuk 3](#hoe-je-hem-weer-uitzet--lees-dit-nu-niet-straks).

---

## 5. De test draaien

Dit hoofdstuk beschrijft de stappen los. Draai je `scripts/pod.sh all`, dan zijn
ze allemaal al gedaan — zie [hoofdstuk 4.6](#46-alles-in-een-commando).

### 5.1 Configuratie klaarzetten

Open `config/default.json` en controleer:

- `endpoint.model` — de naam waaronder vLLM het model aanbiedt. Bij het
  commando hierboven is dat `qwen3-coder`.
- `hardware.vram_gb`, `hardware.gpu_memory_utilization` en
  `hardware.model_weights_gb` — nodig om de gemeten cachebezetting om te rekenen
  naar gigabytes, en dus om vraag 3 te kunnen beantwoorden.

Aanpassen kan ook zonder het bestand te bewerken:

```bash
python3 -m stresstest matrix --set endpoint.model=qwen3-coder \
                             --set hardware.vram_gb=96
```

### 5.2 Eerst controleren

```bash
python3 -m stresstest doctor
```

Dit controleert het endpoint, doet één testverzoek, vergelijkt het
contextvenster van de server met de zwaarste run in je matrix, en zegt of
`/metrics` bruikbaar is. Los alles op wat hier misgaat voordat je de matrix start; een
uur meten met een kapotte metrics-endpoint is een verloren uur. Zegt `doctor`
dat de contextgroottes een schatting zijn, installeer dan `tokenizers` — zie
[hoofdstuk 2](#installeer-tokenizers).

### 5.3 Het plan bekijken

```bash
python3 -m stresstest plan --price-per-hour 3.36
```

Dit toont alle runs, hoe lang ze duren en wat de huur ongeveer kost. Fase 1 is
38 runs en ongeveer **4 uur 45**, waarvan de klifzoeker het grootste blok is
(die stopt zodra de drempels breken, meestal ruim eerder dan het maximum).

De klifzoeker stapt standaard met twee studenten tegelijk
(`matrix.rampup.step_students`). Dat halveert de duurste run van de matrix en
maakt het antwoord op twee studenten na nauwkeurig; zet hem op 1 als je de
grens precies wilt weten, en begin dan met `start_students` vlak onder de
verwachte grens.

### 5.4 Fase 1: de matrix

```bash
python3 -m stresstest matrix
```

Wat je onderweg ziet:

```
[10:14:22] [7/38] sweep_s20_c32k -- 20 studenten, 32k context (6m30s, nog 3u21m te gaan)
[10:14:37]   sweep_s20_c32k | 15s/6m30s | fase=warmup | verzoeken=41 | actief=12 | p90 TTFT(60s)=  3.2s | KV=38% | wachtrij=0 | preempties=0
...
[10:20:57]     GROEN  | p90 TTFT 6.1s | p90 instructie 48s | 612 verzoeken | 71 instructies | preempties 0 | cache 74% | KV-piek 61%
```

Elke run wordt meteen naar schijf geschreven, dus een afgebroken doorloop is
geen verloren doorloop. Aan het eind verschijnt de gekleurde matrix in de
terminal en worden de grafieken en `RESULTATEN.md` gemaakt.

**De vijf runs met vLLM-varianten vereisen een herstart van de server.** Het
harnas toont de vlaggen en wacht op enter. Draai je onbewaakt, gebruik dan
`--no-pause`; die runs worden dan met de huidige serverinstelling gedraaid en
zeggen dus niets. Beter is ze apart te doen:

```bash
# herstart vLLM met de getoonde vlaggen, dan:
python3 -m stresstest matrix --run engine_kv_fp8 --out results/engine
```

### 5.5 Fase 2: de lesvalidatie

Op de configuratie die uit fase 1 als beste komt:

```bash
python3 -m stresstest lesson
```

Eén doorloop van negentig minuten met het volledige lesritme: opstartpiek,
opbouw, tien minuten klassikale uitleg waarin niemand iets vraagt, eindpiek,
afbouw. Wil je eerst controleren of het werkt, zet dan `lesson.time_scale` op
`0.1` voor een droogloop van negen minuten.

### 5.6 Als de tijd beperkt is

De onderdelen los, in volgorde van nut per bestede euro:

```bash
python3 -m stresstest matrix --only rampup       # ~30 min, geeft direct het maximum
python3 -m stresstest matrix --only sweep        # ~1u55, de gekleurde matrix
python3 -m stresstest matrix --only scenarios    # ~50 min, wat er misgaat
python3 -m stresstest matrix --only shared       # ~40 min, wat de gedeelde basis oplevert
python3 -m stresstest matrix --only engine       # ~35 min, plus herstarts
```

Heb je maar één uur: draai `--only rampup` en daarna de koude start
(`--run scen_koudestart_s20`). Dat zijn samen de twee getallen waar het meeste
op hangt.

Losse runs met eigen parameters:

```bash
python3 -m stresstest run --students 24 --context 48000 --shared 0.7
```

Grafieken en conclusie opnieuw maken van bestaande metingen, zonder GPU:

```bash
python3 -m stresstest report results/20260420-101422_matrix
```

Fase 1 en fase 2 landen in aparte mappen, dus geen van beide bevat het hele
antwoord. Eén rapport over allebei, zonder de meetmappen zelf aan te raken:

```bash
python3 -m stresstest report results/20260420-101422_matrix \
  --also results/20260420-183012_les --out RESULTATEN.md
```

De kaart moet in beide mappen dezelfde zijn; anders weigert het commando, want
een rapport dat stilletjes twee kaarten mengt is slechter dan twee rapporten
die er ieder de helft van dekken.

---

## 6. De resultaten lezen

Een resultatenmap ziet er zo uit:

```
results/20260420-101422_matrix/
├── RESULTATEN.md          de conclusie in gewone taal — begin hier
├── analyse.json           dezelfde conclusie als gegevens
├── summary.csv            één regel per run, alle kerngetallen
├── config.json            de configuratie waarmee dit gemeten is
├── environment.json       model, tokenizer, corpus, hardware
├── charts/                de grafieken
└── runs/<run_id>/
    ├── run.json           alles over deze ene run
    ├── requests.csv       elk afzonderlijk verzoek
    ├── bursts.csv         elke student-instructie als geheel
    └── server_metrics.csv de Prometheus-meetreeks over de tijd
```

### Welke grafiek beantwoordt welke vraag

| Vraag | Grafiek |
|---|---|
| 1. Past een klas van twintig? | `01_matrix.svg` — zoek de rij "20 studenten" |
| 2. Waar ligt de klif? | `06_klifzoeker.svg`, en `02_ttft_vs_studenten.svg` voor de knik per contextgrootte |
| 3. Zou minder geheugen volstaan? | de tabel in `RESULTATEN.md` bij vraag 3, gevoed door de KV-piek uit `summary.csv` |
| 4. Welke instellingen? | de tabel bij vraag 4, plus `05_preempties_per_scenario.svg` |

`03_doorlooptijd_instructie.svg` is de grafiek die het dichtst bij de beleving
van de student staat, en `04_cache_en_kv_over_tijd.svg` laat zien wat er tijdens
de klassikale uitleg met de cache gebeurt.

### Wat is een goed getal

| Meting | Goed | Alarm |
|---|---|---|
| **p90 time to first token** | onder 20 s | boven 45 s, of een steile knik tussen twee studentaantallen |
| **p90 doorlooptijd instructie** | onder 90 s | boven 180 s — de student is dan langer aan het wachten dan aan het werk |
| **Decodesnelheid per stream** | boven 12 tokens/s | onder 6 tokens/s leest niemand mee |
| **Prefix cache hit rate** | boven 60 % bij een gedeelde opdracht | onder 30 % waar je een gedeelde basis verwachtte: het gedeelde deel wordt niet hergebruikt |
| **Preempties** | **nul** | elk getal boven nul is een waarschuwing; aanhoudende preempties zijn een harde afwijzing |
| **KV-cachebezetting (piek)** | onder 80 % | boven 90 % is er geen marge meer voor een uitschieter |
| **Wachtrijdiepte** | 0 | structureel boven 0 betekent dat `--max-num-seqs` of het geheugen knelt |

**Het belangrijkste alarmsignaal is een preemptie.** Eén preemptie betekent dat
een sessie uit de cache is gegooid en dat de volgende agentstap van die student
de hele context opnieuw moet doorrekenen. Dat is niet "iets trager"; dat is
tientallen keren zo duur, precies op het moment dat de student verder wil.

Let ook op het verschil tussen **oranje door TTFT** en **oranje door
doorlooptijd**. Het eerste is een wachtrij: er is te weinig doorvoer. Het tweede
is een langzame decode: het model typt te traag. Die twee vragen om een andere
oplossing.

### De kleuren

- **Groen** — p90 TTFT onder 20 s, p90 doorlooptijd van een instructie onder
  90 s, decodesnelheid boven 12 tokens/s, geen preempties, geen afgelopen
  verzoeken.
- **Oranje** — p90 TTFT tot 45 s, of doorlooptijd tot 180 s, of incidentele
  preempties.
- **Rood** — daarboven, of aanhoudende preempties, of verzoeken die aflopen.

De grens van 20 seconden komt uit de functionele eisen van Firda. Twee dingen
zijn bewust toegevoegd of weggelaten, en `RESULTATEN.md` legt dat ook uit:

1. **De doorlooptijd van een hele instructie telt mee.** Time to first token is
   de eerste van drie tot vijftien modelaanroepen. Een run kan een prima TTFT
   hebben en toch twee minuten over één instructie doen.
2. **De prefix cache hit rate is gerapporteerd maar geen groen-eis.** Die hangt
   af van het scenario, niet van de hardware: in de runs met 0 % gedeelde basis
   en in de koude start is 60 % per definitie onhaalbaar. Een lage hit rate waar
   we een hoge verwachtten komt als waarschuwing bij de run te staan.

Elke run krijgt daarnaast een tweede oordeel volgens de oorspronkelijke
definitie (kolom `grade_brief` in `summary.csv`), zodat je kunt zien waar de
twee uiteenlopen. De drempels staan in `config/default.json` onder
`grading.thresholds` en zijn aan te passen.

---

## 7. Herhalen op andere hardware

Het punt van deze test is niet bevestigen dat de duurste kaart werkt. Het is
uitzoeken wat genoeg is. Draai daarom dezelfde matrix op elke kaart die je
serieus overweegt, en verander alleen wat je moet veranderen.

**Wat hetzelfde blijft:** de seed, het gedragsmodel, het corpus, de
contextgroottes, de studentaantallen, de drempels. Anders vergelijk je twee
verschillende experimenten.

**Wat je aanpast:**

### RTX PRO 5000 (72 GB)

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder --host 0.0.0.0 --port 8000 \
  --max-model-len 131072 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --enable-prefix-caching
```

Alleen in het harnas: `hardware.vram_gb` op `72.0` en `hardware.gpu_name`
bijwerken. De serververlaggen blijven gelijk.

### Twee RTX 5090's (2× 32 GB = 64 GB)

Hier verandert wél iets aan de server: het model moet over twee kaarten worden
verdeeld.

```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
  --served-model-name qwen3-coder --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 131072 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 --kv-cache-dtype fp8 --enable-prefix-caching
```

En in het harnas `hardware.vram_gb` op `64.0`.

Let bij deze opstelling op drie dingen:

- **De verbinding tussen de kaarten.** Consumentenkaarten hebben geen NVLink;
  het verkeer loopt over PCIe. Bij een MoE-model als Qwen3-Coder-30B-A3B is dat
  merkbaar, en het effect zie je pas onder gelijktijdige belasting — dus precies
  in deze test.
- **Elke kaart draagt een kopie van een deel van de gewichten.** De 64 GB is
  niet één pool van 64 GB. Het bruikbare cachegeheugen is minder dan de simpele
  optelsom.
- **Twee kaarten in één machine** vragen om een moederbord, voeding en behuizing
  die dat aankunnen. Dat hoort in de kostenvergelijking thuis, niet alleen de
  kaartprijs.

### Kleiner model uitproberen

Het harnas praat tegen elk OpenAI-compatibel endpoint. Wil je eerst goedkoop op
bijvoorbeeld een RTX 4090 kijken:

```bash
vllm serve Qwen/Qwen2.5-Coder-7B-Instruct --served-model-name coder \
  --max-model-len 32768 --max-num-seqs 16 --enable-prefix-caching

python3 -m stresstest matrix --set endpoint.model=coder \
  --set matrix.sweep.context_tokens='[8000, 16000, 32000]' \
  --set hardware.vram_gb=24 --set hardware.model_weights_gb=15
```

De uitkomsten zijn niet overdraagbaar naar het grote model, maar je ziet wel of
de hele keten werkt.

---

## 8. Kostenraming

Bij $ 2,31 tot $ 3,36 per uur voor een RTX PRO 6000 Blackwell:

| Onderdeel | Tijd | Kosten |
|---|---|---|
| Opzetten: instance, vLLM, modeldownload | ~45 min | $ 1,75 – $ 2,50 |
| Fase 1 — systematische sweep (16 runs) | ~1 u 55 | $ 4,40 – $ 6,45 |
| Fase 1 — activiteitsniveau (4 runs) | ~28 min | $ 1,10 – $ 1,60 |
| Fase 1 — gedeelde projectbasis (6 runs) | ~42 min | $ 1,60 – $ 2,35 |
| Fase 1 — benoemde scenario's (6 runs) | ~50 min | $ 1,90 – $ 2,80 |
| Fase 1 — vLLM-varianten (5 runs + herstarts) | ~45 min | $ 1,75 – $ 2,50 |
| Fase 1 — klifzoeker (1 run, stappen van 2) | 20 – 40 min | $ 0,75 – $ 2,25 |
| **Fase 1 totaal** | **~4 u 45** | **$ 11 – $ 16** |
| Fase 2 — lesvalidatie van 90 minuten | ~1 u 40 | $ 3,85 – $ 5,60 |
| **Alles bij elkaar** | **~6 u 25** | **$ 15 – $ 22** |

Ongeveer € 14 tot € 20. Ruim een tiende procent van de aanvraag.

Wil je het op twee of drie kaarten doen om ze te kunnen vergelijken,
vermenigvuldig dan met twee of drie: nog steeds onder de honderd euro.

Twee dingen kunnen dit bedrag laten oplopen:

- **Een vergeten instance.** Zie [hierboven](#hoe-je-hem-weer-uitzet--lees-dit-nu-niet-straks).
- **Een mislukte doorloop.** Draai daarom eerst `doctor` en daarna
  `--only rampup`; als dat goed gaat, gaat de rest ook goed.

---

## 9. Bekende valkuilen

Dit hoofdstuk gaat over de meting: keuzes die je een verkeerd antwoord
opleveren. Voor wat er tijdens het opzetten stukgaat — vLLM die niet start,
een gewiste container disk, een `tail` die weigert — staat
[VALKUILEN.md](VALKUILEN.md) klaar, met per melding de oorzaak en wat je doet.

**`--max-num-seqs` staat standaard hoog.** vLLM kiest 256 of hoger. Bij sommige
modellen mislukt het opstarten daardoor ("no available memory for the cache
blocks"), omdat vLLM ruimte wil reserveren voor meer sequenties dan er passen.
En als het wél start, laat het meer sessies toe dan er geheugen is, waarna het
gaat preempten. Zet hem bewust: 32 is een verstandig beginpunt voor een klas van
twintig. Wat de goede waarde is, zoekt de test uit.

**`--max-model-len` bewust beperken.** De verleiding is om het maximum van het
model te nemen "voor de zekerheid". Doe dat niet: een langer contextvenster kost
cachegeheugen en verlaagt de parallelliteit, dus je betaalt in aantal studenten
voor een contextlengte die niemand gebruikt. Meet met de matrix hoeveel context
je klas werkelijk nodig heeft en zet het daar net boven.

**Blackwell heeft recente software nodig.** CUDA 12.8 of nieuwer en een recente
vLLM. Oudere builds hebben geen kernels voor `sm_120` en falen bij het starten,
soms met een onduidelijke foutmelding over compute capability.

**Prefix caching moet aanstaan.** Staat het uit, dan meet je een wereld waarin
studenten niets delen, en die bestaat niet. Controleer dat
`vllm:prefix_cache_queries_total` in `/metrics` oploopt tijdens een run.

**Meet niet zonder `/metrics`.** Zonder de Prometheus-endpoint heb je geen
preempties, geen cache hit rate en geen KV-bezetting — dan meet je alleen
latentie, en dat is de helft van de vraag. `doctor` waarschuwt hiervoor.

**De eerste run na het starten van de server is niet representatief.** CUDA
graphs, de eerste compilatie en een lege cache maken hem traag. De warmloopfase
in elk run vangt dat op, maar draai bij twijfel eerst een korte
`run --measure 60` en gooi die weg.

**Reproduceerbaarheid vraagt om dezelfde seed én dezelfde cachetoestand.** Het
harnas draait deterministisch, maar de prefix cache van vLLM bewaart wat de
vorige run erin heeft gezet. Wil je twee configuraties eerlijk vergelijken,
herstart vLLM ertussen.

**Het HTTP-verkeer is één verbinding per verzoek.** Dat is bewust: geen
verborgen toestand, geen pooling-fouten. De verbindingsopbouw wordt apart
gemeten (`connect_s` in `requests.csv`) en zit niet in de TTFT. Meet je over een
SSH-tunnel, dan komt die tijd er wel bovenop.

**Schatten in plaats van tellen.** Draait de test zonder `tokenizers`, dan zijn
alle contextgroottes een schatting. De standaardverhouding is gemeten en klopt
binnen enkele procenten op dit corpus, maar op een ander corpus loopt hij uiteen
van ongeveer 4,0 tot 4,8 tekens per token. Draai in dat geval eerst
`stresstest calibrate`.

**Contextgroottes komen op ongeveer 70 % van het doel uit bij de start.** Dat is
opzet, geen fout: de resterende 30 % is de ruimte waarin de agentstappen tijdens
de run groeien, tot de sessie het doel raakt en wordt gecomprimeerd. Een
"32k-run" is dus een sessie die naar 32k toe groeit, niet een die er meteen op
begint.

**Schijfruimte.** 31 GB model, plus de cache, plus pip. Onder de 150 GB loop je
er halverwege tegenaan.

---

## 10. Hoe het harnas werkt

### Het gedragsmodel

Een klas bestaat uit vier soorten studenten. De verdeling en het gedrag staan in
`config/default.json` en zijn aan te passen.

| Persona | Aandeel | Denktijd tussen instructies | Stappen per burst | Contextgedrag |
|---|---|---|---|---|
| Doorpakker | 15 % | 20 – 60 s | 8 – 15 | Groeit door tot 60–80k, wist zelden |
| Gemiddelde | 45 % | 60 – 180 s | 3 – 8 | Groeit tot 30–50k |
| Worstelaar | 25 % | 2 – 8 min | 2 – 4 | Blijft klein, begint regelmatig opnieuw |
| Afhaker | 15 % | 5 – 20 min inactief | 1 – 3 | Klein, sporadisch |

Denktijden worden getrokken uit een **lognormale** verdeling, waarbij de
genoemde waarden ongeveer het tiende en negentigste percentiel zijn. Echte
denktijd heeft een lange staart; uniform trekken zou juist de toevallige
samenloop wegpoetsen die de pieken maakt.

De activiteitsniveaus *rustig* en *intensief* verschuiven deze verdeling —
rustig heeft meer worstelaars en afhakers, intensief meer doorpakkers — en
comprimeren of rekken de denktijd. Het gedrag per persona blijft gelijk.

### De berichten

De context bestaat uit **echte broncode**: drie publieke voorbeeldprojecten
(dezelfde applicatie in Django, React en Laravel, samen een paar duizend
regels). Gegenereerde vulling zou de tokenverdeling en vooral het gedrag van
prefix caching verkeerd weergeven. Er komt geen code van Firda en geen enkel
studentgegeven aan te pas.

De berichten zijn opgebouwd zoals een echte agent dat doet, en bewust gelaagd:

```
[systeemprompt met gereedschapsdefinities]   identiek voor iedereen, altijd
[opdrachtomschrijving]                       identiek voor iedereen, altijd
[gedeeld projectskelet]                      identiek voor iedereen, instelbaar
[eigen bestanden van de student]             per student
[live agentstappen]                          groeit tijdens de run
```

Alles boven de eerste student-specifieke byte is een cachehit voor de hele klas.
Daarom is het aandeel gedeelde projectbasis een eigen as in de matrix: we willen
weten hoeveel het scheelt.

De gereedschappen zijn nep (`read_file`, `edit_file`, `run_tests`,
`search_code`, `run_command`) met plausibele uitvoer. Dat volstaat: het gaat om
de berichtstructuur, want die bepaalt het cachegedrag.

### Wat er gemeten wordt

Per verzoek: time to first token, doorlooptijd, decodesnelheid, tokentelling,
fouten. Per instructie: de doorlooptijd van de hele burst — wat de student
werkelijk ervaart, en wat in vrijwel alle benchmarks ontbreekt. Van de server,
elke twee seconden uit `/metrics`: prefix cache hit rate, preempties,
KV-cachebezetting, wachtrijdiepte, doorvoer gesplitst naar prefill en decode.

### Wat dit niet is

Geen benchmark van modelkwaliteit: we meten capaciteit en latentie, niet of de
code deugt. Geen vergelijking tussen inferentie-engines; vLLM is al gekozen.
Geen productieopstelling.

### Afhankelijkheden

De kern draait op de Python-standaardbibliotheek. Dat is een bewuste keuze: dit
moet over twee jaar op een verse gehuurde machine nog werken.

Daarbovenop staan twee pakketten, allebei optioneel maar niet gelijkwaardig.
`tokenizers` geeft exacte contextgroottes en is sterk aanbevolen, want de
contextgrootte is een as van de matrix; zonder dat pakket schat het harnas en
zegt het dat er bij. `matplotlib` levert PNG-versies van de grafieken naast de
SVG's die er altijd zijn. Zonder beide draait alles door.

---

## Commando's op een rij

```bash
python3 -m stresstest doctor                 # controleer endpoint, metrics, corpus
python3 -m stresstest calibrate              # ijk de tokenschatting op dit corpus
python3 -m stresstest plan --price-per-hour 3.36
python3 -m stresstest corpus                 # haal de voorbeeldprojecten binnen
python3 -m stresstest matrix                 # fase 1, ~5u40
python3 -m stresstest matrix --only rampup   # alleen de klifzoeker, ~40 min
python3 -m stresstest lesson                 # fase 2, ~1u40
python3 -m stresstest run --students 24 --context 48000
python3 -m stresstest report results/<map>   # grafieken en conclusie opnieuw
python3 -m stresstest mock -- --port 8000    # nep-vLLM om het harnas te testen
```
