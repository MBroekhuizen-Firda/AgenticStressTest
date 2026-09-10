# Valkuilen

Alles wat in de praktijk misging bij het draaien van
Qwen3-Coder-30B-A3B-Instruct-FP8 op vLLM, op een RTX PRO 6000 Blackwell in een
RunPod-container. Kort opgeschreven, zodat een volgende keer een kwartier kost
in plaats van een avond. Per punt: wat je ziet, wat het is, wat je doet.

Dit gaat over de opstelling. Valkuilen in de *meting* zelf — instellingen die
je een verkeerd antwoord opleveren — staan in
[hoofdstuk 9 van de README](README.md#9-bekende-valkuilen).

## vLLM start niet op

### `RuntimeError: FlashInfer requires GPUs with sm75 or higher`

De kaart is sm_120, dus de melding klopt niet. Twee regels hoger in
`vllm.log` staat de echte reden:

```
Failed to get device capability: SM 12.x requires CUDA >= 12.9
```

FlashInfer compileert zijn kernels bij het opstarten (JIT) met de nvcc uit de
container. Is die ouder dan 12.9, dan kan hij de kaart niet uitlezen en valt
hij terug op een onzinnige ondergrens. De torch-versie zegt hier niets over:
torch kan `+cu130` zijn terwijl de toolkit ouder is.

vLLM 0.28 grijpt op **twee** plekken naar FlashInfer, en je moet om allebei
heen. Aan de traceback zie je welke van de twee het was:

1. **De sampler.** `topk_topp_sampler.py` → `flashinfer/sampling.py` →
   `check_cuda_arch`. FlashInfer wordt gebruikt voor top-k/top-p sampling,
   los van de attention-backend. Oplossing:
   `export VLLM_USE_FLASHINFER_SAMPLER=0` voor je vLLM start. Voor deze
   meting maakt het niets uit: het harnas draait op temperatuur 0, dus welke
   implementatie de sampling doet verandert de uitkomst niet.
2. **De attention-backend.** `Using FlashInfer backend` in het log, en een
   traceback door `flashinfer/jit/attention/modules.py`
   (`gen_customize_batch_prefill_module`) → `check_cuda_arch`. Met
   `--kv-cache-dtype fp8` kiest vLLM FlashInfer als backend, omdat FLASH_ATTN
   op deze kaart geen FP8-cache kan bedienen (dat vraagt FA3 op Hopper of FA4
   op B200). De sampler uitzetten helpt dan niet: de engine sterft al bij het
   bouwen van de prefill-kernel. Oplossing: `--attention-backend TRITON_ATTN`
   als serverargument. Dat is de andere backend die vLLM zelf noemt
   ("out of potential backends: ['FLASHINFER', 'TRITON_ATTN']") en die kan
   wél met een FP8-cache overweg. Het is een andere backend en dus andere
   getallen — noteer het bij de resultaten.

`scripts/pod.sh` doet allebei zelf, op grond van wat het log zegt: wijst de
traceback naar de attention-backend, dan start het in één keer opnieuw met
`--attention-backend TRITON_ATTN` én zonder de FlashInfer-sampler; wijst hij
alleen naar de sampler, dan verandert alleen die. De keuze blijft staan voor
alle volgende enginevarianten en komt als `hardware.attention_backend` in
`environment.json`. Een `ATTENTION_BACKEND` die je zelf hebt gezet wordt
nooit vervangen.

Blijft het daarna nog misgaan op FlashInfer, dan roept een derde onderdeel van
vLLM hem aan (kijk in de traceback welk). Dan is een image met een toolkit van
12.9 of nieuwer de zekere uitweg; `/workspace` blijft staan, dus het model
hoeft niet opnieuw gedownload.

### `ModuleNotFoundError: No module named 'flashinfer'`

Dit krijg je als je FlashInfer weghaalt om van het vorige punt af te komen.
Niet doen. vLLM 0.28 importeert het pakket ook als het het niet gebruikt:

```
topk_topp_sampler.py:98   and flashinfer_sampler_supported()
topk_topp_sampler.py:51   from vllm.v1.attention.backends.flashinfer import FlashInferBackend
```

Dat is de laatste term van een and-keten waar `VLLM_USE_FLASHINFER_SAMPLER`
eerder in staat. Met die variabele op 0 wordt de import niet eens gedaan, en
dan maakt het niet uit of het pakket er is.

**Oplossing:** laat het pakket staan (of zet het terug met
`pip install flashinfer-python`) en zet de variabele. Zelfde verhaal voor de
attention-backend: `--attention-backend TRITON_ATTN` betekent dat FlashInfer
niet gebouwd wordt, niet dat het pakket weg kan.

Zoek niet naar de "juiste" versie. vLLM 0.28 pint `flashinfer-python==0.6.16.post3`
en pip klaagt over elke andere, maar ook de gepinde versie compileert hier niet:
het probleem is de toolkit, niet de versie.

### `429 Too Many Requests` van huggingface.co

```
ERROR repo_utils.py:117 429 Too Many Requests for url:
  https://huggingface.co/api/models/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8/tree/main
huggingface_hub.errors.HfHubHTTPError
```

vLLM vraagt bij **elke** start de bestandslijst van de repo op bij de Hub, ook
als de 31 GB al lang op schijf staat. De Hub knijpt anoniem verkeer af per
IP-adres, en dat adres deel je op een gehuurde pod met alle andere containers
op die machine. Je start dus stuk op verkeer van iemand anders, terwijl je
zelf niets van het netwerk nodig hebt.

Verwarrend is dat het log verder niets zegt: geen geheugen, geen kernels. Wie
de melding onderaan leest ziet alleen een traceback en gaat `--max-model-len`
verlagen — dat helpt niet.

**Oplossing:** geef vLLM de map in plaats van de repo-naam, en zet de Hub uit.

```bash
export HF_HUB_OFFLINE=1
vllm serve /workspace/hf/hub/models--Qwen--Qwen3-Coder-30B-A3B-Instruct-FP8/snapshots/<hash> \
  --served-model-name qwen3-coder ...
```

Dezelfde gewichten, dezelfde config, dus aan de meting verandert niets.
`scripts/pod.sh` doet dit zelf: ziet het een 429 en staat het model compleet
op schijf, dan start het opnieuw vanaf de snapshotmap en houdt het die keuze
vast voor alle volgende enginevarianten. Staat het model er nog niet, dan
wacht het (60 s, 120 s, 240 s) en probeert het opnieuw.

**Voorkomen:** zet een Hugging Face-token in de omgeving. Ingelogd verkeer
krijgt een veel ruimere limiet dan anoniem. Een read-token is genoeg — dit
model heeft geen toegangsvoorwaarden, het gaat puur om de limiet.

```bash
export HF_TOKEN=hf_...        # op de pod zelf, niet in een chat
```

### `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`

Die variabele bestaat niet meer in vLLM 0.28. Zetten heeft geen effect en de
waarschuwing is het enige dat je ervan merkt: vLLM kiest gewoon zelf, en dat
is dan FlashInfer. Kies de backend via het serverargument
`--attention-backend`, of laat vLLM zelf kiezen. `scripts/pod.sh` geeft de
vlag door; `ATTENTION_BACKEND=...` (of een nog geëxporteerde
`VLLM_ATTENTION_BACKEND`) in de omgeving is genoeg.

### `--kv-cache-dtype fp8` verandert de backendkeuze

Met `auto` koos vLLM hier FLASH_ATTN, met `fp8` de FlashInfer-backend — en
die compileert dus niet. Faalt het alleen bij fp8, dan is dat het spoor.
Twee uitwegen. `--attention-backend TRITON_ATTN` houdt de FP8-cache en dus
het aantal sessies dat erin past; dat is wat `scripts/pod.sh` doet.
`KV_CACHE_DTYPE=auto` is de andere: dat kost KV-ruimte en dus gelijktijdige
gebruikers, maar levert een eerlijke ondergrens. Noteer in beide gevallen wat
je gebruikt hebt — de conclusie hangt eraan.

### `WorkerProc initialization failed due to an exception in a background process`

Alleen met `--tensor-parallel-size 2` of hoger. vLLM start dan een
werkproces per kaart, en wat je onderaan het log ziet is de API-server die
meldt dat er eentje is weggevallen:

```
Exception: WorkerProc initialization failed due to an exception in a
  background process. See stack trace for root cause.
RuntimeError: Engine core initialization failed. Failed core proc(s): {}
```

Die "stack trace for root cause" staat *niet* onderaan. Hij staat honderden
regels hoger, in de regels van het werkproces zelf — de regels die met
`(VllmWorker rank=1 pid=...)` beginnen. `tail` van het log laat precies het
verkeerde deel zien.

```bash
grep -n '^(VllmWorker' /workspace/.stresstest/vllm.log | tail -30
```

Vier oorzaken. De eerste is de enige die niets met de opstelling te maken
heeft en tegelijk de enige waar geen vlag tegen helpt, dus begin daar:

0. **De driver is te oud voor de torch in de image.** In de regels van de
   worker staat dan:

   ```
   File ".../vllm/v1/worker/gpu_worker.py", line 412, in init_device
       torch._C._cuda_init()
   RuntimeError: The NVIDIA driver on your system is too old (found version 12080).
   ```

   `12080` is CUDA 12.8, wat een 570-driver levert. De torch in de image is
   dan gebouwd tegen CUDA 13, en die wil 580 of nieuwer. Dit gaat *niet* over
   tensor parallelism: elke worker sterft bij het openen van de kaart, ook met
   `TENSOR_PARALLEL=1` — daar is het alleen minder zichtbaar, omdat er dan één
   proces omvalt in plaats van twee.

   Verwarrend is dat `nvidia-smi` er tevreden uitziet en de kaarten gewoon in
   het log staan. De driver is nieuw genoeg voor de *kaart* (570+ voor
   Blackwell) en te oud voor de *image*. Twee getallen die allebei kloppen.

   **Oplossing:** een image met een torch die bij de driver past (`cu128` bij
   een 570-driver), of een pod met een driver van 580 of nieuwer.
   `/workspace` blijft staan, dus het model hoeft niet opnieuw gedownload.
   `scripts/pod.sh` vraagt dit nu vóór de download aan torch zelf
   (`torch.cuda.init()`, kost een seconde) en stopt daar, in plaats van een
   kwartier later in een worker.

1. **Te weinig gedeeld geheugen.** De workers praten met elkaar over
   `/dev/shm`, en een container die zonder `--shm-size` is gestart krijgt
   64 MB. In het log van de worker staat dan `OSError: [Errno 28] No space
   left on device` met een `/psm_...`-naam erbij, of een `Bus error`.
   Controleer met `df -h /dev/shm`; het moet gigabytes zijn, niet megabytes.
   Op RunPod stel je dit in bij het aanmaken van de pod — achteraf kan het
   niet, een draaiende container kan zijn eigen `/dev/shm` niet vergroten.
2. **De kaarten bereiken elkaar niet.** NCCL wil de directe weg tussen de
   kaarten (PCIe peer-to-peer), en twee consumentenkaarten in één pod (twee
   5090's bijvoorbeeld) hebben die niet altijd. In het log staat
   `ncclUnhandledCudaError`, `NCCL error` of `DistBackendError`. Oplossing:
   `export NCCL_P2P_DISABLE=1`, dan gaat het verkeer via het werkgeheugen.
   Dat is trager voor alles wat de kaarten uitwisselen, dus het hoort bij de
   resultaten genoteerd te worden. Let op het verschil met `custom allreduce
   is disabled because your platform lacks GPU P2P capability`: dat is een
   `INFO`-regel op een start die verder helemaal goed gaat.
3. **Er zijn minder kaarten dan processen.** `TENSOR_PARALLEL=2` op één
   zichtbare kaart. Kijk wat `nvidia-smi -L` zegt en of
   `CUDA_VISIBLE_DEVICES` de rest wegfiltert.

`scripts/pod.sh` vangt alle vier af. Het nulde en het derde geval worden vóór
de meting geweigerd (bij het derde gaat `--force` er langs), een te kleine
`/dev/shm` levert een waarschuwing op voordat het model geladen wordt, en
noemt het log NCCL, dan
start het script één keer opnieuw met `NCCL_P2P_DISABLE=1` en zegt erbij dat
dat de getallen raakt. Komt de server ook dan niet omhoog, dan zet het de
regels van de workers zelf onder de foutmelding — de regels waar de oorzaak
in staat.

### De laatste regels van het log zijn niet de oorzaak

vLLM print een Python-traceback; de reden staat erbóven, vaak 50 regels
hoger. Zoek op `ValueError`, `RuntimeError`, `no available memory`,
`unrecognized arguments`. `scripts/pod.sh` licht die regels zelf uit — en
sinds kort zonder de traceback zelf mee te nemen: elke frameregel van vLLM
begint met `ERROR`, dus een zoektocht naar "Error" leverde twaalf regels
`return func(*args, **kwargs)` op en niet de ene regel die iets zei.

### Het opstarten duurt minuten

DeepGEMM en consorten compileren kernels tijdens het opwarmen. Stilte in het
log is niet hetzelfde als vastlopen. `SERVER_START_TIMEOUT_S` staat op 900 s.

### `please use at least NVCC 12.9 for the best DeepGEMM performance`

Geen fout, wel relevant voor de conclusie: met een oudere toolkit zijn de
kernels niet optimaal. De doorvoer die je meet is dan een **ondergrens**.
Voor een aanschafbeslissing is dat de goede kant om aan te zitten, maar het
hoort in het rapport genoemd te worden.

## RunPod

- **De container disk wordt gewist als je de pod stopt.** Alleen het
  netwerkvolume (`/workspace`) blijft. Zet `HF_HOME`, de repo en de resultaten
  daar neer, anders download je 29 GB opnieuw.
- **Stop is niet terminate.** Stoppen bewaart het volume, termineren gooit
  alles weg. De doodsklok in het script stopt, niet termineert.
- **Ruim genomen: 80 GB container disk, 100 GB volume.** `pip install` schrijft
  naar de container disk, niet naar `/workspace` — daar is vLLM alleen al enkele
  GB. Vol lopen merk je pas halverwege een installatie.
- **Kaartvariant controleren.** `RTX PRO 6000` is de Server Edition, `WK` de
  Workstation. Max-Q draagt dezelfde 96 GB maar de helft van het vermogen, en
  dat zie je niet aan de naam in de lijst:

  ```bash
  nvidia-smi --query-gpu=name,power.default_limit --format=csv
  ```

  Rond 600 W is de volle kaart, rond 300 W een Max-Q. Meet je op een Max-Q en
  rapporteer je dat als de kaart uit de aanvraag, dan klopt het getal niet.

### De resultaten van de pod halen

Probeer eerst de korte weg: **`scripts/pod.sh push`** zet wat er gemeten is
alsnog op een branch in de repo. `all` doet dat zelf aan het eind, maar een run
die eerder stopt — een weigering bij de start, een verbroken verbinding, een
`kill` — laat alles op de huurschijf staan. Dan hoef je de rest van deze
paragraaf niet.

Drie dingen staan tussen jou en een `scp` die werkt. Ze geven alle drie een
andere fout, en samen kosten ze een halve avond.

**De poort is niet 22.** Onder *Connect* staan twee SSH-regels. De bovenste
gaat via de proxy (`ssh.runpod.io`) en draagt **geen scp of sftp** — het
dashboard zegt dat er zelf bij. Die verbinding valt bij een kopieeropdracht
stil dicht (`Connection closed`). Je hebt de tweede nodig, *SSH over exposed
TCP*, met een hoge poort rond de 30000:

```
ssh root@157.157.221.29 -p 33030 -i ~/.ssh/id_ed25519
```

Dat nummer verandert bij elke nieuwe pod. Kleine `-p` bij `ssh`, hoofdletter
`-P` bij `scp`. Probeer je 22, dan krijg je geen weigering maar een
`Connection timed out` — er luistert daar niets. Staat die tweede regel er
helemaal niet, dan is poort 22 bij het aanmaken niet als *exposed TCP port*
opgegeven; toevoegen kan niet meer op een draaiende pod.

**Je publieke sleutel wordt alleen bij het opstarten geïnstalleerd.** RunPod
schrijft de sleutels uit je accountinstellingen in `/root/.ssh/authorized_keys`
op het moment dat de pod start. Zet je een sleutel er daarna bij, dan bereikt
die een draaiende pod niet. Je merkt het aan een wachtwoordprompt — en omdat er
geen root-wachtwoord is, blijft die eeuwig weigeren. Kijk met `ssh -v` of de
sleutel wordt aangeboden én geaccepteerd. Een nieuwe pod starten met dezelfde
sleutel is de schone oplossing; het netwerkvolume gaat gewoon mee.

**Geef de sleutel expliciet mee.** Zonder `-i` probeert OpenSSH alleen de
standaardnamen in `~/.ssh/`. Let op de naamgeving: `ssh-keygen -f runpod.pub`
levert de *private* sleutel op als `runpod.pub` en de publieke als
`runpod.pub.pub` — je geeft dan de eerste mee, hoe verkeerd dat ook oogt.

Twee uitwegen als de sleutel niet werkt en je de pod niet opnieuw wilt starten:

- De **web terminal** in het dashboard vraagt geen sleutel. Daar plak je de
  sleutel alsnog in `authorized_keys`, of je verstuurt de tarball met
  `runpodctl send <bestand>`; lokaal haal je hem op met `runpodctl receive
  <code>`. Dat loopt over hetzelfde kanaal als de proxy en heeft geen API-sleutel
  nodig.
- **Jupyter** op poort 8888 heeft een bestandsbrowser met downloadknop. Alleen
  losse bestanden, geen mappen, dus pak eerst in — `scripts/pod.sh all` doet dat
  aan het eind al (`/workspace/stresstest-results-<datum>.tar.gz`), en anders:

  ```bash
  tar -czf /workspace/results.tar.gz -C /workspace/AgenticStressTest results
  ```

Is `/workspace` leeg op de nieuwe pod, dan hangt er een ander netwerkvolume
onder dan bij de run. De resultaten staan op het volume, niet op de container
disk; controleer bij de pod-instellingen welk volume gekoppeld is.

## Shell en gereedschap

- **`tail -20 map/pod-*.log` weigert zodra er twee logbestanden zijn:**
  `tail: option used in invalid context -- 2`. De verkorte `-NUM`-vorm mag maar
  bij één bestand. Gebruik `tail -n 20` of `scripts/pod.sh log`.
- **`set -Eeuo pipefail` plus `$(ls ... | head -1)` breekt de run af** als de map
  nog niet bestaat: `ls` faalt, `pipefail` geeft dat door, `set -e` stopt alles.
  Precies bij de eerste schone start, dus na de download. Vang het af met
  `|| true` in een helper.
- **`pkill -f "iets"` matcht ook je eigen commandoregel**, inclusief de tekst
  van een heredoc. Je schiet dan je eigen shell dood. Schrijf het patroon zo
  dat het zichzelf niet vindt (`"stresstest[ ]mock"`).
- **`printf "%s" "$body" | grep -q` liegt over grote invoer.** `grep -q` stopt
  bij zijn eerste treffer; is `$body` groter dan de pijpbuffer (64 KB), dan
  schrijft `printf` op dat moment nog en gaat dood aan SIGPIPE. `pipefail`
  geeft die status door en je test zegt "niet gevonden" over iets dat er wél
  staat. Of het misgaat hangt ervan af waar in de invoer de treffer zit: bovenin
  wel, onderin niet. Dit heeft twee pods gekost (zie *De meting zelf*). Gebruik
  een here-string: `grep -q ... <<<"$body"` heeft geen schrijvend proces dat
  omvalt.
- **`export` vóór het script, niet erin.** Het script geeft zijn omgeving door
  aan vLLM, maar wat er niet is kan het niet doorgeven.

## De meting zelf

- **Twee runs tegelijk meten niets.** Ze delen GPU, poort en resultaatmap, en
  achteraf zie je aan de getallen niet dat het gebeurd is. `scripts/pod.sh`
  weigert het nu; controleer anders met `pgrep -af "pod.sh"`.
- **Vlak na het opstarten is `/metrics` nog niet compleet.** De API-server
  antwoordt op `/v1/models` terwijl de engine zijn reeksen nog registreert;
  een scrape van een seconde later miste hier `vllm:num_preemptions_total`,
  terwijl dezelfde server hem even later gewoon had staan (op 0). Kijk niet
  één keer, maar met een paar seconden tussenruimte (`scripts/pod.sh` doet dat
  zelf). Let op: `vllm:num_preemptions_created` is een tijdstempel van
  prometheus_client, niet het aantal preempties.
- **Zonder `/metrics` meet je alleen latentie.** De hoofdvraag hangt op
  preempties, prefix-cache-hitrate en KV-bezetting. Ontbreken die reeksen, dan
  is de run zinloos — daarom stopt het script erop.
- **Een poortwachter die ten onrechte weigert kost een hele pod.** Twee pods
  (2× RTX 5090 en een PRO 6000 MIG) stopten binnen een seconde na
  "vLLM draait" op `/metrics mist: KV-bezetting`, terwijl `vllm:kv_cache_usage_perc`
  gewoon in de body stond en het harnas hem in dezelfde run had uitgelezen. De
  oorzaak zat in de pijp, niet in vLLM (zie *Shell en gereedschap*). Wat het
  duur maakte: het model was al binnen, de doodsklok stond op acht uur, en de
  pod stond daarna uren stil te huren. Sindsdien haalt een fatale fout de
  doodsklok naar voren (`ABORT_GRACE_HOURS`, standaard een half uur) en zegt de
  weigering erbij welke `vllm:`-reeksen er wél stonden — genoeg om in het log
  te zien of de reeks ontbreekt of de controle stuk is.
- **`--detach` gebruikt `nohup`, geen tmux.** De run overleeft je SSH-sessie.
  Meekijken met `scripts/pod.sh log -f`; ctrl-C stopt het kijken, niet de test.
- **Zet een doodsklok.** Een run die 's nachts vastloopt kost anders tot de
  ochtend GPU-huur door: `--deadman auto --shutdown`.

## Toegang

Plak geen HF-token of SSH-sleutel in een chat: die belandt in een wegwerpbare
container én in het gesprekslogboek. Zet het token op de pod zelf neer.
