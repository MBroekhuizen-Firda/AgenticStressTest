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

**Oplossing:** `export VLLM_USE_FLASHINFER_SAMPLER=0` voor je vLLM start.

FlashInfer wordt gebruikt voor top-k/top-p **sampling**, los van de
attention-backend — daarom helpt een andere backend kiezen niet. Voor deze
meting maakt het niets uit: het harnas draait op temperatuur 0, dus welke
implementatie de sampling doet verandert de uitkomst niet.

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
`pip install flashinfer-python`) en zet de variabele.

Zoek niet naar de "juiste" versie. vLLM 0.28 pint `flashinfer-python==0.6.16.post3`
en pip klaagt over elke andere, maar ook de gepinde versie compileert hier niet:
het probleem is de toolkit, niet de versie.

### `Unknown vLLM environment variable detected: VLLM_ATTENTION_BACKEND`

Die variabele bestaat niet meer in vLLM 0.28. Zetten heeft geen effect en de
waarschuwing is het enige dat je ervan merkt. Kies de backend via
serverargumenten, of laat vLLM zelf kiezen.

### `--kv-cache-dtype fp8` verandert de backendkeuze

Met `auto` koos vLLM hier FLASH_ATTN, met `fp8` de FlashInfer-backend — en
die compileert dus niet. Faalt het alleen bij fp8, dan is dat het spoor.
`KV_CACHE_DTYPE=auto` is de uitweg; dat kost KV-ruimte en dus gelijktijdige
gebruikers, maar levert een eerlijke ondergrens. Noteer welke dtype je gebruikt
hebt — de conclusie hangt eraan.

### De laatste regels van het log zijn niet de oorzaak

vLLM print een Python-traceback; de reden staat erbóven, vaak 50 regels
hoger. Zoek op `ValueError`, `RuntimeError`, `no available memory`,
`unrecognized arguments`. `scripts/pod.sh` licht die regels zelf uit.

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
- **`export` vóór het script, niet erin.** Het script geeft zijn omgeving door
  aan vLLM, maar wat er niet is kan het niet doorgeven.

## De meting zelf

- **Twee runs tegelijk meten niets.** Ze delen GPU, poort en resultaatmap, en
  achteraf zie je aan de getallen niet dat het gebeurd is. `scripts/pod.sh`
  weigert het nu; controleer anders met `pgrep -af "pod.sh"`.
- **Zonder `/metrics` meet je alleen latentie.** De hoofdvraag hangt op
  preempties, prefix-cache-hitrate en KV-bezetting. Ontbreken die reeksen, dan
  is de run zinloos — daarom stopt het script erop.
- **`--detach` gebruikt `nohup`, geen tmux.** De run overleeft je SSH-sessie.
  Meekijken met `scripts/pod.sh log -f`; ctrl-C stopt het kijken, niet de test.
- **Zet een doodsklok.** Een run die 's nachts vastloopt kost anders tot de
  ochtend GPU-huur door: `--deadman auto --shutdown`.

## Toegang

Plak geen HF-token of SSH-sleutel in een chat: die belandt in een wegwerpbare
container én in het gesprekslogboek. Zet het token op de pod zelf neer.
