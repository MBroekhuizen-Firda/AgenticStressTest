# Terug naar één main

Er staan zes branches op de remote. Vijf daarvan zijn tijdelijk bedoeld en
niemand heeft ze opgeruimd, dus het is niet meer op het oog te zien welke nog
iets bevat dat nergens anders staat. Dit bestand zegt per branch wat erin zit,
wat ermee moet, en waarom er steeds nieuwe bijkomen.

Peildatum: 10 september 2026. Controleer de shas voordat je iets weggooit.

## De stand

| Branch | Voor op main | Achter | Wat er uniek in zit |
|---|---|---|---|
| `main` | — | — | de waarheid |
| `claude/nvidia-rtx-pro-analysis-update-cxw2y4` (PR #24) | 2 | 2 | de matrixmeting van 9 september (`20260909-185135_matrix`) en de analyse die daarop is herschreven |
| `claude/blackwell-rtx-pro-results-yjocr0` | 1 | 13 | één correctie in de snelstart van `README.md` |
| `resultaten/…-20260909-142908` | 1 | 6 | niets |
| `resultaten/…-20260909-184641` | 1 | 6 | niets (zelfde commit `bcf5251` als de vorige) |
| `resultaten/…-20260910-001749` | 2 | 6 | de meetmap van 9 september, in `results/` in plaats van `results-van-de-gpu/` |

## Wat er per branch moet gebeuren

### 1. PR #24 — samenvoegen

`claude/nvidia-rtx-pro-analysis-update-cxw2y4` draagt als enige iets wat
nergens anders in de repo staat op een plek waar het bewaard blijft: de
matrixmeting van 9 september onder `results-van-de-gpu/20260909-185135_matrix`,
plus `CONCLUSIES.md`, `OPZET-VERVOLGMETING.md` en `RESULTATEN.md` herschreven
naar die ene meetsessie. Dat is precies de tegenspraak die issue #20 beschrijft
(een matrix van 8 september onder het oude gedragsmodel met een lesvalidatie van
9 september onder het nieuwe, in één bestand) — deze PR haalt hem eruit.

De branch is conflictvrij tegen `main`. Samenvoegen, daarna de branch
verwijderen.

### 2. `claude/blackwell-rtx-pro-results-yjocr0` — al meegenomen

Eén commit (`ef79f08`), één bestand: de snelstart begon bij `git remote
set-url` in plaats van bij het klonen. Er hoort geen PR bij en de branch staat
dertien commits achter.

De commit is met `git cherry-pick -x` in deze branch overgenomen, zodat er geen
losse branch voor hoeft te blijven staan. Verwijderen zodra dit is samengevoegd.

### 3. De drie `resultaten/…`-branches — verwijderen

Deze worden aangemaakt door `push_results()` in `scripts/pod.sh`: elke meting
krijgt een eigen branch `resultaten/<kaart>-<tijdstempel>`. Dat is bewust — de
resultaten moeten van de gehuurde machine af voordat die uit gaat — maar
niemand voegt ze ooit samen, dus ze stapelen zich op.

Wat erin zit, na vergelijking van de boomobjecten:

* **`…-20260909-142908`** en **`…-20260909-184641`** wijzen naar dezelfde commit
  (`bcf5251`) en bevatten `results/20260908-180708_matrix` en
  `results/20260909-125855_les`. Die tweede branch is de run uit issue #18: de
  meting die in tien minuten klaar was omdat alles werd overgeslagen, en die
  daarom exact dezelfde inhoud pushte als de vorige.
* **`…-20260910-001749`** bevat daarnaast `results/20260909-185135_matrix`. Die
  map is byte voor byte gelijk aan
  `results-van-de-gpu/20260909-185135_matrix` in PR #24 (zelfde tree-object,
  `413ecb2d`).
* De `20260909-125855_les` op deze branches is identiek aan de kopie op `main`.
* De `20260908-180708_matrix` verschilt van de kopie op `main` alleen in
  regeleindes: de pod schreef die CSV's met CRLF, met een harnasversie van vóór
  de correctie in `write_csv`. De getallen zijn gelijk. De nieuwere meetmap op
  dezelfde branch heeft al LF.

Er zit dus niets in wat na het samenvoegen van PR #24 nog ergens ontbreekt.
Alle drie verwijderen.

```bash
git push origin --delete resultaten/NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-600W-20260909-142908
git push origin --delete resultaten/NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-600W-20260909-184641
git push origin --delete resultaten/NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition-600W-20260910-001749
```

## De volgorde

1. Deze PR samenvoegen (de vier issues plus de meegenomen README-commit).
2. PR #24 samenvoegen.
3. `claude/nvidia-rtx-pro-analysis-update-cxw2y4`,
   `claude/blackwell-rtx-pro-results-yjocr0` en de drie `resultaten/…`-branches
   verwijderen.
4. Wat overblijft is `main`.

Stap 1 en 2 raken elkaar niet: deze branch komt aan `scripts/pod.sh`,
`stresstest/`, `tests/` en `README.md`, PR #24 aan de meetmappen en de
analysedocumenten. Geen enkel bestand zit in allebei.

## Waarom het steeds opnieuw gebeurt

`push_results()` maakt per meting een branch en niemand ruimt die op. Dat is
geen fout in het script — de branch is de reddingsboei voor resultaten die
alleen op een gehuurde machine staan — maar het is wel de helft van een
werkwijze. De andere helft ontbreekt: na een meting hoort de meetmap via een PR
naar `results-van-de-gpu/` op `main` te gaan, en hoort de pod-branch daarna weg
te kunnen.

Twee afspraken die dat afmaken:

* **Een `resultaten/…`-branch is een tijdelijke opslagplaats, geen archief.**
  Zodra de meting onder `results-van-de-gpu/` op `main` staat, mag hij weg.
  Blijft hij staan, dan bestaat dezelfde meting op twee plekken met twee paden,
  en dat is precies waarom hierboven boom-voor-boom vergeleken moest worden.
* **Eén meting per rapport.** `RESULTATEN.md` in de hoofdmap noemt sinds deze
  PR per bronmap de meetdatum en de vingerafdruk van de opstelling, en
  waarschuwt bovenaan als de bronnen niet bij elkaar horen. Wie een meetmap
  promoveert, ziet dus meteen of het rapport eromheen nog klopt.

Blijft er onverhoopt toch een pod-branch staan die niet meer nodig is: hij is te
herkennen aan het feit dat `git diff` tegen `main` alleen `results/…` toont, en
dat diezelfde mappen onder `results-van-de-gpu/` op `main` staan.
