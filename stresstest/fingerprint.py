"""What a measurement was made with, in one comparable object.

A run id like `sweep_s20_c64k` says how many students and how much context,
and nothing else. Change the corpus, the behaviour model, the tokenizer or the
attention backend and the id stays the same -- so a resume that skips runs "we
already have" can quietly mix two different measurements into one report. That
happened: see issue #18.

The fingerprint is the rest of the description. It travels with every run in
`run.json` and with the directory in `environment.json`, and `differences()`
says which field moved.

Two levels, because not everything is cheap to know:

* `from_config()` needs only the configuration, so a shell script deciding
  which runs are still pending can afford it before anything is loaded.
* `from_run()` adds what only a loaded corpus and a built tokenizer can tell:
  how many files the corpus actually has, and whether token counting is exact.

`differences()` compares only the fields both sides carry. That is what lets
the cheap check and the full check share one stored fingerprint, and it means a
run recorded by an older version (no fingerprint at all) is reported as unknown
rather than as a mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Sequence

from .personas import personas_from_config, work_profiles_from_config

# A path to the model differs between two pods that hold the same model. It is
# recorded, because it says where the tokenizer came from, and never compared.
UNCOMPARED = ("tokenizer.path",)

# The engine settings are a special case. For an ordinary run they are part of
# the setup: resuming a FlashInfer matrix under TRITON_ATTN measures something
# else under the same run ids. For the `engine` group they are the subject --
# varying `--kv-cache-dtype` is the entire point of `engine_kv_fp8` -- so
# comparing them there would refuse exactly the runs that are supposed to
# differ. Hence: compared per run, except where an engine run is involved, and
# left out of the digest so that one matrix directory has one digest rather
# than one per variant.
ENGINE = "hardware"


def from_config(config: dict) -> dict:
    """Everything the configuration alone decides about the measurement."""
    behaviour = config.get("behaviour", {}) or {}
    corpus = config.get("corpus", {}) or {}
    tokenizer = config.get("tokenizer", {}) or {}
    hardware = config.get("hardware", {}) or {}
    personas = personas_from_config(behaviour.get("personas"))
    work = work_profiles_from_config(behaviour.get("work_profiles"))
    return {
        "seed": config.get("seed"),
        "corpus": {
            "groups": {name: sorted(block.get("project", []) or [])
                       for name, block in sorted((corpus.get("groups") or {}).items())},
            "extensions": corpus.get("extensions"),
            "shared_skeleton_fraction": corpus.get("shared_skeleton_fraction"),
            "shared_skeleton_files": corpus.get("shared_skeleton_files"),
            "local_path": corpus.get("local_path"),
        },
        "behaviour": {
            # The whole profile, not only name and share: a persona that thinks
            # twice as long is a different measurement under the same name.
            "personas": {p.name: _without_name(p.to_dict()) for p in personas},
            "work_profiles": {w.name: _without_name(w.to_dict()) for w in work},
            "activity_levels": behaviour.get("activity_levels"),
        },
        "tokenizer": {
            "prefer_exact": tokenizer.get("prefer_exact", True),
            "chars_per_token": tokenizer.get("chars_per_token"),
            "path": tokenizer.get("path"),
        },
        "hardware": {
            "attention_backend": hardware.get("attention_backend"),
            "kv_cache_dtype": hardware.get("kv_cache_dtype"),
        },
    }


def behaviour_digest(config: dict) -> str:
    """Hash of the behaviour model alone: the personas, the work profiles, the
    activity levels and the seed they are drawn with.

    Narrower than `digest()` on purpose. A report says two things about how it
    was made -- which measurement each half came from, and which class model
    every run in it shared -- and those are different claims that must not be
    printed under the same hash.
    """
    mark = from_config(config)
    return digest({"behaviour": mark["behaviour"], "seed": mark["seed"]})


def from_run(config: dict, corpus: Any = None, counter: Any = None) -> dict:
    """The configuration fingerprint plus what the loaded corpus and the
    tokenizer add: the file counts that decide how a context window fills, and
    whether the token counts are exact or estimated."""
    out = from_config(config)
    if corpus is not None:
        described = corpus.describe()
        groups = described.get("groups")
        out["corpus_files"] = {
            "name": described.get("name"),
            "files": described.get("files"),
            "per_group": ({name: block.get("files") for name, block in sorted(groups.items())}
                          if groups else None),
        }
    if counter is not None:
        out["tokenizer_exact"] = bool(counter.exact)
    return out


def _without_name(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "name"}


def digest(fingerprint: dict | None) -> str:
    """Short, stable hash of what was measured.

    Corpus, behaviour model, tokenizer, seed -- the things two measurements
    have to share before their numbers may be compared or folded into one
    report. Deliberately not the engine settings: a matrix contains runs that
    vary those on purpose, and a directory needs one digest, not five.
    """
    if not fingerprint:
        return "onbekend"
    canonical = json.dumps(_comparable(fingerprint, ignore=(ENGINE,)),
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _comparable(fingerprint: dict, ignore: Sequence[str] = ()) -> dict:
    out = json.loads(json.dumps(fingerprint, sort_keys=True))
    for path in tuple(UNCOMPARED) + tuple(ignore):
        _drop(out, path.split("."))
    return out


def _drop(node: Any, keys: Sequence[str]) -> None:
    for key in keys[:-1]:
        if not isinstance(node, dict) or key not in node:
            return
        node = node[key]
    if isinstance(node, dict):
        node.pop(keys[-1], None)


# --------------------------------------------------------------------------
# Comparing
# --------------------------------------------------------------------------

def differences(stored: dict | None, current: dict | None,
                ignore: Sequence[str] = ()) -> list[tuple[str, Any, Any]]:
    """Fields present in both fingerprints whose value differs.

    Absent on either side is not a difference: a fingerprint written by the
    cheap check carries fewer fields than one written by a full run, and the
    two must not accuse each other.
    """
    if not stored or not current:
        return []
    found: list[tuple[str, Any, Any]] = []
    _walk(_comparable(stored, ignore), _comparable(current, ignore), "", found)
    return sorted(found)


def _walk(left: Any, right: Any, path: str, found: list) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) & set(right)):
            _walk(left[key], right[key], f"{path}.{key}" if path else key, found)
        return
    if left != right:
        found.append((path, left, right))


def describe(diffs: Sequence[tuple[str, Any, Any]]) -> list[str]:
    return [f"{path}: was {_short(was)}, nu {_short(now)}" for path, was, now in diffs]


def _short(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text if len(text) <= 120 else text[:117] + "..."


# --------------------------------------------------------------------------
# Reading back what is on disk
# --------------------------------------------------------------------------

def stored_in(directory: str) -> dict[str, tuple[str, dict]]:
    """Kind and fingerprint per run in a results directory, keyed by run id.

    The kind comes along because an `engine` run is compared differently: its
    engine settings are what it measures.

    A run without a fingerprint -- measured before this existed -- is left out
    entirely; `unfingerprinted()` is what reports those.
    """
    out: dict[str, tuple[str, dict]] = {}
    runs = os.path.join(directory, "runs")
    if not os.path.isdir(runs):
        return out
    for run_id in sorted(os.listdir(runs)):
        payload = _read(os.path.join(runs, run_id, "run.json"))
        if isinstance(payload.get("fingerprint"), dict):
            kind = (payload.get("spec") or {}).get("kind", "")
            out[run_id] = (kind, payload["fingerprint"])
    return out


def unfingerprinted(directory: str) -> list[str]:
    runs = os.path.join(directory, "runs")
    if not os.path.isdir(runs):
        return []
    out = []
    for run_id in sorted(os.listdir(runs)):
        path = os.path.join(runs, run_id, "run.json")
        if os.path.exists(path) and not isinstance(_read(path).get("fingerprint"), dict):
            out.append(run_id)
    return out


def environment_of(directory: str) -> dict:
    return _read(os.path.join(directory, "environment.json"))


def _read(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

class Mismatch(Exception):
    """A results directory was measured with a different setup."""

    def __init__(self, directory: str, offenders: dict[str, list]) -> None:
        self.directory = directory
        self.offenders = offenders
        super().__init__(self.message())

    def message(self, with_options: bool = True) -> str:
        lines = [f"{self.directory} is gemeten met een andere opstelling dan de huidige."]
        for run_id, diffs in sorted(self.offenders.items()):
            lines.append(f"  {run_id}:")
            lines += [f"    {line}" for line in describe(diffs)]
        if with_options:
            lines.append("")
            lines.append("Runs met dezelfde naam zouden hier iets anders meten dan wat er al")
            lines.append("staat, en het rapport eromheen zou de twee door elkaar halen. Kies er een:")
            lines.append("  - meet in een nieuwe map:  RESULTS_DIR=results/$(date +%Y%m%d-%H%M%S)_matrix")
            lines.append("  - of hervat toch, als je weet waarom:  --resume-anyway")
        return "\n".join(lines)


def check(directory: str, current: dict, adding_engine_runs: bool = False) -> dict[str, list]:
    """Runs in `directory` whose fingerprint differs from `current`.

    `adding_engine_runs` says what is about to be measured. An engine variant
    restarts vLLM with its own flags, so its engine settings differ from every
    baseline run in the directory by design; comparing them would turn the
    engine group into a stop. Everything else about the setup is still
    compared, and so is everything about the runs already stored.
    """
    offenders: dict[str, list] = {}
    for run_id, (kind, stored) in stored_in(directory).items():
        ignore = (ENGINE,) if adding_engine_runs or kind == "engine" else ()
        diffs = differences(stored, current, ignore=ignore)
        if diffs:
            offenders[run_id] = diffs
    environment = environment_of(directory).get("fingerprint")
    if isinstance(environment, dict) and not offenders:
        # environment.json is rewritten by every run, so its engine settings
        # are whichever variant went last. Only the measurement part of it
        # means anything at directory level.
        diffs = differences(environment, current, ignore=(ENGINE,))
        if diffs:
            offenders["environment.json"] = diffs
    return offenders


def guard(directory: str, current: dict, allow_mismatch: bool = False,
          warn=None, adding_engine_runs: bool = False) -> dict[str, list]:
    """Raise unless the runs already in `directory` were measured like this.

    With `allow_mismatch` the differences are reported and the caller carries
    on -- for someone who knows why the setup changed and wants the runs in one
    directory anyway.
    """
    offenders = check(directory, current, adding_engine_runs=adding_engine_runs)
    if offenders and not allow_mismatch:
        raise Mismatch(directory, offenders)
    if offenders and warn is not None:
        warn(Mismatch(directory, offenders).message(with_options=False))
        warn("--resume-anyway: toch hervat. De runs in deze map zijn dus niet "
             "allemaal met dezelfde opstelling gemeten.")
    if warn is not None:
        old = unfingerprinted(directory)
        if old:
            warn(f"{len(old)} runs in {directory} dragen geen vingerafdruk "
                 f"({', '.join(old[:4])}{', ...' if len(old) > 4 else ''}); "
                 f"of ze bij deze opstelling horen is niet na te gaan.")
    return offenders
