"""Real source code as context.

Generated filler does not behave like code: the token distribution is wrong
and, far more importantly, prefix caching only tells you the truth when the
repeated prefixes are the ones a real agent would repeat. So we clone a
public project of roughly the size an MBO student works on and build the
agent messages out of its actual files.

No Firda code and no student data ever enters this harness -- everything is
either public or generated.

Two properties matter for the experiment:

*  A **shared skeleton**: students in one class work on the same assignment,
   so they share the system prompt and most of the project scaffolding. That
   produces cache hits across students and is the single biggest lever on
   memory pressure. ``shared_fraction`` controls how much of each student's
   context comes from that identical block.
*  **Byte-identical ordering**: the shared block is emitted first and in a
   fixed order, otherwise it is not a prefix and the cache never sees it.

A corpus holds one or more **groups**. A group is one assignment: the students
working on it share its skeleton, and students on a different assignment share
nothing with them beyond the system prompt. One group reproduces the original
single-assignment behaviour exactly; several groups model a class where some
students are on a small web app and others on a Unity project, which is both
more realistic and harder on the cache. Which group a student works in is
decided by their work profile, not here.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
from dataclasses import dataclass
from typing import Iterable, Sequence

from .util import log

# Public example applications of roughly MBO project size. The RealWorld
# projects are deliberate: same app, three stacks, all MIT/permissive, all a
# few thousand lines -- exactly the scale of a school assignment.
DEFAULT_PROJECTS: list[dict[str, str]] = [
    {
        "name": "django-realworld",
        "url": "https://github.com/gothinkster/django-realworld-example-app",
        "extensions": ".py,.html,.txt,.md,.cfg",
    },
    {
        "name": "react-realworld",
        "url": "https://github.com/gothinkster/react-redux-realworld-example-app",
        "extensions": ".js,.jsx,.json,.css,.md",
    },
    {
        "name": "laravel-realworld",
        "url": "https://github.com/gothinkster/laravel-realworld-example-app",
        "extensions": ".php,.blade.php,.json,.md",
    },
    {
        # A Unity game project: the files are several times larger than a
        # RealWorld controller, which is the whole reason it is here. Unity
        # repositories are mostly binary art, so only the scripts are checked
        # out -- a full clone is gigabytes of textures nobody reads.
        "name": "unity-fpssample",
        "url": "https://github.com/Unity-Technologies/FPSSample",
        "extensions": ".cs",
        "sparse_paths": "/Assets/Scripts/**/*.cs",
    },
]

SKIP_DIRECTORIES = {".git", "node_modules", "vendor", "dist", "build", "__pycache__",
                    ".venv", "venv", ".idea", ".vscode", "coverage", ".next"}

MIN_FILE_BYTES = 200
MAX_FILE_BYTES = 24_000


@dataclass
class SourceFile:
    path: str
    content: str

    @property
    def characters(self) -> int:
        return len(self.content)


class CorpusGroup:
    """One assignment: a file set with its own shared skeleton.

    Students inside a group share the skeleton byte-for-byte, which is what
    the prefix cache lives on. Students in different groups share nothing
    below the system prompt -- that is the point of having groups.
    """

    def __init__(self, name: str, files: Sequence[SourceFile], shared_count: int) -> None:
        if not files:
            raise ValueError(f"corpus group {name!r} is empty")
        self.name = name
        # Deterministic order: the shared block must be byte-identical and in
        # the same sequence for every student, or prefix caching sees nothing.
        self.files = sorted(files, key=lambda f: f.path)
        shared_count = max(1, min(shared_count, len(self.files) - 1 if len(self.files) > 1 else 1))
        self.shared_files = self.files[:shared_count]
        self.private_files = self.files[shared_count:] or self.files

    @property
    def total_characters(self) -> int:
        return sum(f.characters for f in self.files)

    def student_files(self, student_index: int, count: int) -> list[SourceFile]:
        """A stable, student-specific selection from the non-shared pool."""
        rng = random.Random(hashlib.sha256(f"{self.name}:{student_index}".encode()).digest()[:8])
        pool = list(self.private_files)
        rng.shuffle(pool)
        if count <= len(pool):
            return pool[:count]
        # Small corpus: repeat, but rotate so different students still differ.
        out: list[SourceFile] = []
        while len(out) < count:
            offset = len(out) % max(1, len(pool))
            out.extend(pool[offset:] + pool[:offset])
        return out[:count]

    def describe(self) -> dict:
        return {
            "name": self.name,
            "files": len(self.files),
            "shared_files": len(self.shared_files),
            "private_files": len(self.private_files),
            "total_characters": self.total_characters,
            "median_file_tokens_estimate": self._median_file_tokens(),
        }

    def _median_file_tokens(self) -> int:
        """Rough median file size, for the record.

        How big the files are is the difference between "read a controller"
        and "read a Unity system", and that difference drives how fast a
        context window fills. It belongs in environment.json.
        """
        if not self.files:
            return 0
        sizes = sorted(f.characters for f in self.files)
        return int(round(sizes[len(sizes) // 2] / 4.65))


class CodeCorpus:
    """One or more assignments the simulated class works on.

    A single group behaves exactly like the original flat corpus, so runs made
    before groups existed stay reproducible.
    """

    def __init__(self, name: str, groups: Sequence[CorpusGroup] | Sequence[SourceFile],
                 shared_count: int | None = None) -> None:
        # Two call shapes, deliberately: a list of groups, or the original
        # (name, files, shared_count) which makes one group. Keeping the second
        # means a flat corpus is not a special case anywhere else.
        if shared_count is not None:
            groups = [CorpusGroup(name, groups, shared_count)]  # type: ignore[arg-type]
        if not groups:
            raise ValueError("corpus has no groups")
        self.name = name
        self.groups = {g.name: g for g in groups}   # type: ignore[union-attr]
        self.default_group = groups[0].name         # type: ignore[union-attr]

    def group(self, name: str | None = None) -> CorpusGroup:
        """The named group, falling back to the default.

        An unknown name is a configuration mistake worth failing on: silently
        handing back the wrong assignment would show up as an inexplicable
        cache-hit rate three hours into a run.
        """
        if name is None:
            return self.groups[self.default_group]
        try:
            return self.groups[name]
        except KeyError:
            raise RuntimeError(
                f"unknown corpus group {name!r}; known: "
                + ", ".join(sorted(self.groups))) from None

    # ------------------------------------------------- flat-corpus interface

    @property
    def files(self) -> list[SourceFile]:
        return [f for g in self.groups.values() for f in g.files]

    @property
    def shared_files(self) -> list[SourceFile]:
        return self.group().shared_files

    @property
    def private_files(self) -> list[SourceFile]:
        return self.group().private_files

    @property
    def total_characters(self) -> int:
        return sum(g.total_characters for g in self.groups.values())

    def student_files(self, student_index: int, count: int,
                      group: str | None = None) -> list[SourceFile]:
        return self.group(group).student_files(student_index, count)

    def describe(self) -> dict:
        out = self.group().describe()
        out["name"] = self.name
        out["files"] = len(self.files)
        out["total_characters"] = self.total_characters
        if len(self.groups) > 1:
            out["groups"] = {name: g.describe() for name, g in self.groups.items()}
        return out


def _iter_source_files(root: str, extensions: Sequence[str]) -> Iterable[SourceFile]:
    normalized = tuple(e.strip().lower() for e in extensions if e.strip())
    for directory, subdirs, filenames in os.walk(root):
        subdirs[:] = [d for d in subdirs if d not in SKIP_DIRECTORIES and not d.startswith(".")]
        for filename in sorted(filenames):
            if normalized and not filename.lower().endswith(normalized):
                continue
            full = os.path.join(directory, filename)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
                continue
            try:
                with open(full, "r", encoding="utf-8") as handle:
                    content = handle.read()
            except (UnicodeDecodeError, OSError):
                continue
            if "\x00" in content:
                continue
            yield SourceFile(os.path.relpath(full, root).replace(os.sep, "/"), content)


def fetch_project(project: dict, cache_dir: str, quiet: bool = False) -> str:
    """Shallow-clone a public project. Returns the checkout directory.

    A project with ``sparse_paths`` is fetched without blobs and then narrowed
    to those paths. That is not an optimisation: a Unity repository is mostly
    textures, meshes and audio, and cloning it whole would cost gigabytes and
    minutes for source files that add up to a few megabytes.
    """
    target = os.path.join(cache_dir, project["name"])
    if os.path.isdir(os.path.join(target, ".git")):
        if not quiet:
            log(f"corpus: {project['name']} already present at {target}")
        return target
    os.makedirs(cache_dir, exist_ok=True)
    sparse = project.get("sparse_paths")
    if not quiet:
        log(f"corpus: cloning {project['url']}"
            + (f" (alleen {sparse})" if sparse else ""))
    command = ["git", "clone", "--depth", "1", "--quiet"]
    if sparse:
        command += ["--filter=blob:none", "--sparse"]
    command += [project["url"], target]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        raise RuntimeError(f"git clone of {project['url']} failed: {result.stderr.strip()[:300]}")
    if sparse:
        patterns = [p.strip() for p in str(sparse).split(",") if p.strip()]
        narrowed = subprocess.run(
            ["git", "-C", target, "sparse-checkout", "set", "--no-cone", *patterns],
            capture_output=True, text=True, timeout=900)
        if narrowed.returncode != 0:
            raise RuntimeError(f"sparse-checkout of {project['url']} failed: "
                               f"{narrowed.stderr.strip()[:300]}")
    return target


def _build_group(name: str, spec: dict, cache_dir: str, projects: Sequence[dict],
                 quiet: bool) -> CorpusGroup:
    local_path = spec.get("local_path")
    if local_path:
        extensions = [e.strip() for e in
                      str(spec.get("extensions", ".py,.js,.php,.md")).split(",")]
        files = list(_iter_source_files(local_path, extensions))
        if not files:
            raise RuntimeError(f"no usable source files under {local_path}")
        return CorpusGroup(name, files, _shared_count(spec, len(files)))

    wanted = spec.get("project", DEFAULT_PROJECTS[0]["name"])
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    files: list[SourceFile] = []
    for project_name in names:
        project = next((p for p in projects if p["name"] == project_name), None)
        if project is None:
            raise RuntimeError(f"unknown corpus project {project_name!r}; known: "
                               + ", ".join(p["name"] for p in projects))
        checkout = fetch_project(project, cache_dir, quiet=quiet)
        extensions = [e.strip() for e in project.get("extensions", ".py,.md").split(",")]
        found = list(_iter_source_files(checkout, extensions))
        if not found:
            raise RuntimeError(f"no usable source files in {checkout}")
        # Namespace the paths so two projects cannot collide on e.g. README.md.
        files.extend(SourceFile(f"{project_name}/{f.path}", f.content) for f in found)
    if not files:
        raise RuntimeError(f"corpus group {name!r} is empty")
    return CorpusGroup(name, files, _shared_count(spec, len(files)))


def load_corpus(config: dict, quiet: bool = False) -> CodeCorpus:
    """Build a corpus from config.

    Two shapes are accepted. The flat one -- ``project``/``local_path`` at the
    top level -- makes a single group and behaves exactly as it always did.
    The grouped one, ``corpus.groups``, makes one group per assignment:

        "groups": {
          "web":   {"project": ["django-realworld", "react-realworld"]},
          "unity": {"project": ["unity-fpssample"]}
        }

    Work profiles then say which group a student works in. Settings given at
    the top level (``extensions``, ``shared_skeleton_fraction``) are defaults
    that a group can override.
    """
    cache_dir = config.get("cache_dir", "corpus_cache")
    projects = config.get("projects") or DEFAULT_PROJECTS
    groups_config = config.get("groups")

    if not groups_config:
        group = _build_group(_flat_name(config, projects), config, cache_dir, projects, quiet)
        return CodeCorpus(group.name, [group])

    groups: list[CorpusGroup] = []
    for name, spec in groups_config.items():
        merged = {k: v for k, v in config.items() if k not in ("groups", "projects")}
        merged.update(spec or {})
        groups.append(_build_group(name, merged, cache_dir, projects, quiet))
    return CodeCorpus("+".join(g.name for g in groups), groups)


def _flat_name(config: dict, projects: Sequence[dict]) -> str:
    local_path = config.get("local_path")
    if local_path:
        return os.path.basename(local_path.rstrip("/")) or "local"
    wanted = config.get("project", DEFAULT_PROJECTS[0]["name"])
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    return "+".join(names)


def _shared_count(config: dict, total: int) -> int:
    """How many files form the shared project skeleton.

    An explicit count wins; otherwise a fraction of the corpus, so the split
    stays sensible whichever project you point it at.
    """
    explicit = config.get("shared_skeleton_files")
    if explicit:
        return int(explicit)
    fraction = float(config.get("shared_skeleton_fraction", 0.4))
    return max(1, min(total - 1, int(round(total * fraction))))
