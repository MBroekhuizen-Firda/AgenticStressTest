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


class CodeCorpus:
    """An indexed checkout, split into a shared skeleton and per-student files."""

    def __init__(self, name: str, files: Sequence[SourceFile], shared_count: int) -> None:
        if not files:
            raise ValueError("corpus is empty")
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
        }


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
    """Shallow-clone a public project. Returns the checkout directory."""
    target = os.path.join(cache_dir, project["name"])
    if os.path.isdir(os.path.join(target, ".git")):
        if not quiet:
            log(f"corpus: {project['name']} already present at {target}")
        return target
    os.makedirs(cache_dir, exist_ok=True)
    if not quiet:
        log(f"corpus: cloning {project['url']}")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", project["url"], target],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git clone of {project['url']} failed: {result.stderr.strip()[:300]}")
    return target


def load_corpus(config: dict, quiet: bool = False) -> CodeCorpus:
    """Build a corpus from config.

    ``corpus.local_path`` wins if set (useful on a machine without outbound
    git access); otherwise the configured project or projects are cloned into
    ``corpus.cache_dir``. Combining the three RealWorld example apps gives a
    corpus of a few thousand lines across Python, JavaScript and PHP, which
    is what an MBO class actually has in front of it.
    """
    cache_dir = config.get("cache_dir", "corpus_cache")

    local_path = config.get("local_path")
    if local_path:
        extensions = [e.strip() for e in
                      str(config.get("extensions", ".py,.js,.php,.md")).split(",")]
        files = list(_iter_source_files(local_path, extensions))
        if not files:
            raise RuntimeError(f"no usable source files under {local_path}")
        name = os.path.basename(local_path.rstrip("/")) or "local"
        return CodeCorpus(name, files, _shared_count(config, len(files)))

    wanted = config.get("project", DEFAULT_PROJECTS[0]["name"])
    names = [wanted] if isinstance(wanted, str) else list(wanted)
    projects = config.get("projects") or DEFAULT_PROJECTS

    files: list[SourceFile] = []
    for name in names:
        project = next((p for p in projects if p["name"] == name), None)
        if project is None:
            raise RuntimeError(f"unknown corpus project {name!r}; known: "
                               + ", ".join(p["name"] for p in projects))
        checkout = fetch_project(project, cache_dir, quiet=quiet)
        extensions = [e.strip() for e in project.get("extensions", ".py,.md").split(",")]
        found = list(_iter_source_files(checkout, extensions))
        if not found:
            raise RuntimeError(f"no usable source files in {checkout}")
        # Namespace the paths so two projects cannot collide on e.g. README.md.
        files.extend(SourceFile(f"{name}/{f.path}", f.content) for f in found)

    if not files:
        raise RuntimeError("corpus is empty")
    return CodeCorpus("+".join(names), files, _shared_count(config, len(files)))


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
