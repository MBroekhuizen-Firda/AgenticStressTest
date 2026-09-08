"""Optional PNG versions of the charts, when matplotlib happens to be there.

The SVG charts are always written and are the canonical output. This module
exists because a PNG pastes into a slide deck more easily, and a rented GPU
image often has matplotlib installed already.
"""

from __future__ import annotations

from typing import Sequence

from .svgplot import COLOURS, GRADE_FILL


def available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def heatmap(path: str, *, title: str, row_labels: Sequence[str],
            column_labels: Sequence[str], grades: Sequence[Sequence[str]],
            annotations: Sequence[Sequence[str]], subtitle: str = "") -> None:
    plt = _plt()
    from matplotlib.patches import Rectangle
    figure, axes = plt.subplots(figsize=(2.2 * len(column_labels) + 3,
                                         1.0 * len(row_labels) + 2.2))
    axes.set_xlim(0, len(column_labels))
    axes.set_ylim(0, len(row_labels))
    for row in range(len(row_labels)):
        for column in range(len(column_labels)):
            grade = grades[row][column] if column < len(grades[row]) else "onbekend"
            axes.add_patch(Rectangle((column, len(row_labels) - row - 1), 1, 1,
                                     facecolor=GRADE_FILL.get(grade, "#a0aec0"),
                                     edgecolor="white", linewidth=2))
            text = annotations[row][column] if column < len(annotations[row]) else ""
            axes.text(column + 0.5, len(row_labels) - row - 0.5, text,
                      ha="center", va="center", color="white", fontsize=9)
    axes.set_xticks([i + 0.5 for i in range(len(column_labels))])
    axes.set_xticklabels(column_labels)
    axes.set_yticks([len(row_labels) - i - 0.5 for i in range(len(row_labels))])
    axes.set_yticklabels(row_labels)
    axes.set_title(title + (f"\n{subtitle}" if subtitle else ""), fontsize=11)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def line_chart(path: str, *, title: str, x_label: str, y_label: str,
               series: Sequence[dict], hlines: Sequence[dict] = ()) -> None:
    plt = _plt()
    figure, axes = plt.subplots(figsize=(9, 5.2))
    for index, item in enumerate(series):
        axes.plot(item["x"], item["y"], marker="o",
                  color=COLOURS[index % len(COLOURS)], label=item["name"])
    for line in hlines:
        axes.axhline(line["y"], linestyle="--", linewidth=1,
                     color=line.get("colour", "#c53030"), label=line.get("label"))
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.grid(alpha=0.3)
    axes.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def bar_chart(path: str, *, title: str, labels: Sequence[str],
              values: Sequence[float], y_label: str,
              colours: Sequence[str] | None = None) -> None:
    plt = _plt()
    figure, axes = plt.subplots(figsize=(9, 5.2))
    axes.bar(range(len(labels)), values,
             color=list(colours) if colours else COLOURS[0])
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    axes.set_ylabel(y_label)
    axes.set_title(title)
    axes.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
