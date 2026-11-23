"""Cost–quality figure for cascades: nDCG@30 vs seconds per query, dev Pareto front highlighted, RAG25 if run."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from studies.llm_reranking.plot_results import INK, INK_SECONDARY, MUTED, REFERENCE_FILL, SERIES_COLORS, new_figure, save, style

app = typer.Typer(add_completion=False)
LABELS = {
    "ours_depth100": "ours, top-100",
    "ours_depth500": "ours, top-500",
    "depth100_top30_listwise": "top-100, listwise@30",
    "depth100_top30_fusion": "top-100, fusion@30",
    "depth100_top100_listwise": "top-100, listwise@100",
    "depth100_top100_fusion": "top-100, fusion@100",
    "depth500_top30_listwise": "top-500, listwise@30",
    "depth500_top30_fusion": "top-500, fusion@30",
    "depth500_top100_listwise": "top-500, listwise@100",
    "depth500_top100_fusion": "top-500, fusion@100",
}


def pareto_front(rows: dict[str, dict], metric: str) -> list[str]:
    """Systems no other system beats on both cost and score, by increasing cost."""
    front, best = [], float("-inf")
    for name, row in sorted(rows.items(), key=lambda item: (item[1]["s/query"], -item[1][metric])):
        if row[metric] > best:
            front.append(name)
            best = row[metric]
    return front


@app.command()
def main(
    cascade_dir: Path = Path("outputs/llm_reranking"),
    out: Path = Path("studies/llm_reranking/figures/cascade_frontier.png"),
    metric: str = "ndcg@30",
):
    dev = json.loads((cascade_dir / "rag24-dev" / "cascade" / "summary.json").read_text())
    test_path = cascade_dir / "rag25" / "cascade" / "summary.json"
    test = json.loads(test_path.read_text()) if test_path.exists() else {}
    front = pareto_front(dev, metric)
    display = metric.replace("ndcg", "nDCG")
    figure, axis = new_figure(6.8, 4.2)
    for name, row in dev.items():
        on_front = name in front
        axis.plot(row["s/query"], row[metric], marker="o", markersize=8, linestyle="none",
                  color=SERIES_COLORS[0] if on_front else REFERENCE_FILL, markeredgecolor=INK if on_front else MUTED)
        offset = (6, 4) if name == "depth500_top30_fusion" else (6, -3)  # keeps the two top-500@30 labels apart
        axis.annotate(LABELS.get(name, name), (row["s/query"], row[metric]), xytext=offset, textcoords="offset points",
                      fontsize=7, color=INK if on_front else MUTED)
    axis.plot([dev[name]["s/query"] for name in front], [dev[name][metric] for name in front], color=SERIES_COLORS[0],
              linewidth=1.5, label="dev Pareto front (rag24-dev)")
    tested = [name for name in front if name in test]
    if tested:
        axis.plot([test[name]["s/query"] for name in tested], [test[name][metric] for name in tested],
                  color=SERIES_COLORS[1], linewidth=1.5, marker="s", markersize=7, markeredgecolor=INK,
                  label="same systems on RAG25")
    style(axis, f"Cascades: {display} vs cost (SPLADE-v3 first stage)", "seconds per query (model loading excluded)", display)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower right", fontsize=8)
    save(figure, out)
    typer.echo(f"front: {front} -> {out}")


if __name__ == "__main__":
    app()
