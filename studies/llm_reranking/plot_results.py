"""Reranker training figures: per-run loss and dev-metric curves, a run overlay, and a best-score bar chart.

Run spec: name:checkpoint_dir:dev_summary_json (repeatable). Colors follow run order.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import typer  # noqa: E402

app = typer.Typer(add_completion=False)

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
REFERENCES = {
    "splade": "SPLADE-v3 (no rerank)",
    "ce:cross-encoder/ms-marco-MiniLM-L6-v2": "MiniLM-L6 cross-encoder",
    "llm:Qwen/Qwen3-Reranker-0.6B": "Qwen3-Reranker-0.6B",
    "llm:Qwen/Qwen3-Reranker-4B": "Qwen3-Reranker-4B",
}
CURVE_REFERENCES = ("splade", "llm:Qwen/Qwen3-Reranker-0.6B")  # kept few so per-run plots stay readable

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans"})


def style(axis, title: str, xlabel: str, ylabel: str):
    axis.set_facecolor(SURFACE)
    axis.set_title(title, loc="left", color=INK)
    axis.set_xlabel(xlabel, color=INK_SECONDARY)
    axis.set_ylabel(ylabel, color=INK_SECONDARY)
    axis.grid(color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.tick_params(colors=MUTED)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(MUTED)


def new_figure(width=4.6, height=3.0):
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    return figure, axis


def save(figure, path: Path):
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def describe(config: dict) -> str:
    return f"{config['loss_type']}, lr {config['learning_rate']:g}, {Path(config['data']).stem}, {config['epochs']:g} ep"


def load_run(spec: str, index: int) -> dict:
    name, checkpoint_dir, summary_path = spec.split(":", 2)
    run_dir = Path(checkpoint_dir)
    config = json.loads((run_dir / "train_config.json").read_text())
    summary = json.loads(Path(summary_path).read_text()) if Path(summary_path).exists() else {}
    dev_points = []
    for key, scores in summary.items():
        if key.startswith("lora:"):
            folder = Path(key[len("lora:"):]).name
            step = int(folder.split("-")[1]) if folder.startswith("step-") else config["total_steps"]
            dev_points.append((step, scores))
    base_key = f"llm:{config['base_model']}"
    if base_key in summary:  # step 0: an untrained LoRA adapter equals the zero-shot base model
        dev_points.append((0, summary[base_key]))
    return {"name": name, "color": SERIES_COLORS[index], "config": config, "summary": summary,
            "train_log": read_jsonl(run_dir / "train_log.jsonl"), "dev_log": read_jsonl(run_dir / "dev_log.jsonl"),
            "dev_points": sorted(dev_points, key=lambda point: point[0])}


def smooth(values: list[float], weight: float = 0.6) -> list[float]:
    """Exponential moving average, as in TensorBoard."""
    smoothed, last = [], values[0]
    for value in values:
        last = weight * last + (1 - weight) * value
        smoothed.append(last)
    return smoothed


def plot_run_loss(run: dict, path: Path):
    figure, axis = new_figure()
    steps = [entry["step"] for entry in run["train_log"]]
    losses = [entry["loss"] for entry in run["train_log"]]
    axis.plot(steps, losses, color=run["color"], linewidth=1, alpha=0.3)
    axis.plot(steps, smooth(losses), color=run["color"], linewidth=2, label="train (smoothed)")
    if run["dev_log"]:
        axis.plot([e["step"] for e in run["dev_log"]], [e["dev_loss"] for e in run["dev_log"]], color=INK_SECONDARY,
                  linewidth=1.6, linestyle=(0, (4, 2)), marker="o", markersize=4.5, markeredgecolor=SURFACE,
                  label="validation")
    style(axis, f"{run['name']}: {run['config']['loss_type']} loss", "training step", "loss")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY)
    save(figure, path)


def plot_run_metric(run: dict, path: Path, metric: str, display: str):
    figure, axis = new_figure()
    for key, dashes in zip(CURVE_REFERENCES, [(0, (1, 2)), (0, (5, 3))]):
        if key in run["summary"]:
            value = run["summary"][key][metric]
            axis.axhline(value, color=MUTED, linewidth=1.2, linestyle=dashes, label=f"{REFERENCES[key]} {value:.3f}")
    steps = [step for step, _ in run["dev_points"]]
    values = [scores[metric] for _, scores in run["dev_points"]]
    axis.plot(steps, values, color=run["color"], linewidth=2, marker="o", markersize=5, markeredgecolor=SURFACE)
    best = max(range(len(values)), key=values.__getitem__)
    axis.plot(steps[best], values[best], marker="*", markersize=13, color=run["color"], markeredgecolor=INK,
              linestyle="none", label=f"best {values[best]:.3f} at step {steps[best]}")
    style(axis, f"{run['name']}: dev {display}", "training step", display)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower right", fontsize=7)
    save(figure, path)


def plot_overlay(runs: list[dict], path: Path, metric: str, display: str):
    figure, axis = new_figure(7.2, 3.8)
    for key in CURVE_REFERENCES:
        values = [run["summary"][key][metric] for run in runs if key in run["summary"]]
        if values:
            axis.axhline(values[0], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            axis.annotate(f"{REFERENCES[key]} {values[0]:.3f}", (1, values[0]), xycoords=("axes fraction", "data"),
                          xytext=(-3, 3), textcoords="offset points", ha="right", fontsize=7, color=INK_SECONDARY)
    for run in runs:
        if run["dev_points"]:
            axis.plot([s for s, _ in run["dev_points"]], [v[metric] for _, v in run["dev_points"]], color=run["color"],
                      linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                      label=f"{run['name']}: {describe(run['config'])}")
    style(axis, f"Dev {display} by checkpoint, all runs (rag24-dev, 89 topics)", "training step", display)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(figure, path)


def plot_best_bars(runs: list[dict], path: Path, metric: str, display: str):
    rows = []
    for run in runs:
        if run["dev_points"]:
            step, scores = max(run["dev_points"], key=lambda point: point[1][metric])
            rows.append((f"Qwen3.5-0.8B + LoRA {run['name']} (step {step})", scores[metric], run["color"]))
    summary = next((run["summary"] for run in runs if run["summary"]), {})
    zero_shot = f"llm:{runs[0]['config']['base_model']}"
    if zero_shot in summary:
        rows.append(("Qwen3.5-0.8B zero-shot", summary[zero_shot][metric], "#c3c2b7"))
    for key, label in REFERENCES.items():
        if key in summary and key != "splade":
            rows.append((label, summary[key][metric], "#c3c2b7"))
    rows.sort(key=lambda row: row[1])
    figure, axis = new_figure(6.4, 0.42 * len(rows) + 1.2)
    labels = [label for label, _, _ in rows]
    for position, (_, value, color) in enumerate(rows):  # dot plot: the axis is zoomed, so no bars
        axis.plot(value, position, marker="o", markersize=9, color=color, markeredgecolor=INK if color != "#c3c2b7"
                  else MUTED, markeredgewidth=1)
        axis.text(value + 0.004, position, f"{value:.3f}", va="center", fontsize=8, color=INK)
    axis.set_yticks(range(len(rows)), labels)
    if "splade" in summary:
        value = summary["splade"][metric]
        axis.axvline(value, color=INK_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        axis.annotate(f"SPLADE-v3, no rerank {value:.3f}", (value, 0), xycoords=("data", "axes fraction"),
                      xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    low = min(value for _, value, _ in rows)
    axis.set_xlim(max(0, low - 0.02), max(value for _, value, _ in rows) + 0.03)
    axis.set_ylim(-0.8, len(rows) - 0.4)
    style(axis, f"Best dev {display} (rag24-dev, SPLADE-v3 top-100)", display, "")
    axis.grid(axis="y", visible=False)
    save(figure, path)


@app.command()
def main(
    run: list[str] = typer.Option(..., help="name:checkpoint_dir:dev_summary_json, repeatable"),
    out_dir: Path = typer.Option(Path("studies/llm_reranking/figures")),
    metric: str = "ndcg@30",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(spec, index) for index, spec in enumerate(run)]
    display = metric.replace("ndcg", "nDCG")
    for item in runs:
        plot_run_loss(item, out_dir / f"{item['name']}_loss.png")
        if item["dev_points"]:
            plot_run_metric(item, out_dir / f"{item['name']}_dev.png", metric, display)
    plot_overlay(runs, out_dir / "dev_all_runs.png", metric, display)
    plot_best_bars(runs, out_dir / "dev_best.png", metric, display)
    typer.echo(f"figures -> {out_dir}")


if __name__ == "__main__":
    app()
