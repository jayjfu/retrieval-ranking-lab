"""Reranker figures: per-run loss and dev curves, a run overlay, best dev scores per section, and the RAG25 test.

Runs are checkpoint directories; the directory name is the run name.
Dev summaries are read from <dev_dir>/<run name>/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
import typer  # noqa: E402

app = typer.Typer(add_completion=False)

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK_SECONDARY, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
REFERENCE_FILL, OVERVIEW_FILL = "#c3c2b7", "#fcfcfb"
REFERENCES = {
    "splade": "SPLADE-v3 (no rerank)",
    "ce:cross-encoder/ms-marco-MiniLM-L6-v2": "MiniLM-L6 cross-encoder",
    "llm:Qwen/Qwen3.5-0.8B": "Qwen3.5-0.8B zero-shot",
    "llm:Qwen/Qwen3-Reranker-0.6B": "Qwen3-Reranker-0.6B",
    "llm:Qwen/Qwen3-Reranker-4B": "Qwen3-Reranker-4B",
}
CURVE_REFERENCES = ("splade", "llm:Qwen/Qwen3-Reranker-0.6B")  # few, so per-run plots stay readable
OVERVIEW_LABELS = {
    "[overview] best: 4method_merge": "overview: best run (4method_merge)",
    "[overview] rag25_test_rankqwen3": "overview: RankQwen3-32B (organizers)",
    "[overview] splade-v3-arctic-l": "overview: SPLADE-v3 + Arctic-L fusion",
    "[overview] rag25_test_splade-v3": "overview: SPLADE-v3",
    "[overview] rag25_test_arctic-l": "overview: Arctic-Embed-L",
    "[overview] anserini_bm25_top100": "overview: BM25",
}

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


def new_figure(width: float = 4.6, height: float = 3.0):
    figure, axis = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    return figure, axis


def save(figure, path: Path):
    figure.tight_layout()
    figure.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def load_run(run_dir: Path, dev_dir: Path, color: str) -> dict:
    config = json.loads((run_dir / "train_config.json").read_text())
    summary_path = dev_dir / run_dir.name / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    dev_points = []
    for key, scores in summary.items():
        if key.startswith("lora:"):
            folder = Path(key[len("lora:"):]).name
            step = int(folder.split("-")[1]) if folder.startswith("step-") else config["total_steps"]
            dev_points.append((step, scores))
    base_key = f"llm:{config['base_model']}"
    if base_key in summary:  # step 0: an untrained LoRA adapter equals the zero-shot base model
        dev_points.append((0, summary[base_key]))
    return {"name": run_dir.name, "color": color, "config": config, "summary": summary,
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
        axis.plot([entry["step"] for entry in run["dev_log"]], [entry["dev_loss"] for entry in run["dev_log"]],
                  color=INK_SECONDARY, linewidth=1.6, linestyle=(0, (4, 2)), marker="o", markersize=4.5,
                  markeredgecolor=SURFACE, label="validation")
    style(axis, f"{run['name']}: loss", "training step", f"{run['config']['loss_type']} loss")
    axis.set_ylim(bottom=0)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY)
    save(figure, path)


def plot_run_metric(run: dict, references: dict, path: Path, metric: str, display: str):
    figure, axis = new_figure()
    for key, dashes in zip(CURVE_REFERENCES, [(0, (1, 2)), (0, (5, 3))]):
        if key in references:
            value = references[key][metric]
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


def load_references(dev_dir: Path) -> dict:
    """Reference rerankers scored on dev, kept apart from training runs."""
    path = dev_dir / "references" / "summary.json"
    return json.loads(path.read_text()) if path.exists() else {}


def plot_overlay(runs: list[dict], references: dict, path: Path, metric: str, display: str):
    figure, axis = new_figure(7.2, 3.8)
    for key in CURVE_REFERENCES:
        if key in references:
            value = references[key][metric]
            axis.axhline(value, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            axis.annotate(f"{REFERENCES[key]} {value:.3f}", (1, value), xycoords=("axes fraction", "data"),
                          xytext=(-3, 3), textcoords="offset points", ha="right", fontsize=7, color=INK_SECONDARY)
    for run in runs:
        trained = [(step, scores) for step, scores in run["dev_points"] if step > 0]  # zero-shot start would squash the y-axis
        if trained:
            axis.plot([step for step, _ in trained], [scores[metric] for _, scores in trained],
                      color=run["color"], linewidth=2, marker="o", markersize=4.5, markeredgecolor=SURFACE,
                      label=run["name"])
    style(axis, f"Dev {display} by checkpoint (rag24-dev, 89 topics, SPLADE-v3 top-100)", "training step", display)
    axis.legend(frameon=False, labelcolor=INK_SECONDARY, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(figure, path)


OURS_GROUP, ZERO_SHOT_GROUP, OVERVIEW_GROUP = ("Ours", "Off-the-shelf rerankers, zero-shot (our runs)",
                                              "TREC 2025 RAG overview runs")


def dot_plot(groups: dict[str, list[tuple[str, float, str, str]]], path: Path, title: str, display: str,
             baseline: tuple[str, float] | None = None):
    """One dot plot sorted by value; a legend names each group. Rows are (label, value, fill, edge).
    The axis is zoomed, so no bars."""
    rows = sorted(((group, *row) for group, members in groups.items() for row in members), key=lambda row: row[2])
    figure, axis = new_figure(6.8, 0.4 * len(rows) + 1.6)
    handles = {name: [] for name in groups}
    for position, (group, _, value, fill, edge) in enumerate(rows):
        (marker,) = axis.plot(value, position, marker="o", markersize=9, color=fill, markeredgecolor=edge,
                              markeredgewidth=1.2, linestyle="none")
        if fill not in [handle.get_color() for handle in handles[group]]:
            handles[group].append(marker)
        axis.annotate(f"{value:.3f}", (value, position), xytext=(8, 0), textcoords="offset points", va="center",
                      fontsize=8, color=INK)
    axis.set_yticks(range(len(rows)), [row[1] for row in rows])
    values = [row[2] for row in rows]
    if baseline:
        label, value = baseline
        values.append(value)
        axis.axvline(value, color=INK_SECONDARY, linewidth=1, linestyle=(0, (1, 2)))
        axis.annotate(f"{label} {value:.3f}", (value, 0), xycoords=("data", "axes fraction"), xytext=(3, 3),
                      textcoords="offset points", fontsize=7, color=INK_SECONDARY)
    axis.set_xlim(min(values) - 0.03, max(values) + 0.035)
    axis.set_ylim(-0.8, len(rows) - 0.4)
    style(axis, title, display, "")
    axis.grid(axis="y", visible=False)
    names = [name for name in groups if handles[name]]
    axis.legend([tuple(handles[name]) for name in names], names, handler_map={tuple: HandlerTuple(ndivide=None)},
                frameon=False, labelcolor=INK_SECONDARY, loc="upper center", bbox_to_anchor=(0.4, -0.1 - 4 / (len(rows) * 10 + 16)),
                ncol=1, fontsize=8, handlelength=4)
    save(figure, path)


PLAIN = "listwise_lr1e-4_pos32_epoch1_cosine_clip1"
DISTILLED = f"{PLAIN}_distill_alpha0_tau1"
AUGMENTED = f"{DISTILLED}_group16_unjudged8"
TEACHER = "ensemble_9b-yes_no_35b-a3b-graded"
SYSTEMS = {  # dev-selected system of each section: id -> (label, group, source)
    PLAIN: ("Qwen3.5-0.8B + LoRA (ours, fine-tuned)", "ours", "run"),
    DISTILLED: ("Qwen3.5-0.8B + LoRA, distilled (ours)", "ours", "run"),
    AUGMENTED: ("Qwen3.5-0.8B + LoRA, distilled + augmented (ours)", "ours", "run"),
    TEACHER: ("Qwen3.5-9B yes/no + 35B-A3B graded (teacher)", "zero_shot", "teacher"),
    "rrf_ours+teacher+splade": ("ours + teacher + SPLADE-v3 (RRF)", "ours", "fusion"),
    "qwen35b_ours": ("listwise Qwen3.5-35B-A3B, after ours", "ours", "listwise"),
    "rrf_ours+list35bspl+qwen9b": ("ours + listwise Qwen3.5-35B-A3B + Qwen3.5-9B (RRF)", "ours", "fusion"),
}
DEV_STAGES = [  # each figure adds the next section's systems to the previous one
    ("dev_best.png", "fine-tuning", [PLAIN]),
    ("dev_best_distillation.png", "augmentation", [TEACHER, DISTILLED, AUGMENTED]),
    ("dev_best_fusion.png", "fusion", ["rrf_ours+teacher+splade"]),
    ("dev_best_listwise.png", "listwise reranking", ["qwen35b_ours", "rrf_ours+list35bspl+qwen9b"]),
]


def system_score(system: str, split_dir: Path, checkpoint_root: Path, metric: str) -> tuple[float, int | None]:
    """Score of a system on one split (dev or RAG25), and the checkpoint step for fine-tuned runs on dev."""
    source = SYSTEMS[system][2]
    if source == "run" and split_dir.name == "rag24-dev":
        step, scores = max(load_run(checkpoint_root / system, split_dir, SERIES_COLORS[0])["dev_points"],
                           key=lambda point: point[1][metric])
        return scores[metric], step
    if source == "run":
        summary = json.loads((split_dir / "splade@100" / "summary.json").read_text())
        key = next(key for key in summary if key.startswith(f"lora:{checkpoint_root / system}/"))
        return summary[key][metric], None
    if source == "teacher":
        from retrieval_ranking_lab.data.trec_rag import load_topic_set
        from retrieval_ranking_lab.eval.metrics import evaluate
        from retrieval_ranking_lab.types import read_run
        run = read_run(split_dir.parent / "teachers" / split_dir.name / f"{system}.trec")
        return evaluate(run, load_topic_set(split_dir.name).qrels, [metric])[metric], None
    return json.loads((split_dir / source / "summary.json").read_text())[system][metric], None


def system_row(system: str, value: float, step: int | None) -> tuple[str, tuple[str, float, str, str]]:
    label, group, _ = SYSTEMS[system]
    label = f"{label}, step {step}" if step is not None else label
    if group == "ours":
        return OURS_GROUP, (label, value, SERIES_COLORS[0], INK)
    return ZERO_SHOT_GROUP, (label, value, REFERENCE_FILL, MUTED)


def plot_dev_stages(dev_dir: Path, checkpoint_root: Path, references: dict, out_dir: Path, metric: str, display: str):
    """Best dev score per section, one cumulative figure per section, with the zero-shot rerankers we ran."""
    groups = {OURS_GROUP: [], ZERO_SHOT_GROUP: []}
    for key, label in REFERENCES.items():
        if key in references and key != "splade":
            groups[ZERO_SHOT_GROUP].append((label, references[key][metric], REFERENCE_FILL, MUTED))
    baseline = ("SPLADE-v3, no rerank", references["splade"][metric]) if "splade" in references else None
    for file_name, stage, systems in DEV_STAGES:
        for system in systems:
            group, row = system_row(system, *system_score(system, dev_dir, checkpoint_root, metric))
            groups[group].append(row)
        dot_plot(groups, out_dir / file_name, f"Best dev {display} up to {stage} (rag24-dev, SPLADE-v3 top-100)",
                 display, baseline)


def plot_test(summary_path: Path, checkpoint_root: Path, path: Path, metric: str, display: str):
    """RAG25 test: the dev-selected system of each section, zero-shot rerankers we ran, and overview runs."""
    summary = json.loads(summary_path.read_text())
    groups = {OURS_GROUP: [], ZERO_SHOT_GROUP: [], OVERVIEW_GROUP: []}
    for key, scores in summary.items():
        if key in OVERVIEW_LABELS:
            label = OVERVIEW_LABELS[key].removeprefix("overview: ")
            groups[OVERVIEW_GROUP].append((label, scores[metric], OVERVIEW_FILL, INK_SECONDARY))
        elif key in REFERENCES and key != "splade":
            groups[ZERO_SHOT_GROUP].append((REFERENCES[key], scores[metric], REFERENCE_FILL, MUTED))
    for system in SYSTEMS:
        group, row = system_row(system, *system_score(system, summary_path.parent.parent, checkpoint_root, metric))
        groups[group].append(row)
    dot_plot(groups, path, f"RAG25 test {display} (22 NIST-judged topics)", display)


@app.command()
def main(
    run_dir: list[Path] = typer.Option(..., help="checkpoint directory per run, repeatable"),
    dev_dir: Path = typer.Option(Path("outputs/llm_reranking/rag24-dev")),
    test_summary: Path = typer.Option(Path("outputs/llm_reranking/rag25/splade@100/summary.json")),
    checkpoint_root: Path = typer.Option(Path("checkpoints/reranker")),
    out_dir: Path = typer.Option(Path("studies/llm_reranking/figures")),
    metric: str = "ndcg@30",
    overlay_runs: int = typer.Option(0, help="overlay only the first n runs, 0 = all"),
):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [load_run(path, dev_dir, SERIES_COLORS[index]) for index, path in enumerate(run_dir)]
    display = metric.replace("ndcg", "nDCG")
    references = load_references(dev_dir)
    for run in runs:
        plot_run_loss(run, out_dir / f"{run['name']}_loss.png")
        if run["dev_points"]:
            plot_run_metric(run, references, out_dir / f"{run['name']}_dev.png", metric, display)
    plot_overlay(runs[:overlay_runs or None], references, out_dir / "dev_curves.png", metric, display)
    plot_dev_stages(dev_dir, checkpoint_root, references, out_dir, metric, display)
    if test_summary.exists():
        plot_test(test_summary, checkpoint_root, out_dir / "rag25_test.png", metric, display)
    typer.echo(f"figures -> {out_dir}")


if __name__ == "__main__":
    app()
