"""Full-data sweep figures: dev curves against the best config, training diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import typer

from studies.llm_reranking.plot_results import INK, INK_SECONDARY, MUTED, SERIES_COLORS, SURFACE, load_run, save, smooth, style

app = typer.Typer(add_completion=False)
PREFIX = "listwise_lr1e-4_pos32_epoch1_"
BEST = PREFIX + "cosine_clip1"
OPTIONS = ["wsd_clip1", "cosine_clip1_loraplus4", "cosine_clip1_beta2_0.95_wd0.01", "cosine_clip10", "cosine_noclip",
           "cosine_clip1_temperature2", "cosine_clip1_smoothing0.1", "cosine_clip1_softcap20", "cosine_clip1_zloss1e-3",
           "cosine_clip1_ema0.99", "cosine_noclip_zloss1e-3", "cosine_noclip_zloss1e-3_beta2_0.95_wd0.01"]
DIAGNOSTIC_OPTIONS = ["cosine_clip1_temperature2", "cosine_clip1_zloss1e-3", "cosine_noclip", "cosine_clip1_smoothing0.1"]
METRIC = "ndcg@30"


def dev_curve(run: dict) -> list[tuple[int, float]]:
    return [(step, scores[METRIC]) for step, scores in run["dev_points"] if step > 0]


def plot_curves(runs: dict, path: Path):
    """One panel per option, with the best config behind it."""
    best_curve = dev_curve(runs[BEST])
    figure, axes = plt.subplots(6, 2, figsize=(11, 14.4), sharex=True, sharey=True)
    figure.patch.set_facecolor(SURFACE)
    for axis, option in zip(axes.flat, OPTIONS):
        axis.plot([step for step, _ in best_curve], [value for _, value in best_curve], color=MUTED, linewidth=1.6,
                  linestyle=(0, (4, 2)), label=f"{BEST} (best)")
        curve = dev_curve(runs[PREFIX + option])
        axis.plot([step for step, _ in curve], [value for _, value in curve], color=SERIES_COLORS[0], linewidth=2,
                  marker="o", markersize=4, markeredgecolor=SURFACE, label="panel config")
        style(axis, PREFIX + option, "", "")
        axis.title.set_fontsize(8.5)
    for axis in axes[-1]:
        axis.set_xlabel("training step", color=INK_SECONDARY)
    for axis in axes[:, 0]:
        axis.set_ylabel("dev nDCG@30", color=INK_SECONDARY)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, labelcolor=INK_SECONDARY, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.02))
    figure.suptitle("Full data: dev nDCG@30 by checkpoint", x=0.01, y=1.01, ha="left", color=INK, fontsize=10)
    save(figure, path)


def plot_diagnostics(runs: dict, path: Path):
    """Gradient norm and largest |margin| during training, smoothed."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    figure.patch.set_facecolor(SURFACE)
    panels = [("grad_norm", "gradient norm (before clipping)"), ("margin_abs_max", "largest |yes − no margin|")]
    for axis, (key, display) in zip(axes, panels):
        for color, option in zip(SERIES_COLORS, DIAGNOSTIC_OPTIONS):
            log = runs[PREFIX + option]["train_log"]
            axis.plot([entry["step"] for entry in log], smooth([entry[key] for entry in log]), color=color,
                      linewidth=2, label=PREFIX + option)
        style(axis, display, "training step", "")
        axis.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, labelcolor=INK_SECONDARY, loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.14), fontsize=8)
    figure.suptitle("Full data: training diagnostics (smoothed)", x=0.01, ha="left", color=INK, fontsize=10)
    save(figure, path)


@app.command()
def main(
    checkpoint_dir: Path = typer.Option(Path("checkpoints/reranker")),
    dev_dir: Path = typer.Option(Path("outputs/llm_reranking/rag24-dev")),
    out_dir: Path = typer.Option(Path("studies/llm_reranking/figures")),
):
    names = [PREFIX + option for option in OPTIONS] + [BEST]
    runs = {name: load_run(checkpoint_dir / name, dev_dir, SERIES_COLORS[0]) for name in names}
    plot_curves(runs, out_dir / "full_data_curves.png")
    plot_diagnostics(runs, out_dir / "full_data_diagnostics.png")
    typer.echo(f"figures -> {out_dir}")


if __name__ == "__main__":
    app()
