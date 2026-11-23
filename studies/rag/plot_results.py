"""RAG figures in the style of the LLM reranking study: best dev strict vital after each section, and the RAG25 test.

Dev figures are cumulative: each adds the best context of its section to the previous one.
Scores are read from outputs/rag/<topic set>/Qwen3.5-9B/<experiment>/summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from studies.llm_reranking.plot_results import INK, INK_SECONDARY, MUTED, OVERVIEW_FILL, REFERENCE_FILL, SERIES_COLORS, dot_plot

app = typer.Typer(add_completion=False)
OURS_GROUP, REFERENCE_GROUP, ORGANIZERS_GROUP = ("Ours (reranked contexts)", "SPLADE-v3 context, no reranking",
                                                 "Organizers' RankQwen3-32B context* (*our generator and judge)")
# (experiment, condition) -> label; conditions are summary keys without the generator prefix
DEV_STAGES = [  # file, stage, rows added by the section
    ("dev_best_context_quality.png", "context quality",
     [("dev_context_runs", "ours-top20", "ours, top 20"), ("dev_context_runs", "ours_listwise_rrf-top20", "RRF, top 20")]),
    ("dev_best_context_size.png", "context size", [("dev_context_diversity", "ours-top50", "ours, top 50")]),
    ("dev_best_subquestions.png", "sub-question retrieval",
     [("dev_subquestions", "subq_roundrobin-top50", "sub-questions (round-robin), top 50")]),
]
DEV_REFERENCE = ("dev_context_runs", "splade-top20", "SPLADE-v3 context, top 20")
TEST_ROWS = [  # experiment, condition, label, group
    ("context_runs", "splade-top20", "SPLADE-v3, top 20", REFERENCE_GROUP),
    ("test_verification", "ours-top20", "ours, top 20", OURS_GROUP),
    ("context_runs", "ours_listwise_rrf-top20", "RRF, top 20", OURS_GROUP),
    ("test_verification", "ours-top50", "ours, top 50 (dev choice)", OURS_GROUP),
    ("test_verification", "ours_on_rankqwen3-top50", "ours over organizer candidates, top 50", OURS_GROUP),
    ("test_verification", "rankqwen3_32b-top20", "organizers' RankQwen3-32B*, top 20", ORGANIZERS_GROUP),
    ("test_verification", "rankqwen3_32b-top50", "organizers' RankQwen3-32B*, top 50", ORGANIZERS_GROUP),
]
STYLE = {OURS_GROUP: (SERIES_COLORS[0], INK), REFERENCE_GROUP: (REFERENCE_FILL, MUTED), ORGANIZERS_GROUP: (OVERVIEW_FILL, INK_SECONDARY)}


def score(rag_dir: Path, topic_set: str, experiment: str, condition: str, metric: str) -> float:
    summary = json.loads((rag_dir / topic_set / "Qwen3.5-9B" / experiment / "summary.json").read_text())
    return summary[f"Qwen3.5-9B-{condition}"][metric]


@app.command()
def main(
    rag_dir: Path = Path("outputs/rag"),
    out_dir: Path = Path("studies/rag/figures"),
    metric: str = "strict_vital",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    display = metric.replace("_", " ")
    experiment, condition, label = DEV_REFERENCE
    baseline = (label, score(rag_dir, "rag24-dev", experiment, condition, metric))
    groups = {OURS_GROUP: []}
    for file_name, stage, rows in DEV_STAGES:
        for experiment, condition, label in rows:
            groups[OURS_GROUP].append((label, score(rag_dir, "rag24-dev", experiment, condition, metric), *STYLE[OURS_GROUP]))
        dot_plot(groups, out_dir / file_name, f"Best dev {display} up to {stage} (RAG24 dev, Qwen3.5-9B)", display, baseline)

    groups = {OURS_GROUP: [], REFERENCE_GROUP: [], ORGANIZERS_GROUP: []}
    for experiment, condition, label, group in TEST_ROWS:
        groups[group].append((label, score(rag_dir, "rag25", experiment, condition, metric), *STYLE[group]))
    dot_plot(groups, out_dir / "rag25_test.png", f"RAG25 test {display} (22 topics, Qwen3.5-9B, Qwen3.5-35B-A3B judge)", display)
    typer.echo(f"figures -> {out_dir}")


if __name__ == "__main__":
    app()
