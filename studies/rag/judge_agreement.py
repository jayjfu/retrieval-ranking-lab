"""Judge check: our nugget judge against the NIST assessors' nugget assignments on RAG24 (45 runs, 56 topics).

Re-judges every assessed answer with the same prompt and batching as run_study.py, then reports label agreement,
answer-level and run-level agreement of strict vital scores, and the judge's bias.
Writes outputs/rag/judge_agreement/<judge>.json (+ labels .jsonl).
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import typer
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import cohen_kappa_score

from retrieval_ranking_lab.data.trec_rag import DEFAULT_DATA_DIR
from retrieval_ranking_lab.generators import make_generator
from studies.rag.nugget_eval import NUGGET_BATCH_SIZE, nugget_prompts, nugget_scores, parse_nugget_labels

app = typer.Typer(add_completion=False)


def run_level(rows: list[dict], key: str) -> dict[str, float]:
    by_run: dict[str, list[float]] = {}
    for row in rows:
        by_run.setdefault(row["run_id"], []).append(row[key])
    return {run_id: statistics.fmean(values) for run_id, values in by_run.items()}


@app.command()
def main(
    judge: str = typer.Option("vllm:Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"),
    gpu_memory_utilization: float = 0.90,
    data_dir: Path = DEFAULT_DATA_DIR,
    out_dir: Path = Path("outputs/rag/judge_agreement"),
):
    rows = [json.loads(line) for line in open(data_dir / "rag24_nuggets_assignments.jsonl")]
    jobs = []
    for row_index, row in enumerate(rows):
        nuggets = [{"text": nugget["text"].strip(), "importance": nugget["importance"]} for nugget in row["nuggets"]]
        for batch_index, prompt in enumerate(nugget_prompts(row["query"], row["answer_text"], nuggets)):
            jobs.append((row_index, batch_index, prompt))
    model = make_generator(judge, max_tokens=128, max_model_len=8192, gpu_memory_utilization=gpu_memory_utilization)
    outputs = model.generate([job[2] for job in jobs])

    texts: dict[int, dict[int, str]] = {}
    for (row_index, batch_index, _), text in zip(jobs, outputs):
        texts.setdefault(row_index, {})[batch_index] = text
    official_labels, judge_labels, scored = [], [], []
    for row_index, row in enumerate(rows):
        nuggets = row["nuggets"]
        labels = []
        for batch_index in range(-(-len(nuggets) // NUGGET_BATCH_SIZE)):
            count = len(nuggets[batch_index * NUGGET_BATCH_SIZE : (batch_index + 1) * NUGGET_BATCH_SIZE])
            labels += parse_nugget_labels(texts.get(row_index, {}).get(batch_index, ""), count)
        official = [nugget["assignment"] for nugget in nuggets]
        official_labels += official
        judge_labels += labels
        scored.append({"run_id": row["run_id"], "qid": row["qid"], "labels": labels,
                       "official": nugget_scores(nuggets, official)["strict_vital"],
                       "judge": nugget_scores(nuggets, labels)["strict_vital"]})

    scored = [row for row in scored if row["official"] == row["official"]]  # topics with vital nuggets
    official_runs, judge_runs = run_level(scored, "official"), run_level(scored, "judge")
    run_ids = sorted(official_runs)
    report = {
        "judge": judge, "answers": len(scored), "runs": len(run_ids), "nugget_labels": len(official_labels),
        "label_accuracy": sum(first == second for first, second in zip(official_labels, judge_labels)) / len(official_labels),
        "label_kappa": cohen_kappa_score(official_labels, judge_labels),
        "answer_pearson": pearsonr([row["official"] for row in scored], [row["judge"] for row in scored])[0],
        "run_kendall_tau": kendalltau([official_runs[run_id] for run_id in run_ids], [judge_runs[run_id] for run_id in run_ids])[0],
        "run_pearson": pearsonr([official_runs[run_id] for run_id in run_ids], [judge_runs[run_id] for run_id in run_ids])[0],
        "mean_strict_vital_official": statistics.fmean(row["official"] for row in scored),
        "mean_strict_vital_judge": statistics.fmean(row["judge"] for row in scored),
        "label_counts_official": {label: official_labels.count(label) for label in sorted(set(official_labels))},
        "label_counts_judge": {label: judge_labels.count(label) for label in sorted(set(judge_labels))},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = judge.split("/")[-1]
    (out_dir / f"{tag}.json").write_text(json.dumps(report, indent=2))
    with open(out_dir / f"{tag}_labels.jsonl", "w") as file:
        file.writelines(json.dumps(row) + "\n" for row in scored)
    typer.echo(json.dumps(report, indent=2))


if __name__ == "__main__":
    app()
