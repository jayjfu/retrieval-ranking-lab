"""Validation loss for saved checkpoints of a finished run. Writes <run>/dev_log.jsonl, same format as train.py."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
import typer

from retrieval_ranking_lab.rerank.llm_reranker import PromptEncoder, load_causal_lm
from studies.llm_reranking.train import load_validation_groups, validation_loss

app = typer.Typer(add_completion=False)


@app.command()
def main(run_dir: Path = typer.Option(...), dev_data: Path = typer.Option(...)):
    config = json.loads((run_dir / "train_config.json").read_text())
    groups = load_validation_groups(dev_data)
    checkpoints = [(0, None)]
    checkpoints += sorted((int(path.name.split("-")[1]), path) for path in run_dir.glob("step-*"))
    checkpoints.append((config["total_steps"], run_dir))
    with open(run_dir / "dev_log.jsonl", "w") as log_file:
        for step, adapter_path in checkpoints:
            model, tokenizer, _ = load_causal_lm(config["base_model"], str(adapter_path) if adapter_path else None)
            encoder = PromptEncoder(tokenizer, max_length=config["max_length"])
            entry = {"step": step, **validation_loss(model, encoder, groups, config["loss_type"])}
            log_file.write(json.dumps(entry) + "\n")
            typer.echo(entry)
            del model
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    app()
