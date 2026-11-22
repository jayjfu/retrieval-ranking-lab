"""LoRA fine-tuning of a causal LM as a yes/no reranker.

Score: logit("yes") - logit("no") at the last token.
Losses: listwise = softmax CE over 1 positive + (group_size - 1) negatives (LCE, RankLLaMA);
pointwise = yes/no classification per pair.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from tqdm.auto import tqdm

from retrieval_ranking_lab.rerank.llm_reranker import PromptEncoder, yes_no_logits

app = typer.Typer(add_completion=False)


def sample_group_docs(group: dict, group_size: int, rng: random.Random, all_groups: list[dict]) -> list[str]:
    """Positive first, then negatives. Tops up with other queries' negatives if short."""
    positive = rng.choice(group["positives"])
    negatives = list(group["negatives"])
    rng.shuffle(negatives)
    while len(negatives) < group_size - 1:
        negatives.append(rng.choice(rng.choice(all_groups)["negatives"]))
    return [positive] + negatives[: group_size - 1]


def compute_loss(logits: torch.Tensor, loss_type: str, num_queries: int, group_size: int):
    """logits: [num_queries * group_size, 2] (no, yes). Returns (loss, accuracy)."""
    if loss_type == "listwise":
        scores = (logits[:, 1] - logits[:, 0]).view(num_queries, group_size)
        target = torch.zeros(num_queries, dtype=torch.long, device=scores.device)
        return F.cross_entropy(scores, target), (scores.argmax(1) == 0).float().mean()
    labels = torch.zeros(num_queries, group_size, dtype=torch.long, device=logits.device)
    labels[:, 0] = 1
    labels = labels.view(-1)
    return F.cross_entropy(logits, labels), (logits.argmax(1) == labels).float().mean()


@torch.no_grad()
def validation_loss(model, encoder: PromptEncoder, groups: list[dict], loss_type: str, batch_groups: int = 8) -> dict:
    """Mean loss and accuracy on frozen groups (first positive + stored negatives, fixed order)."""
    was_training = model.training
    model.eval()
    total_loss, total_accuracy, count = 0.0, 0.0, 0
    group_size = 1 + len(groups[0]["negatives"])
    for start in range(0, len(groups), batch_groups):
        batch_groups_ = groups[start : start + batch_groups]
        pairs = [(g["query"], doc) for g in batch_groups_ for doc in [g["positives"][0]] + g["negatives"]]
        batch = {name: tensor.cuda() for name, tensor in encoder(pairs).items()}
        logits = yes_no_logits(model, batch, encoder.yes_id, encoder.no_id)
        loss, accuracy = compute_loss(logits, loss_type, len(batch_groups_), group_size)
        total_loss += loss.item() * len(batch_groups_)
        total_accuracy += accuracy.item() * len(batch_groups_)
        count += len(batch_groups_)
    if was_training:
        model.train()
    return {"dev_loss": total_loss / count, "dev_accuracy": total_accuracy / count}


def load_validation_groups(path: Path | None) -> list[dict]:
    if path is None:
        return []
    groups = [json.loads(line) for line in open(path)]
    size = 1 + len(groups[0]["negatives"])
    return [g for g in groups if 1 + len(g["negatives"]) == size]  # equal group sizes for batching


@app.command()
def main(
    data: Path = typer.Option(...),
    base_model: str = "Qwen/Qwen3.5-0.8B",
    out: Path = typer.Option(Path("checkpoints/llm-reranker")),
    loss_type: str = typer.Option("listwise", help="listwise | pointwise"),
    group_size: int = 8,
    queries_per_step: int = 8,
    grad_accum: int = 2,
    epochs: float = 1.0,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    max_length: int = 512,
    max_queries: int = typer.Option(-1, help="-1 for all"),
    log_every: int = 10,
    save_every: int = typer.Option(0, help="also save out/step-<n> every n steps, 0 = final only"),
    dev_data: Path = typer.Option(None, help="frozen validation groups; loss logged at step 0, each save, end"),
    seed: int = 13,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    torch.manual_seed(seed)
    rng = random.Random(seed)
    groups = [json.loads(line) for line in open(data)]
    if max_queries > 0:
        groups = groups[:max_queries]

    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    encoder = PromptEncoder(tokenizer, max_length=max_length)
    try:
        import flash_attn  # noqa: F401
        attention = "flash_attention_2"
    except ImportError:
        attention = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, attn_implementation=attention).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    total_steps = max(1, int(math.ceil(len(groups) / (queries_per_step * grad_accum)) * epochs))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate,
                                  weight_decay=0.0)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)

    out.mkdir(parents=True, exist_ok=True)
    config = {"data": str(data), "base_model": base_model, "loss_type": loss_type, "group_size": group_size,
              "queries_per_step": queries_per_step, "grad_accum": grad_accum, "epochs": epochs,
              "learning_rate": learning_rate, "lora_rank": lora_rank, "lora_alpha": lora_alpha,
              "max_length": max_length, "seed": seed, "total_steps": total_steps, "num_train_queries": len(groups)}
    (out / "train_config.json").write_text(json.dumps(config, indent=2))

    dev_groups = load_validation_groups(dev_data)
    dev_log = open(out / "dev_log.jsonl", "w") if dev_groups else None

    def log_validation(step: int):
        if dev_log:
            dev_log.write(json.dumps({"step": step, **validation_loss(model, encoder, dev_groups, loss_type)}) + "\n")
            dev_log.flush()

    log_validation(0)
    model.train()
    query_order: list[int] = []
    log_file = open(out / "train_log.jsonl", "w")
    progress = tqdm(range(total_steps), desc="train")
    start_time, running_loss, running_accuracy, steps_since_log = time.time(), 0.0, 0.0, 0
    for step in progress:
        for _ in range(grad_accum):
            if len(query_order) < queries_per_step:
                query_order += rng.sample(range(len(groups)), len(groups))
            batch_groups = [groups[query_order.pop()] for _ in range(queries_per_step)]
            pairs = [(group["query"], doc)
                     for group in batch_groups for doc in sample_group_docs(group, group_size, rng, groups)]
            batch = {name: tensor.cuda() for name, tensor in encoder(pairs).items()}
            logits = yes_no_logits(model, batch, encoder.yes_id, encoder.no_id)
            loss, accuracy = compute_loss(logits, loss_type, queries_per_step, group_size)
            (loss / grad_accum).backward()
            running_loss += loss.item() / grad_accum
            running_accuracy += accuracy.item() / grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        steps_since_log += 1
        if steps_since_log == log_every or step + 1 == total_steps:
            entry = {"step": step + 1, "loss": running_loss / steps_since_log,
                     "accuracy": running_accuracy / steps_since_log, "learning_rate": scheduler.get_last_lr()[0],
                     "elapsed_s": round(time.time() - start_time, 1)}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            progress.set_postfix(loss=f"{entry['loss']:.4f}", accuracy=f"{entry['accuracy']:.3f}")
            running_loss, running_accuracy, steps_since_log = 0.0, 0.0, 0
        if save_every and (step + 1) % save_every == 0 and step + 1 < total_steps:
            model.save_pretrained(out / f"step-{step + 1}")
            tokenizer.save_pretrained(out / f"step-{step + 1}")
            log_validation(step + 1)

    log_validation(total_steps)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    typer.echo(f"saved LoRA adapter -> {out}")


if __name__ == "__main__":
    app()
