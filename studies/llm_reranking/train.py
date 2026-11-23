"""LoRA fine-tuning of a causal LM as a yes/no reranker.

Score: margin = logit("yes") - logit("no") at the last token.
Losses: listwise = softmax CE over 1 positive + (group_size - 1) negatives (LCE, RankLLaMA);
pointwise = yes/no classification per pair.
Stability options act on training only: margin soft-cap and temperature are monotonic, so ranking is unchanged.
Validation loss always uses raw margins, so it is comparable across runs.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from tqdm.auto import tqdm

from retrieval_ranking_lab.rerank.llm_reranker import PromptEncoder, yes_no_logits

app = typer.Typer(add_completion=False)


@dataclass
class LossOptions:
    loss_type: str = "listwise"
    score_zloss: float = 0.0  # weight of logsumexp(scores)^2, keeps margins near zero
    margin_softcap: float = 0.0  # scores = cap * tanh(margin / cap); 0 = off
    temperature: float = 1.0
    label_smoothing: float = 0.0


def sample_group_docs(group: dict, group_size: int, rng: random.Random, all_groups: list[dict],
                      extra_per_group: int = 0) -> list[tuple[str, str | None]]:
    """(text, doc id) pairs, positive first, then up to extra_per_group extra (unjudged) negatives, then judged ones.
    Tops up with other queries' negatives if short; those get no id.
    Index sampling consumes the RNG exactly like sampling the texts."""
    positive_ids = group.get("positive_ids") or [None] * len(group["positives"])
    negative_ids = group.get("negative_ids") or [None] * len(group["negatives"])
    positive = rng.choice(range(len(group["positives"])))
    docs = [(group["positives"][positive], positive_ids[positive])]
    if extra_per_group > 0:
        extras = list(range(len(group["extra_negatives"])))
        rng.shuffle(extras)
        docs += [(group["extra_negatives"][index], group["extra_negative_ids"][index])
                 for index in extras[:extra_per_group]]
    negatives = list(range(len(group["negatives"])))
    rng.shuffle(negatives)
    docs += [(group["negatives"][index], negative_ids[index]) for index in negatives]
    while len(docs) < group_size:
        other = rng.choice(all_groups)
        docs.append((rng.choice(other["negatives"]), None))
    return docs[:group_size]


def load_teacher_scores(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    scores = {}
    for line in open(path):
        row = json.loads(line)
        scores[(row["query_id"], row["doc_id"])] = row["score"]
    return scores


def distillation_loss(margins: torch.Tensor, teacher: torch.Tensor, temperature: float) -> torch.Tensor:
    """tau^2 * KL(softmax(teacher / tau) || softmax(student / tau)) over each group (Hinton et al.)."""
    student = F.log_softmax(margins / temperature, dim=1)
    target = F.softmax(teacher / temperature, dim=1)
    return F.kl_div(student, target, reduction="batchmean") * temperature**2


def compute_loss(logits: torch.Tensor, num_queries: int, group_size: int, options: LossOptions):
    """logits: [num_queries * group_size, 2] (no, yes), positive first in each group.
    Returns (loss, accuracy, margin statistics)."""
    margins = (logits[:, 1] - logits[:, 0]).view(num_queries, group_size)
    scores = margins
    if options.margin_softcap > 0:
        scores = options.margin_softcap * torch.tanh(scores / options.margin_softcap)
    scores = scores / options.temperature
    if options.loss_type == "listwise":
        target = torch.zeros(num_queries, dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scores, target, label_smoothing=options.label_smoothing)
        partition = torch.logsumexp(scores, dim=1)
        accuracy = (scores.argmax(1) == 0).float().mean()
    else:  # pointwise: softmax over (no, yes) is shift-invariant, so (0, score) gives the same loss
        pair_logits = torch.stack([torch.zeros_like(scores), scores], dim=-1).view(-1, 2)
        labels = torch.zeros(num_queries, group_size, dtype=torch.long, device=scores.device)
        labels[:, 0] = 1
        labels = labels.view(-1)
        loss = F.cross_entropy(pair_logits, labels, label_smoothing=options.label_smoothing)
        partition = torch.logsumexp(pair_logits, dim=1)
        accuracy = (pair_logits.argmax(1) == labels).float().mean()
    if options.score_zloss > 0:
        loss = loss + options.score_zloss * partition.pow(2).mean()
    with torch.no_grad():
        stats = {"margin_mean": margins.mean().item(), "margin_abs_max": margins.abs().max().item(),
                 "positive_gap": (margins[:, 0] - margins[:, 1:].max(dim=1).values).mean().item()}
    return loss, accuracy, stats


@torch.no_grad()
def validation_loss(model, encoder: PromptEncoder, groups: list[dict], loss_type: str, batch_groups: int = 8) -> dict:
    """Mean raw loss and accuracy on frozen groups (first positive + stored negatives, fixed order)."""
    was_training = model.training
    model.eval()
    total_loss, total_accuracy, count = 0.0, 0.0, 0
    group_size = 1 + len(groups[0]["negatives"])
    raw = LossOptions(loss_type=loss_type)
    for start in range(0, len(groups), batch_groups):
        batch = groups[start : start + batch_groups]
        pairs = [(group["query"], doc) for group in batch for doc in [group["positives"][0]] + group["negatives"]]
        inputs = {name: tensor.cuda() for name, tensor in encoder(pairs).items()}
        logits = yes_no_logits(model, inputs, encoder.yes_id, encoder.no_id)
        loss, accuracy, _ = compute_loss(logits, len(batch), group_size, raw)
        total_loss += loss.item() * len(batch)
        total_accuracy += accuracy.item() * len(batch)
        count += len(batch)
    if was_training:
        model.train()
    return {"dev_loss": total_loss / count, "dev_accuracy": total_accuracy / count}


def load_validation_groups(path: Path | None) -> list[dict]:
    if path is None:
        return []
    groups = [json.loads(line) for line in open(path)]
    size = 1 + len(groups[0]["negatives"])
    return [group for group in groups if 1 + len(group["negatives"]) == size]  # equal group sizes for batching


def make_schedule(optimizer, schedule: str, warmup_steps: int, total_steps: int, decay_ratio: float):
    """cosine: warmup then cosine to 0. wsd: warmup, constant, then linear decay to 0 over the last decay_ratio."""
    decay_steps = max(1, int(decay_ratio * total_steps))

    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if schedule == "wsd":
            return 1.0 if step < total_steps - decay_steps else max(0.0, (total_steps - step) / decay_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class WeightAverage:
    """Exponential moving average of the trainable (LoRA) weights."""

    def __init__(self, model, decay: float):
        self.decay = decay
        self.parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.shadow = {name: parameter.detach().float().clone() for name, parameter in self.parameters.items()}

    @torch.no_grad()
    def update(self):
        for name, parameter in self.parameters.items():
            self.shadow[name].mul_(self.decay).add_(parameter.detach().float(), alpha=1 - self.decay)

    @contextlib.contextmanager
    def applied(self):
        """Temporarily load the averaged weights (for validation and saving)."""
        backup = {name: parameter.detach().clone() for name, parameter in self.parameters.items()}
        with torch.no_grad():
            for name, parameter in self.parameters.items():
                parameter.copy_(self.shadow[name].to(parameter.dtype))
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(backup[name])


@app.command()
def main(
    data: Path = typer.Option(...),
    base_model: str = "Qwen/Qwen3.5-0.8B",
    out: Path = typer.Option(..., help="checkpoints/reranker/<loss>_lr<lr>_pos<positives>_epoch<epochs>[_<schedule>_<clip>[_<option>]]"),
    loss_type: str = typer.Option("listwise", help="listwise | pointwise"),
    group_size: int = 8,
    queries_per_step: int = 8,
    grad_accum: int = 2,
    epochs: float = 1.0,
    learning_rate: float = 1e-4,
    warmup_ratio: float = 0.05,
    schedule: str = typer.Option("cosine", help="cosine | wsd"),
    decay_ratio: float = typer.Option(0.2, help="wsd: share of steps in the final linear decay"),
    adam_beta2: float = 0.999,
    weight_decay: float = 0.0,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.05,
    rslora: bool = typer.Option(False, help="rank-stabilized LoRA scaling alpha / sqrt(rank)"),
    lora_plus_ratio: float = typer.Option(1.0, help="LoRA+: learning-rate multiplier for B matrices"),
    score_zloss: float = 0.0,
    margin_softcap: float = 0.0,
    temperature: float = 1.0,
    label_smoothing: float = 0.0,
    ema_decay: float = typer.Option(0.0, help="EMA of LoRA weights; checkpoints store the EMA; 0 = off"),
    max_grad_norm: float = typer.Option(1.0, help="gradient clipping threshold; 0 = off (norm still logged)"),
    max_length: int = 512,
    max_queries: int = typer.Option(-1, help="-1 for all"),
    log_every: int = 10,
    save_every: int = typer.Option(0, help="also save out/step-<n> every n steps, 0 = final only"),
    dev_data: Path = typer.Option(None, help="frozen validation groups; loss logged at step 0, each save, end"),
    teacher_scores: Path = typer.Option(None, help="teacher score cache jsonl (query_id, doc_id, score); groups need ids"),
    distill_weight: float = typer.Option(0.0, help="1 - alpha: weight of the KL term; listwise CE gets 1 - this"),
    distill_temperature: float = typer.Option(1.0, help="tau for teacher and student softmax in the KL term"),
    extra_negatives_per_group: int = typer.Option(0, help="unjudged negatives per group from augment_groups.py"),
    seed: int = 13,
):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    rng = random.Random(seed)
    groups = [json.loads(line) for line in open(data)]
    if max_queries > 0:
        groups = groups[:max_queries]
    loss_options = LossOptions(loss_type, score_zloss, margin_softcap, temperature, label_smoothing)
    teacher = load_teacher_scores(teacher_scores)
    if distill_weight > 0 and not teacher:
        raise typer.BadParameter("--distill-weight needs --teacher-scores")

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
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, use_rslora=rslora,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    parameter_groups = [
        {"params": [parameter for name, parameter in trainable if "lora_B" not in name], "lr": learning_rate},
        {"params": [parameter for name, parameter in trainable if "lora_B" in name], "lr": learning_rate * lora_plus_ratio},
    ]
    optimizer = torch.optim.AdamW(parameter_groups, lr=learning_rate, betas=(0.9, adam_beta2),
                                  weight_decay=weight_decay)
    total_steps = max(1, int(math.ceil(len(groups) / (queries_per_step * grad_accum)) * epochs))
    scheduler = make_schedule(optimizer, schedule, int(warmup_ratio * total_steps), total_steps, decay_ratio)
    weight_average = WeightAverage(model, ema_decay) if ema_decay > 0 else None

    out.mkdir(parents=True, exist_ok=True)
    config = {"data": str(data), "base_model": base_model, "group_size": group_size,
              "queries_per_step": queries_per_step, "grad_accum": grad_accum, "epochs": epochs,
              "learning_rate": learning_rate, "warmup_ratio": warmup_ratio, "schedule": schedule,
              "decay_ratio": decay_ratio, "adam_beta2": adam_beta2, "weight_decay": weight_decay,
              "lora_rank": lora_rank, "lora_alpha": lora_alpha, "rslora": rslora, "lora_plus_ratio": lora_plus_ratio,
              "ema_decay": ema_decay, "max_grad_norm": max_grad_norm, "max_length": max_length, "seed": seed, "total_steps": total_steps,
              "num_train_queries": len(groups), "teacher_scores": str(teacher_scores) if teacher_scores else None,
              "distill_weight": distill_weight, "distill_temperature": distill_temperature,
              "extra_negatives_per_group": extra_negatives_per_group, **asdict(loss_options)}
    (out / "train_config.json").write_text(json.dumps(config, indent=2))

    dev_groups = load_validation_groups(dev_data)
    dev_log = open(out / "dev_log.jsonl", "w") if dev_groups else None

    def averaged():
        return weight_average.applied() if weight_average else contextlib.nullcontext()

    def log_validation(step: int):
        if dev_log:
            with averaged():
                entry = {"step": step, **validation_loss(model, encoder, dev_groups, loss_type)}
            dev_log.write(json.dumps(entry) + "\n")
            dev_log.flush()

    def save(path: Path):
        with averaged():
            model.save_pretrained(path)
        tokenizer.save_pretrained(path)

    log_validation(0)
    model.train()
    query_order: list[int] = []
    log_file = open(out / "train_log.jsonl", "w")
    progress = tqdm(range(total_steps), desc="train")
    start_time = time.time()
    window: dict[str, list[float]] = {}
    for step in progress:
        for _ in range(grad_accum):
            if len(query_order) < queries_per_step:
                query_order += rng.sample(range(len(groups)), len(groups))
            batch_groups = [groups[query_order.pop()] for _ in range(queries_per_step)]
            sampled = [sample_group_docs(group, group_size, rng, groups, extra_negatives_per_group) for group in batch_groups]
            pairs = [(group["query"], text) for group, docs in zip(batch_groups, sampled) for text, _ in docs]
            batch = {name: tensor.cuda() for name, tensor in encoder(pairs).items()}
            logits = yes_no_logits(model, batch, encoder.yes_id, encoder.no_id)
            loss, accuracy, stats = compute_loss(logits, queries_per_step, group_size, loss_options)
            if distill_weight > 0:
                rows = []
                for group, docs in zip(batch_groups, sampled):
                    known = [teacher.get((group["query_id"], doc_id)) for _, doc_id in docs]
                    floor = min(value for value in known if value is not None)  # other-topic negatives: lowest score
                    rows.append([floor if value is None else value for value in known])
                teacher_tensor = torch.tensor(rows, dtype=torch.float32, device=logits.device)
                margins = (logits[:, 1] - logits[:, 0]).view(queries_per_step, group_size)
                distill = distillation_loss(margins, teacher_tensor, distill_temperature)
                loss = (1 - distill_weight) * loss + distill_weight * distill
                stats["distill"] = distill.item()
            (loss / grad_accum).backward()
            for name, value in {"loss": loss.item(), "accuracy": accuracy.item(), **stats}.items():
                window.setdefault(name, []).append(value)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm or float("inf")).item()
        window.setdefault("grad_norm", []).append(grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if weight_average:
            weight_average.update()
        if (step + 1) % log_every == 0 or step + 1 == total_steps:
            entry = {"step": step + 1, **{name: sum(values) / len(values) for name, values in window.items()},
                     "grad_norm_max": max(window["grad_norm"]), "learning_rate": scheduler.get_last_lr()[0],
                     "elapsed_s": round(time.time() - start_time, 1)}
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            progress.set_postfix(loss=f"{entry['loss']:.4f}", grad_norm=f"{entry['grad_norm']:.2f}")
            window = {}
        if save_every and (step + 1) % save_every == 0 and step + 1 < total_steps:
            save(out / f"step-{step + 1}")
            log_validation(step + 1)

    log_validation(total_steps)
    save(out)
    typer.echo(f"saved LoRA adapter -> {out}")


if __name__ == "__main__":
    app()
