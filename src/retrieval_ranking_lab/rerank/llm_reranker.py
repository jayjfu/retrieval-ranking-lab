"""Pointwise LLM reranker: score = P("yes") after a yes/no relevance prompt.

Uses the Qwen3-Reranker template, so Qwen3-Reranker works zero-shot and any ChatML causal LM
can be fine-tuned with the same head. Training reuses PromptEncoder and yes_no_logits.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from retrieval_ranking_lab.rerank.base import Reranker

DEFAULT_LLM_RERANKER = "Qwen/Qwen3-Reranker-4B"
DEFAULT_INSTRUCTION = "Given a user's information need, retrieve passages that answer part of it"

PROMPT_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_content(query: str, doc: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class PromptEncoder:
    """Left-padded prefix + content + suffix. Truncation only cuts the document end."""

    def __init__(self, tokenizer, max_length: int = 512, instruction: str = DEFAULT_INSTRUCTION):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)
        self.yes_id = tokenizer.convert_tokens_to_ids("yes")
        self.no_id = tokenizer.convert_tokens_to_ids("no")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, pairs: list[tuple[str, str]]) -> dict[str, torch.Tensor]:
        content_budget = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        contents = [format_content(query, doc, self.instruction) for query, doc in pairs]
        content_ids = self.tokenizer(contents, add_special_tokens=False, truncation=True,
                                     max_length=content_budget)["input_ids"]
        sequences = [self.prefix_ids + ids + self.suffix_ids for ids in content_ids]
        max_len = max(map(len, sequences))
        input_ids = torch.full((len(sequences), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, sequence in enumerate(sequences):
            input_ids[row, max_len - len(sequence):] = torch.tensor(sequence)
            attention_mask[row, max_len - len(sequence):] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def yes_no_logits(model, batch: dict[str, torch.Tensor], yes_id: int, no_id: int) -> torch.Tensor:
    """[batch, 2] logits (no, yes) at the last position."""
    try:
        output = model(**batch, logits_to_keep=1)  # skip the full [batch, len, vocab] logits
    except TypeError:
        output = model(**batch)
    last = output.logits[:, -1, :]
    return torch.stack([last[:, no_id], last[:, yes_id]], dim=1).float()


def load_causal_lm(model_name: str | None, adapter_path: str | None = None, device: str = "cuda",
                   dtype=torch.bfloat16, merge_adapter: bool = True):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter_path and model_name is None:
        adapter_config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
        model_name = adapter_config["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, padding_side="left")
    try:
        import flash_attn  # noqa: F401
        attention = "flash_attention_2"
    except ImportError:
        attention = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation=attention).to(device)
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_adapter:
            model = model.merge_and_unload()
    model.eval()
    return model, tokenizer, model_name


class LLMReranker(Reranker):
    name = "llm"

    def __init__(self, model_name: str | None = DEFAULT_LLM_RERANKER, adapter_path: str | None = None,
                 batch_size: int = 16, max_length: int = 512, instruction: str = DEFAULT_INSTRUCTION,
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, tokenizer, self.model_name = load_causal_lm(model_name, adapter_path, self.device)
        self.encoder = PromptEncoder(tokenizer, max_length=max_length, instruction=instruction)
        self.batch_size = batch_size
        self.name = f"llm:{Path(adapter_path).name if adapter_path else self.model_name.split('/')[-1]}"

    @torch.no_grad()
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))  # less padding
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            batch = {name: tensor.to(self.device)
                     for name, tensor in self.encoder([pairs[i] for i in batch_indices]).items()}
            logits = yes_no_logits(self.model, batch, self.encoder.yes_id, self.encoder.no_id)
            yes_probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
            for pair_index, probability in zip(batch_indices, yes_probs):
                scores[pair_index] = probability
        return scores
